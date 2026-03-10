"""
src/predict.py  [Step C]
职责：字义预测主逻辑，处理三种情形

情形 1 — EXACT：声旁+部首均在库 → 直接返回
情形 2 — PREDICTED：声旁在库，部首不同 → 向量算术 + 近邻 + LLM
情形 3 — INFERRED：声旁不在库 → encode 声旁字本身 + 部首质心 + LLM

三部首情形下，情形 2 的锚点可能来自 1 或 2 个已知部首（如某声旁只有人部字，
或既有人部字又有心部字），多锚点时取加权平均。
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── 返回结构 ──────────────────────────────────────────────────

@dataclass
class PredictionResult:
    phonetic:    str
    radical:     str
    mode:        str          # EXACT | PREDICTED | INFERRED
    confidence:  str          # high | medium | low

    exact_char:     Optional[str] = None
    exact_shuowen:  Optional[str] = None
    exact_dazidian: Optional[str] = None

    anchor_chars:    list = field(default_factory=list)
    shift_direction: Optional[str] = None
    neighbors:       list = field(default_factory=list)
    predicted_vec:   Optional[np.ndarray] = None

    llm_prediction:  Optional[str] = None
    reasoning:       Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"声旁「{self.phonetic}」+ 部首「{self.radical}」",
            f"模式：{self.mode}  置信度：{self.confidence}",
        ]
        if self.mode == "EXACT":
            lines += [
                f"命中字：{self.exact_char}",
                f"說文：  {self.exact_shuowen}",
                f"大字典：{self.exact_dazidian}",
            ]
        else:
            if self.anchor_chars:
                anchors_str = " | ".join(
                    f"{a['character']}({a['radical']})" for a in self.anchor_chars
                )
                lines.append(f"锚点字：{anchors_str}")
            if self.shift_direction:
                lines.append(f"偏移路径：{self.shift_direction}")
            if self.neighbors:
                nb_str = " | ".join(
                    f"{n['character']}({n['radical']}) sim={n['similarity']:.3f}"
                    for n in self.neighbors[:3]
                )
                lines.append(f"语义近邻：{nb_str}")
            if self.llm_prediction:
                lines.append(f"\n🔮 预测字义：\n{self.llm_prediction}")
            if self.reasoning:
                lines.append(f"\n💡 推理依据：\n{self.reasoning}")
        return "\n".join(lines)


# ── 预测器 ────────────────────────────────────────────────────

class RadicalPredictor:

    def __init__(self, phonetic_index, shift_vectors: dict, llm_client=None):
        self.phonetic_index = phonetic_index
        self.shift_vectors  = shift_vectors
        self.llm_client     = llm_client

    def predict(
        self,
        phonetic: str,
        target_radical: str,
        top_k: int = 5,
        use_llm: bool = True,
    ) -> PredictionResult:
        # 情形 1
        if self.phonetic_index.has_exact(phonetic, target_radical):
            entry = self.phonetic_index.get_exact(phonetic, target_radical)
            return PredictionResult(
                phonetic=phonetic, radical=target_radical,
                mode="EXACT", confidence="high",
                exact_char=entry["character"],
                exact_shuowen=entry["shuowen"],
                exact_dazidian=entry["dazidian"],
            )
        # 情形 2
        if self.phonetic_index.has_phonetic(phonetic):
            return self._predict_by_shift(phonetic, target_radical, top_k, use_llm)
        # 情形 3
        return self._predict_fallback(phonetic, target_radical, top_k, use_llm)

    # ── 情形 2 ────────────────────────────────────────────────

    def _predict_by_shift(self, phonetic, target_radical, top_k, use_llm):
        anchors = {r: v for r, v in self.phonetic_index.get_anchors(phonetic).items()
                   if r != target_radical}

        usable = []
        for anchor_radical, entries in anchors.items():
            shift_key = (anchor_radical, target_radical)
            if shift_key in self.shift_vectors:
                entry = entries[0]
                usable.append({
                    "character": entry["character"],
                    "radical":   anchor_radical,
                    "shuowen":   entry["shuowen"],
                    "dazidian":  entry["dazidian"],
                    "vector":    self._get_vector(entry["character"]),
                    "shift":     self.shift_vectors[shift_key],
                })

        if not usable:
            # 有声旁但缺偏移向量（例如新部首还没建向量），降级
            return self._predict_fallback(phonetic, target_radical, top_k, use_llm)

        predicted_vecs = [a["vector"] + a["shift"] for a in usable]
        predicted_vec  = np.mean(predicted_vecs, axis=0)
        exclude        = [a["character"] for a in usable]

        # 近邻搜索：优先目标部首的字
        neighbors_target = self.phonetic_index.cosine_neighbors(
            predicted_vec, top_k=top_k,
            exclude_chars=exclude,
            filter_radical=target_radical,
        )
        # 补充其他部首近邻
        neighbors_other = self.phonetic_index.cosine_neighbors(
            predicted_vec, top_k=top_k,
            exclude_chars=exclude + [n["character"] for n in neighbors_target],
        )
        neighbors = (neighbors_target + neighbors_other)[:top_k]

        # 置信度：锚点越多、向量一致性越高置信度越高
        confidence = "high" if len(usable) >= 2 else "medium"

        result = PredictionResult(
            phonetic=phonetic, radical=target_radical,
            mode="PREDICTED", confidence=confidence,
            anchor_chars=usable,
            shift_direction=" & ".join(f"{a['radical']}→{target_radical}" for a in usable),
            neighbors=neighbors,
            predicted_vec=predicted_vec,
        )
        if use_llm and self.llm_client is not None:
            self._fill_llm(result)
        return result

    # ── 情形 3 ────────────────────────────────────────────────

    def _predict_fallback(self, phonetic, target_radical, top_k, use_llm):
        phonetic_vec  = self.phonetic_index.encode_phonetic_char(phonetic)
        radical_mean  = self._radical_mean_vector(target_radical)
        if radical_mean is not None:
            predicted_vec = 0.6 * phonetic_vec + 0.4 * radical_mean
        else:
            predicted_vec = phonetic_vec

        neighbors = self.phonetic_index.cosine_neighbors(predicted_vec, top_k=top_k)

        result = PredictionResult(
            phonetic=phonetic, radical=target_radical,
            mode="INFERRED", confidence="low",
            neighbors=neighbors, predicted_vec=predicted_vec,
        )
        if use_llm and self.llm_client is not None:
            self._fill_llm(result)
        return result

    # ── 辅助 ──────────────────────────────────────────────────

    def _get_vector(self, character: str) -> np.ndarray:
        entry = self.phonetic_index._find_entry_by_char(character)
        if entry is None:
            raise KeyError(f"'{character}' 不在索引中")
        return self.phonetic_index._all_vectors[entry["vector_idx"]]

    def _radical_mean_vector(self, radical: str) -> Optional[np.ndarray]:
        vecs = []
        for ph_dict in self.phonetic_index._index.values():
            for r, entries in ph_dict.items():
                if r == radical:
                    for e in entries:
                        vecs.append(self.phonetic_index._all_vectors[e["vector_idx"]])
        return np.mean(vecs, axis=0) if vecs else None

    def _fill_llm(self, result: PredictionResult):
        from src.llm_generate import generate_prediction
        out = generate_prediction(result, self.llm_client)
        result.llm_prediction = out.get("prediction", "")
        result.reasoning      = out.get("reasoning",  "")

    # ── 工厂 ──────────────────────────────────────────────────

    @classmethod
    def load(cls, use_llm: bool = True) -> "RadicalPredictor":
        from src.phonetic_index  import PhoneticIndex
        from src.radical_vectors import load_shift_vectors

        print("📦 加载声旁索引...")
        phonetic_index = PhoneticIndex.load()

        print("📐 加载偏移向量...")
        shift_vectors = load_shift_vectors()

        llm_client = None
        if use_llm:
            from src.llm_generate import build_client
            llm_client = build_client()

        return cls(phonetic_index, shift_vectors, llm_client)


# ── CLI ───────────────────────────────────────────────────────

def run():
    import argparse
    parser = argparse.ArgumentParser(description="部首语义预测")
    parser.add_argument("phonetic", help="声旁，如：童")
    parser.add_argument("radical",  help="目标部首，如：言部")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--top-k",  type=int, default=5)
    args = parser.parse_args()

    predictor = RadicalPredictor.load(use_llm=not args.no_llm)
    result    = predictor.predict(
        args.phonetic, args.radical,
        top_k=args.top_k, use_llm=not args.no_llm
    )
    print("\n" + "="*55)
    print(result.summary())
    print("="*55)


if __name__ == "__main__":
    run()
