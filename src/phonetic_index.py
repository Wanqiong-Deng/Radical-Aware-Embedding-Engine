"""
src/phonetic_index.py  [Step B]
职责：构建声旁倒排索引，支持快速检索「同声旁已知字」和余弦近邻搜索

索引结构：
  { phonetic → { radical → [entry, ...] } }

新增：对声旁字本身做 BGE encode（作为降级处理的语义锚）
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PhoneticIndex:

    def __init__(self):
        self._index: dict = defaultdict(lambda: defaultdict(list))
        self._all_vectors: Optional[np.ndarray] = None   # (N, 768)
        self._all_chars:   Optional[list]        = None
        self._phonetic_vecs: dict                = {}

    # ── 构建 ──────────────────────────────────────────────────

    def build(self, df_coords: pd.DataFrame, embed_index: dict):
        required = {"Character", "Radical", "Phonetic", "GroupID"}
        missing  = required - set(df_coords.columns)
        if missing:
            raise ValueError(f"coords.csv 缺少列：{missing}")

        vectors_list, chars_list = [], []

        for _, row in df_coords.iterrows():
            char     = str(row["Character"])
            radical  = str(row["Radical"])
            phonetic = str(row["Phonetic"])
            group_id = str(row["GroupID"])

            if char not in embed_index:
                continue

            info  = embed_index[char]
            entry = {
                "character":   char,
                "radical":     radical,
                "phonetic":    phonetic,
                "group_id":    group_id,
                "shuowen":     str(row.get("Shuowen_clean",  row.get("Shuowen",  ""))),
                "dazidian":    str(row.get("Dazidian_clean", row.get("Dazidian", ""))),
                "combined":    str(row.get("Combined", "")),
                "x":           float(row.get("x", 0)),
                "y":           float(row.get("y", 0)),
                "vector_idx":  len(chars_list),
            }
            self._index[phonetic][radical].append(entry)
            vectors_list.append(info["vector"])
            chars_list.append(char)

        self._all_vectors = np.array(vectors_list) if vectors_list else np.empty((0, 768))
        self._all_chars   = chars_list

        n_phonetics = len(self._index)
        n_entries   = sum(len(e) for ph in self._index.values() for e in ph.values())
        print(f"📚 声旁索引：{n_phonetics} 个声旁，{n_entries} 条字记录，"
              f"已知部首：{self.known_radicals()}")

    # ── 查询接口 ──────────────────────────────────────────────

    def get_exact(self, phonetic: str, radical: str) -> Optional[dict]:
        entries = self._index.get(phonetic, {}).get(radical, [])
        return entries[0] if entries else None

    def get_anchors(self, phonetic: str) -> dict:
        """返回某声旁下所有已知字：{ radical: [entry, ...] }"""
        return dict(self._index.get(phonetic, {}))

    def known_phonetics(self) -> list:
        return sorted(self._index.keys())

    def known_radicals(self) -> list:
        radicals = set()
        for ph_dict in self._index.values():
            radicals.update(ph_dict.keys())
        return sorted(radicals)

    def has_phonetic(self, phonetic: str) -> bool:
        return phonetic in self._index

    def has_exact(self, phonetic: str, radical: str) -> bool:
        return bool(self._index.get(phonetic, {}).get(radical))

    def get_all_entries_for_phonetic(self, phonetic: str) -> list:
        """返回某声旁下所有部首的所有字（扁平列表）。"""
        result = []
        for entries in self._index.get(phonetic, {}).values():
            result.extend(entries)
        return result

    # ── 近邻搜索 ──────────────────────────────────────────────

    def cosine_neighbors(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        exclude_chars: Optional[list] = None,
        filter_radical: Optional[str] = None,
    ) -> list:
        """
        768 维余弦近邻搜索。
        filter_radical: 只返回特定部首的字（可选）。
        """
        if self._all_vectors is None or len(self._all_vectors) == 0:
            return []

        exclude_set = set(exclude_chars or [])
        q     = query_vector / (np.linalg.norm(query_vector) + 1e-12)
        norms = np.linalg.norm(self._all_vectors, axis=1, keepdims=True) + 1e-12
        sims  = (self._all_vectors / norms) @ q

        results = []
        for idx in np.argsort(-sims):
            char = self._all_chars[idx]
            if char in exclude_set:
                continue
            entry = self._find_entry_by_char(char)
            if entry is None:
                continue
            if filter_radical and entry["radical"] != filter_radical:
                continue
            e = dict(entry)
            e["similarity"] = float(sims[idx])
            results.append(e)
            if len(results) >= top_k:
                break
        return results

    def _find_entry_by_char(self, char: str) -> Optional[dict]:
        for ph_dict in self._index.values():
            for entries in ph_dict.values():
                for e in entries:
                    if e["character"] == char:
                        return e
        return None

    # ── 声旁字本身的 embedding（降级用）─────────────────────

    def encode_phonetic_char(self, phonetic: str, model=None) -> Optional[np.ndarray]:
        """对声旁字本身 encode，缓存在内存里。"""
        if phonetic in self._phonetic_vecs:
            return self._phonetic_vecs[phonetic]

        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.MODEL_NAME)

        vec = model.encode(
            [config.BGE_INSTRUCTION + phonetic],
            normalize_embeddings=True
        )[0]
        self._phonetic_vecs[phonetic] = vec
        return vec

    # ── 序列化 ────────────────────────────────────────────────

    def save_json(self, path: Optional[str] = None):
        path = path or os.path.join(config.DATA_OUT, "phonetic_index.json")
        serializable = {}
        for phonetic, radical_dict in self._index.items():
            serializable[phonetic] = {}
            for radical, entries in radical_dict.items():
                serializable[phonetic][radical] = [
                    {k: v for k, v in e.items() if k != "vector_idx"}
                    for e in entries
                ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"✅ 声旁索引 JSON → {path}")

    # ── 工厂方法 ──────────────────────────────────────────────

    @classmethod
    def load(cls) -> "PhoneticIndex":
        idx         = cls()
        df_coords   = pd.read_csv(config.COORDS_CSV)
        npz_path    = os.path.join(config.DATA_OUT, "embed_index.npz")
        npz         = np.load(npz_path, allow_pickle=True)
        embed_index = {}
        for i, char in enumerate(npz["characters"]):
            embed_index[str(char)] = {
                "radical":  str(npz["radicals"][i]),
                "phonetic": str(npz["phonetics"][i]),
                "vector":   npz["vectors"][i],
            }
        idx.build(df_coords, embed_index)
        return idx


# ── 主入口 ────────────────────────────────────────────────────

def run():
    from src.radical_vectors import load_embed_index
    print(" 加载数据...")
    embed_index = load_embed_index()
    df_coords   = pd.read_csv(config.COORDS_CSV)

    idx = PhoneticIndex()
    idx.build(df_coords, embed_index)
    idx.save_json()

    print("\n 声旁索引摘要：")
    for phonetic in idx.known_phonetics():
        anchors = idx.get_anchors(phonetic)
        parts   = [f"{r}:{anchors[r][0]['character']}" for r in sorted(anchors)]
        print(f"  声旁「{phonetic}」→ " + "  |  ".join(parts))


if __name__ == "__main__":
    run()
