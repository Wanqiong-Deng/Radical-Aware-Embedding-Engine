"""
src/embed.py
职责：BGE encode → 缓存 768 维向量 → PCA 降维 → 输出 coords.csv + embed_index.npz
"""

import os
import sys
import hashlib
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _text_hash(texts: list) -> str:
    content = "\n".join(texts).encode("utf-8")
    return hashlib.md5(content).hexdigest()[:12]


def load_model():
    from sentence_transformers import SentenceTransformer
    print(f" 加载模型 {config.MODEL_NAME} ...")
    return SentenceTransformer(config.MODEL_NAME)


def encode_texts(texts: list, model=None, force: bool = False) -> np.ndarray:
    """encode 并缓存。hash 变了自动重新计算。"""
    h = _text_hash(texts)
    cache_file = config.EMBED_NPY.replace(".npy", f"_{h}.npy")

    if not force and os.path.exists(cache_file):
        print(f"💾 命中缓存 ({os.path.basename(cache_file)})")
        return np.load(cache_file)

    if model is None:
        model = load_model()

    print(f"🔢 encode {len(texts)} 条文本...")
    prefixed = [config.BGE_INSTRUCTION + t for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=True)

    os.makedirs(config.DATA_OUT, exist_ok=True)
    np.save(cache_file, embeddings)
    np.save(config.EMBED_NPY, embeddings)  
    print(f"✅ 向量已缓存 → {cache_file}")
    return embeddings


def reduce_to_2d(embeddings: np.ndarray):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    ev = pca.explained_variance_ratio_
    print(f"📊 PCA 解释方差: PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  累计={sum(ev):.1%}")
    return coords, pca


def run(force_encode: bool = False):
    df = pd.read_csv(config.CLEAN_CSV)
    texts = df["Combined"].tolist()

    embeddings = encode_texts(texts, force=force_encode)
    coords, pca = reduce_to_2d(embeddings)

    df["x"], df["y"] = coords[:, 0], coords[:, 1]

    # ── PCA 对象序列化（供 app.py 投影新字用）───────────────
    pca_path = os.path.join(config.DATA_OUT, "pca_model.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)

    # ── 768 维索引（供 radical_vectors / predict 用）────────
    phonetics = df["Phonetic"].astype(str).tolist() if "Phonetic" in df.columns \
                else [""] * len(df)

    index_path = os.path.join(config.DATA_OUT, "embed_index.npz")
    np.savez(
        index_path,
        vectors    = embeddings,
        characters = np.array(df["Character"].tolist()),
        radicals   = np.array(df["Radical"].tolist()),
        phonetics  = np.array(phonetics),
        group_ids  = np.array(df["GroupID"].astype(str).tolist()),
    )
    print(f"✅ 768维索引 → {index_path}  shape={embeddings.shape}")

    df.to_csv(config.COORDS_CSV, index=False, encoding="utf_8_sig")
    print(f"✅ 坐标 → {config.COORDS_CSV}")
    return df, embeddings, pca


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force_encode=args.force)
