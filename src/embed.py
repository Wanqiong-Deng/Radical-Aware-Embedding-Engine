"""
src/embed.py
职责：加载 BGE 模型 → encode 文本 → 缓存 .npy → PCA 降维 → 输出 coords.csv

关键改进：
  - embeddings 缓存为 .npy，只有数据变化时才重新 encode（省时间）
  - PCA 对象可序列化，方便后续 app 用同一个投影空间
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _text_hash(texts: list[str]) -> str:
    """用文本内容的 hash 判断是否需要重新 encode。"""
    content = "\n".join(texts).encode("utf-8")
    return hashlib.md5(content).hexdigest()[:12]


def load_model():
    from sentence_transformers import SentenceTransformer
    print(f"🤖 加载模型 {config.MODEL_NAME} ...")
    return SentenceTransformer(config.MODEL_NAME)


def encode_texts(texts: list[str], model=None, force: bool = False) -> np.ndarray:
    """
    encode 并缓存。force=True 强制重新 encode。
    缓存文件名包含文本 hash，内容变了自动重新计算。
    """
    h = _text_hash(texts)
    cache_file = config.EMBED_NPY.replace(".npy", f"_{h}.npy")
    
    if not force and os.path.exists(cache_file):
        print(f" 命中缓存，直接加载向量 ({cache_file})")
        return np.load(cache_file)
    
    if model is None:
        model = load_model()
    
    print(f" 正在 encode {len(texts)} 条文本...")
    prefixed = [config.BGE_INSTRUCTION + t for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=True)
    
    os.makedirs(config.DATA_OUT, exist_ok=True)
    np.save(cache_file, embeddings)

    np.save(config.EMBED_NPY, embeddings)
    print(f" 向量已缓存 → {cache_file}")
    return embeddings


def reduce_to_2d(embeddings: np.ndarray, n_components: int = 2):
    """PCA 降维，返回 (coords, pca_object)。"""
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    print(f" PCA 解释方差: PC1={explained[0]:.1%}, PC2={explained[1]:.1%} "
          f"(累计 {sum(explained):.1%})")
    return coords, pca


def run(force_encode: bool = False):
    df = pd.read_csv(config.CLEAN_CSV)
    texts = df["Combined"].tolist()
    
    embeddings = encode_texts(texts, force=force_encode)
    coords, pca = reduce_to_2d(embeddings)
    
    df["x"], df["y"] = coords[:, 0], coords[:, 1]
    
    
    pca_path = os.path.join(config.DATA_OUT, "pca_model.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    
    df.to_csv(config.COORDS_CSV, index=False, encoding="utf_8_sig")
    print(f" 坐标已保存 → {config.COORDS_CSV}")
    return df, embeddings, pca


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制重新 encode，忽略缓存")
    args = parser.parse_args()
    run(force_encode=args.force)