#!/usr/bin/env python3
"""
train_content_based_premium.py
Premium pipeline:
 - TF-IDF on synopsis (max_features=50000)
 - MultiLabelBinarizer on genres/themes/demographics
 - Studio encoding (top-K -> one-hot)
 - Combine sparse features -> optional UMAP -> dense embeddings
 - Build ANN (FAISS if available, otherwise sklearn)
 - Save artifacts: vectorizer, mlb, scaler, umap, ann index, id map
"""

import os
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import ast, re
from scipy.sparse import hstack, csr_matrix, save_npz, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
import umap
from joblib import dump, load
from tqdm import tqdm

# Optionally use FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(filename=LOG_DIR / "train_content_premium.log",
                    level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("content_premium")

# Helpers
def parse_list_column(s):
    if pd.isna(s) or s == "" or s == "[]":
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x).lower().strip() for x in val]
    except Exception:
        return [t.strip().lower() for t in re.split(r"[,\|;]+", str(s)) if t.strip()]
    return []

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def train(args):
    logger.info("Loading dataset")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded df {df.shape}")

    # parse list fields
    df['genres_list'] = df.get('genres_list', df.get('genres', "")).apply(parse_list_column)
    df['themes_list'] = df.get('themes_list', df.get('themes', "")).apply(parse_list_column)
    df['demo_list'] = df.get('demo_list', df.get('demographics', "")).apply(parse_list_column)

    # TF-IDF
    logger.info("Fitting TF-IDF")
    tfidf = TfidfVectorizer(max_features=args.tfidf_max_features, stop_words='english', ngram_range=args.ngram_range, min_df=args.min_df)
    X_tfidf = tfidf.fit_transform(df['synopsis'].fillna("").astype(str))
    logger.info(f"TF-IDF shape: {X_tfidf.shape}")

    # Multi-label
    mlb_genres = MultiLabelBinarizer(sparse_output=True)
    mlb_themes = MultiLabelBinarizer(sparse_output=True)
    mlb_demo = MultiLabelBinarizer(sparse_output=True)

    X_genres = mlb_genres.fit_transform(df['genres_list'])
    X_themes = mlb_themes.fit_transform(df['themes_list'])
    X_demo = mlb_demo.fit_transform(df['demo_list'])

    logger.info(f"Genres shape: {X_genres.shape}, Themes: {X_themes.shape}, Demo: {X_demo.shape}")

    # Studios top-K one-hot
    studios = df.get('studios', pd.Series([""]*len(df)))
    studios_clean = studios.fillna("").astype(str)
    topk = studios_clean.value_counts().index[:args.topk_studios]
    studio_map = {s:i for i,s in enumerate(topk)}
    studio_onehot = np.zeros((len(df), len(topk)), dtype=np.float32)
    for i, s in enumerate(studios_clean):
        if s in studio_map:
            studio_onehot[i, studio_map[s]] = 1.0
    X_studios = csr_matrix(studio_onehot)

    # Numeric features
    df['members_log'] = np.log1p(df.get('members', 0).astype(float))
    num_cols = ['score','members_log','weighted_score']
    num_df = df[num_cols].fillna(0.0)
    scaler = StandardScaler()
    X_num = csr_matrix(scaler.fit_transform(num_df))

    # Combine with weights
    if args.weight_genres != 1.0:
        X_genres = X_genres.multiply(args.weight_genres)
    if args.weight_themes != 1.0:
        X_themes = X_themes.multiply(args.weight_themes)
    if args.weight_tfidf != 1.0:
        X_tfidf = X_tfidf.multiply(args.weight_tfidf)

    X_comb = hstack([X_genres, X_themes, X_demo, X_studios, X_num, X_tfidf], format='csr')
    logger.info(f"Combined sparse matrix shape: {X_comb.shape}")

    # Optionally reduce TF-IDF with SVD for speed before UMAP
    if args.svd_components and args.svd_components > 0:
        logger.info(f"Applying SVD to TF-IDF -> {args.svd_components} components")
        svd = TruncatedSVD(n_components=args.svd_components, random_state=42)
        X_tfidf_reduced = svd.fit_transform(X_tfidf)
        # replace tfidf block with reduced dense -> csr
        X_tfidf_block = csr_matrix(X_tfidf_reduced)
        # rebuild combined
        X_comb = hstack([X_genres, X_themes, X_demo, X_studios, X_num, X_tfidf_block], format='csr')
        logger.info(f"New combined shape after SVD: {X_comb.shape}")
    else:
        svd = None

    # UMAP (default enabled)
    embeddings = None
    if args.use_umap:
        logger.info("Fitting UMAP")
        umapper = umap.UMAP(
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            n_components=args.umap_n_components,
            metric='cosine',
            random_state=42
        )
        # UMAP expects dense or sparse; it can handle sparse
        embeddings = umapper.fit_transform(X_comb)
        logger.info(f"UMAP embeddings shape: {embeddings.shape}")
    else:
        logger.info("UMAP disabled: using SVD/dense fallback")
        # fallback: dense via SVD if not provided
        if svd is not None:
            embeddings = svd.transform(X_tfidf)
        else:
            # densify a small projection via truncated SVD on combined matrix
            svd2 = TruncatedSVD(n_components=min(256, X_comb.shape[1]-1), random_state=42)
            embeddings = svd2.fit_transform(X_comb)

    # Save artifacts
    ensure_dir(args.output_emb)
    np.save(args.output_emb, embeddings)
    logger.info(f"Saved embeddings to {args.output_emb}")

    meta = {
        'mal_id_index': df['mal_id'].tolist(),
        'tfidf': tfidf,
        'mlb_genres': mlb_genres,
        'mlb_themes': mlb_themes,
        'mlb_demo': mlb_demo,
        'scaler': scaler,
        'studio_map': studio_map,
        'svd': svd,
        'umapper': (umapper if args.use_umap else None),
    }
    with open(args.output_meta, 'wb') as f:
        pickle.dump(meta, f)
    logger.info(f"Saved meta to {args.output_meta}")

    # Build ANN
    if FAISS_AVAILABLE and args.use_faiss:
        logger.info("Building FAISS index")
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d) if args.faiss_metric == 'ip' else faiss.IndexFlatL2(d)
        # normalize for inner product if needed
        if args.faiss_normalize:
            faiss.normalize_L2(embeddings)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, args.output_faiss)
        logger.info(f"Saved FAISS index at {args.output_faiss}")
    else:
        from sklearn.neighbors import NearestNeighbors
        logger.info("Building sklearn NearestNeighbors index")
        nn = NearestNeighbors(n_neighbors=args.n_neighbors, metric='cosine', algorithm='auto')
        nn.fit(embeddings)
        dump(nn, args.output_nn)
        logger.info(f"Saved NN index at {args.output_nn}")

    logger.info("Premium training complete.")

# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/anime_master_clean.csv")
    p.add_argument("--tfidf_max_features", type=int, default=50000)
    p.add_argument("--ngram_range", type=tuple, default=(1,2))
    p.add_argument("--min_df", type=int, default=3)
    p.add_argument("--topk_studios", type=int, default=100)
    p.add_argument("--svd_components", type=int, default=128)  # optional SVD pre-reduction
    p.add_argument("--use_umap", type=lambda x: (str(x).lower() == 'true'), default=True)
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--umap_n_components", type=int, default=256)
    p.add_argument("--use_faiss", type=lambda x: (str(x).lower() == 'true'), default=False)
    p.add_argument("--faiss_metric", type=str, default='ip')
    p.add_argument("--faiss_normalize", type=lambda x: (str(x).lower() == 'true'), default=True)
    p.add_argument("--n_neighbors", type=int, default=50)
    p.add_argument("--weight_genres", type=float, default=1.0)
    p.add_argument("--weight_themes", type=float, default=1.0)
    p.add_argument("--weight_tfidf", type=float, default=1.0)
    p.add_argument("--output_emb", default="data/models/anime_embeddings_premium.npy")
    p.add_argument("--output_meta", default="data/models/anime_premium_meta.pkl")
    p.add_argument("--output_nn", default="data/models/anime_premium_nn.joblib")
    p.add_argument("--output_faiss", default="data/models/anime_premium.faiss")
    args = p.parse_args()
    train(args)
