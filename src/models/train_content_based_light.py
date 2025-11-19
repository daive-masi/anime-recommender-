#!/usr/bin/env python3
"""
train_content_based_light.py
Light content-based recommender:
 - TF-IDF on synopsis (max_features=10000)
 - MultiLabelBinarizer on genres
 - Combined sparse matrix saved as .npz
 - KNN index with sklearn (brute / ball_tree depending)
 - CLI params for fine-tuning
"""

import os
import argparse
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load
from tqdm import tqdm
import ast, re

# ----------------------------
# Logging
# ----------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "train_content_light.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("content_light")


# ----------------------------
# Helpers
# ----------------------------
def parse_list_column(s):
    if pd.isna(s) or s == "" or s == "[]":
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x).lower().strip() for x in val]
    except Exception:
        # fallback split by comma
        return [t.strip().lower() for t in re.split(r"[,\|;]+", str(s)) if t.strip()]
    return []

def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Training function
# ----------------------------
def train(args):
    logger.info("Loading cleaned master dataset")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded dataframe shape: {df.shape}")

    # Parse lists
    df['genres_list'] = df.get('genres_list', df.get('genres', "")).apply(parse_list_column)

    # Numeric features (optional) - small set
    df['members_log'] = np.log1p(df.get('members', 0).astype(float))

    numeric_cols = ['score', 'members_log']
    num_df = df[numeric_cols].fillna(0.0)

    # TF-IDF
    logger.info("Fitting TF-IDF on synopsis")
    tfidf = TfidfVectorizer(max_features=args.tfidf_max_features, stop_words='english', ngram_range=(1,2), min_df=2)
    tfidf_mat = tfidf.fit_transform(df['synopsis'].fillna("").astype(str))
    logger.info(f"TF-IDF matrix shape: {tfidf_mat.shape}")

    # Genres MultiLabelBinarizer
    logger.info("Fitting MultiLabelBinarizer for genres")
    mlb = MultiLabelBinarizer(sparse_output=True)
    genres_mat = mlb.fit_transform(df['genres_list'])
    logger.info(f"Genres matrix shape: {genres_mat.shape}")

    # Numeric scaled
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_df)
    num_sparse = csr_matrix(num_scaled)

    # Combine matrices with weighting
    # weights: allow adjusting importance
    w_genres = args.weight_genres
    w_tfidf = args.weight_tfidf
    w_num = args.weight_numeric

    # apply weights by multiplying numeric matrices
    if w_genres != 1.0:
        genres_mat = genres_mat.multiply(w_genres)
    if w_tfidf != 1.0:
        tfidf_mat = tfidf_mat.multiply(w_tfidf)
    if w_num != 1.0:
        num_sparse = num_sparse * w_num

    X = hstack([genres_mat, num_sparse, tfidf_mat], format='csr')
    logger.info(f"Combined feature matrix shape: {X.shape}")

    # Save artefacts
    ensure_dir(args.output_matrix)
    save_npz(args.output_matrix, X)
    logger.info(f"Saved feature matrix to {args.output_matrix}")

    meta = {
        'mal_id_index': df['mal_id'].tolist(),
        'tfidf': tfidf,
        'mlb_genres': mlb,
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'weights': {'genres': w_genres, 'tfidf': w_tfidf, 'numeric': w_num}
    }
    with open(args.output_meta, 'wb') as f:
        pickle.dump(meta, f)
    logger.info(f"Saved metadata to {args.output_meta}")

    # Build KNN index
    logger.info("Fitting NearestNeighbors index")
    nn = NearestNeighbors(n_neighbors=args.n_neighbors, metric='cosine', algorithm='brute')
    nn.fit(X)
    dump(nn, args.output_knn)
    logger.info(f"KNN model saved to {args.output_knn}")

    print("Training (light) complete.")
    logger.info("Completed training (light).")

# ----------------------------
# Recommendation util
# ----------------------------
def get_recommendations_light(title, meta_path, matrix_path, top_k=10):
    import pickle
    from sklearn.metrics.pairwise import cosine_similarity
    from joblib import load
    from scipy.sparse import load_npz

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    mal_index = meta['mal_id_index']
    tfidf = meta['tfidf']
    mlb = meta['mlb_genres']
    scaler = meta['scaler']
    X = load_npz(matrix_path)

    # find title index
    # naive exact match
    try:
        idx = mal_index.index(title)  # if you stored titles instead of mal_ids, adjust
    except ValueError:
        # try searching by substring
        raise ValueError("Title not found in index")

    row = X[idx]
    sim = cosine_similarity(row, X).ravel()
    top_idx = np.argsort(-sim)[1:top_k+1]
    return [(mal_index[i], float(sim[i])) for i in top_idx]


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/anime_master_clean.csv", help="input master clean csv")
    p.add_argument("--tfidf_max_features", type=int, default=10000)
    p.add_argument("--weight_genres", type=float, default=1.0)
    p.add_argument("--weight_tfidf", type=float, default=1.0)
    p.add_argument("--weight_numeric", type=float, default=1.0)
    p.add_argument("--n_neighbors", type=int, default=50)
    p.add_argument("--output_matrix", default="data/models/anime_features_light.npz")
    p.add_argument("--output_meta", default="data/models/anime_features_light_meta.pkl")
    p.add_argument("--output_knn", default="data/models/anime_knn_light.joblib")
    args = p.parse_args()
    train(args)
