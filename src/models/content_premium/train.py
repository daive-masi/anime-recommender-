import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from src.utils.run_manager import create_run_folder, save_config_copy
from src.utils.logger import get_logger


# ------------------------------------------------------------
#   Feature builders
# ------------------------------------------------------------

def build_tfidf(df, cfg, logger):
    logger.info("TF-IDF (premium) building...")

    vect = TfidfVectorizer(
        max_features=cfg["tfidf"]["max_features"],
        min_df=cfg["tfidf"]["min_df"],
        ngram_range=tuple(cfg["tfidf"]["ngram_range"])
    )

    X = vect.fit_transform(df["synopsis_clean"])
    logger.info(f"TF-IDF = {X.shape}")

    return X, vect


def build_genres(df, logger):
    logger.info("One-hot encoding genres...")
    m = df["genres_list"].str.join("|").str.get_dummies()
    logger.info(f"Genres = {m.shape}")
    return m


def build_numeric(df, cfg, logger):
    logger.info("Numeric features...")
    cols = cfg["numeric_features"]
    scaler = StandardScaler()
    arr = scaler.fit_transform(df[cols])
    logger.info(f"Numeric = {arr.shape}")
    return arr, scaler


def concat_sparse(parts, logger):
    from scipy.sparse import hstack
    logger.info("Concatenating...")
    X = hstack(parts).tocsr()
    logger.info(f"Concatenated = {X.shape}")
    return X


# ------------------------------------------------------------
#  Dimensionality Reduction
# ------------------------------------------------------------

def apply_svd(X, cfg, logger):
    if not cfg["svd"]["enabled"]:
        logger.info("SVD disabled.")
        return X, None

    logger.info("Applying SVD...")
    svd = TruncatedSVD(n_components=cfg["svd"]["n_components"])
    X2 = svd.fit_transform(X)
    logger.info(f"SVD output = {X2.shape}")
    return X2, svd


def apply_umap(X, cfg, logger):
    if not cfg["umap"]["enabled"]:
        logger.info("UMAP disabled.")
        return X, None

    import umap

    logger.info("Applying UMAP...")
    reducer = umap.UMAP(
        n_neighbors=cfg["umap"]["n_neighbors"],
        min_dist=cfg["umap"]["min_dist"],
        n_components=cfg["umap"]["n_components"],
        metric="cosine"
    )
    X2 = reducer.fit_transform(X)
    logger.info(f"UMAP output = {X2.shape}")
    return X2, reducer


# ------------------------------------------------------------
#  ANN (FAISS or sklearn)
# ------------------------------------------------------------

def build_ann(X, cfg, logger):
    if cfg["ann"]["use_faiss"]:
        logger.info("Using FAISS index...")
        import faiss

        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        index.add(X_norm.astype("float32"))
        return index

    else:
        logger.info("Using sklearn NearestNeighbors...")
        nn = NearestNeighbors(
            n_neighbors=cfg["ann"]["n_neighbors"],
            metric="cosine"
        )
        nn.fit(X)
        return nn


# ------------------------------------------------------------
#  Main train
# ------------------------------------------------------------

def train(cfg, config_path):
    run_folder = create_run_folder("premium")
    save_config_copy(config_path, run_folder)
    logger = get_logger(run_folder / "training.log")

    logger.info("===== START PREMIUM TRAIN =====")

    # ----------------------------------------------------
    # 1) Load dataset
    # ----------------------------------------------------
    df = pd.read_csv(cfg["dataset"])
    logger.info(f"Dataset = {df.shape}")

    # ----------------------------------------------------
    # 2) Build features
    # ----------------------------------------------------
    X_parts = []
    meta = {}

    X_tfidf, vect = build_tfidf(df, cfg, logger)
    X_parts.append(X_tfidf)
    meta["tfidf_vectorizer"] = vect

    X_genres = build_genres(df, logger)
    X_parts.append(X_genres)

    X_num, scaler_num = build_numeric(df, cfg, logger)
    X_parts.append(X_num)
    meta["numeric_scaler"] = scaler_num

    X = concat_sparse(X_parts, logger)

    # ----------------------------------------------------
    # 3) Dimensionality reduction
    # ----------------------------------------------------
    X_svd, svd = apply_svd(X, cfg, logger)
    if svd: meta["svd"] = svd

    X_final, umap_model = apply_umap(X_svd, cfg, logger)
    if umap_model: meta["umap"] = umap_model

    logger.info(f"Embedding final size = {X_final.shape}")

    # ----------------------------------------------------
    # 4) ANN Training
    # ----------------------------------------------------
    ann = build_ann(X_final, cfg, logger)

    # ----------------------------------------------------
    # 5) Save outputs
    # ----------------------------------------------------
    out = run_folder / "latest"
    out.mkdir(exist_ok=True)

    np.save(out / "embeddings.npy", X_final)
    joblib.dump(meta, out / "meta.pkl")

    df[["mal_id", "title"]].to_csv(out / "metadata.csv", index=False)

    # Save ANN
    if cfg["ann"]["use_faiss"]:
        import faiss
        faiss.write_index(ann, str(out / "faiss_index.faiss"))
    else:
        joblib.dump(ann, out / "nn_model.joblib")

    logger.info("===== PREMIUM TRAIN FINISHED =====")

    return {
        "run_folder": str(run_folder),
        "num_items": len(df),
        "embed_dim": X_final.shape[1]
    }
