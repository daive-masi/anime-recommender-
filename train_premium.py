"""
PREMIUM MODEL: Point d'entrée principal pour le pipeline avancé
- Embeddings avancés (TF-IDF 50k + UMAP/SVD)
- Recherche approchée avec FAISS
- Support GPU optionnel
"""

from src.utils.config_loader import load_config
from src.models.content_premium.train import train

if __name__ == "__main__":
    import argparse

    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Lance le pipeline PREMIUM pour les recommandations avancées")
    parser.add_argument("--config", required=True, type=str, help="Chemin vers le fichier YAML de configuration")
    parser.add_argument("--gpu", action="store_true", help="Active l'accélération GPU pour FAISS/UMAP (nécessite CUDA)")
    parser.add_argument("--debug", action="store_true", help="Mode debug avec logs détaillés")

    # Parsing des arguments
    args = parser.parse_args()

    # Chargement de la configuration
    cfg = load_config(args.config)

    # Exécution du pipeline
    train(
        config=cfg,
        config_path=args.config,
        use_gpu=args.gpu,
        debug_mode=args.debug
    )
