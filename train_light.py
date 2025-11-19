from src.utils.config_loader import load_config
from src.models.content_light.train import train

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, config_path=args.config)
