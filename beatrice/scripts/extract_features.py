import argparse

from ragprobe.features.extractor import run_feature_extraction
from ragprobe.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_feature_extraction(config)


if __name__ == "__main__":
    main()
