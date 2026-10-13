import argparse

from ragprobe.training.evaluator import evaluate_probe
from ragprobe.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    evaluate_probe(config)


if __name__ == "__main__":
    main()
