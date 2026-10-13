import argparse

from ragprobe.llm.openai_client import build_llm
from ragprobe.regression.analyzer import run_regression_analysis
from ragprobe.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    judge_llm = build_llm(config["judge_model"])
    run_regression_analysis(config, judge_llm)


if __name__ == "__main__":
    main()
