import argparse

from ragprobe.evaluation.generation_runner import run_generation_eval
from ragprobe.llm.openai_client import build_llm
from ragprobe.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_llm = build_llm(config["model"])
    judge_llm = build_llm(config["judge_model"])
    run_generation_eval(config, model_llm, judge_llm)


if __name__ == "__main__":
    main()
