import argparse

from ragprobe.triage.runner import run_probe_triage
from ragprobe.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_probe_triage(config)


if __name__ == "__main__":
    main()
