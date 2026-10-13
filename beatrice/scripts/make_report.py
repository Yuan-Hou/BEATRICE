import argparse

from ragprobe.utils.config import load_yaml
from ragprobe.visualization.rich_diagnostic_report import make_probe_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Override output.report_path for this report run.")
    parser.add_argument("--train-log", help="Override report.train_log_path for this report run.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.output:
        config.setdefault("output", {})["report_path"] = args.output
    if args.train_log:
        config.setdefault("report", {})["train_log_path"] = args.train_log
    make_probe_report(config)


if __name__ == "__main__":
    main()
