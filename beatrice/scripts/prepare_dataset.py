import argparse

from ragprobe.datasets.registry import get_dataset_builder
from ragprobe.utils.config import load_yaml
from ragprobe.utils.io import write_jsonl
from ragprobe.utils.scope import filter_records_by_scope, load_scope_ids, scope_path_from_config
from ragprobe.utils.logging import log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    scope_path = scope_path_from_config(config, "dataset")
    dataset_args = dict(config["dataset"].get("args", {}))
    if config["dataset"]["name"] == "popqa" and scope_path:
        dataset_args.setdefault("scope_ids_path", scope_path)
    builder = get_dataset_builder(
        config["dataset"]["name"],
        **dataset_args,
    )

    records = builder.build()
    if scope_path:
        before = len(records)
        records = filter_records_by_scope(records, load_scope_ids(scope_path))
        log(f"Applied fixed sample scope: {len(records)} / {before} records kept from {scope_path}")
    write_jsonl(records, config["output_path"])
    log(f"Saved {len(records)} examples to {config['output_path']}")


if __name__ == "__main__":
    main()
