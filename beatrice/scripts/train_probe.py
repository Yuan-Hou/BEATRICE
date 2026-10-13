import argparse

# DataLoader workers pass large per-sample tensors (4096-d L17 hidden states) to the main
# process. The default 'file_descriptor' strategy routes these through /dev/shm, which is
# only 64GB here and overflows under multiple concurrent trainings -> workers get SIGKILLed
# ("DataLoader worker ... killed by signal: Killed"). 'file_system' backs them with temp
# files instead (absorbed by the 2TB page cache), removing the shm ceiling.
# DataLoader workers pass large per-sample tensors (4096-d L17 hidden states) to the main
# process through /dev/shm (only 64GB here). Raise the soft fd limit (helps both sharing
# strategies) and keep the default file_descriptor strategy, which auto-refcounts shm.
import resource as _resource

try:
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (_hard, _hard))
except Exception:
    pass

from ragprobe.training.trainer import train_probe
from ragprobe.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    train_probe(config)


if __name__ == "__main__":
    main()
