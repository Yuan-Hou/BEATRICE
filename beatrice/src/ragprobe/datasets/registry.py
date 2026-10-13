from .popqa import PopQABuilder
from .musique import MusiqueBuilder
from .hotpotqa import HotpotQABuilder
from .nq import NQBuilder
from .triviaqa import TriviaQABuilder
from .twowiki import TwoWikiMultiHopQABuilder


DATASET_REGISTRY = {
    "popqa": PopQABuilder,
    "musique": MusiqueBuilder,
    "hotpotqa": HotpotQABuilder,
    "nq": NQBuilder,
    "triviaqa": TriviaQABuilder,
    "twowiki": TwoWikiMultiHopQABuilder,
    "2wiki": TwoWikiMultiHopQABuilder,
    "2wikimultihopqa": TwoWikiMultiHopQABuilder,
}


def get_dataset_builder(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name](**kwargs)
