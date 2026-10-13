import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .base import DatasetBuilder
from ragprobe.utils.scope import load_scope_ids


class PopQABuilder(DatasetBuilder):
    name = "popqa"

    def __init__(
        self,
        split="test",
        max_examples=None,
        num_workers=32,
        wikipedia_user_agent=None,
        scope_ids_path=None,
    ):
        self.split = split
        self.max_examples = max_examples
        self.num_workers = num_workers
        self.scope_ids = load_scope_ids(scope_ids_path) if scope_ids_path else set()
        self.wikipedia_user_agent = wikipedia_user_agent or (
            "PopQAResearchBot/1.0 academic_research@example.com"
        )

    def build(self):
        ds = load_dataset("akariasai/PopQA", split=self.split)
        if self.max_examples:
            ds = ds.select(range(int(self.max_examples)))

        session = self._build_session()

        def fetch(idx_item):
            idx, item = idx_item
            context, title = self._fetch_wikipedia_intro(session, item["subj"])
            if not context:
                return None

            try:
                answers = ast.literal_eval(item["possible_answers"])
            except Exception:
                answers = [item["possible_answers"]]

            return {
                "id": f"popqa-{self.split}-{idx:06d}",
                "dataset": "popqa",
                "split": self.split,
                "question": item["question"],
                "contexts": [
                    {
                        "text": context,
                        "title": title or item["subj"],
                        "source": "wikipedia",
                        "is_supporting": True,
                        "metadata": {}
                    }
                ],
                "answers": answers,
                "answerable": True,
                "metadata": {
                    "subj": item.get("subj"),
                    "prop": item.get("prop"),
                    "obj": item.get("obj"),
                },
            }

        records = []
        indexed = list(enumerate(ds))
        if self.scope_ids:
            indexed = [(idx, item) for idx, item in indexed if f"popqa-{self.split}-{idx:06d}" in self.scope_ids]

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(fetch, x): x[0] for x in indexed}
            for future in tqdm(as_completed(futures), total=len(futures), desc="PopQA Wikipedia"):
                result = future.result()
                if result is not None:
                    records.append(result)

        records.sort(key=lambda x: x["id"])
        return records

    def _build_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=64,
            pool_maxsize=64,
        )
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": self.wikipedia_user_agent})
        return session

    def _fetch_wikipedia_intro(self, session, subject):
        url = "https://en.wikipedia.org/w/api.php"

        try:
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": subject,
                "format": "json",
                "utf8": 1,
                "srlimit": 1,
            }
            res = session.get(url, params=search_params, timeout=10).json()
            search_results = res.get("query", {}).get("search", [])
            if not search_results:
                return None, None

            title = search_results[0]["title"]

            summary_params = {
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
                "format": "json",
            }
            res2 = session.get(url, params=summary_params, timeout=10).json()
            pages = res2.get("query", {}).get("pages", {})

            for page_id, page_data in pages.items():
                if page_id != "-1":
                    context = page_data.get("extract", "").strip()
                    return (context or None), title

        except Exception:
            return None, None

        return None, None
