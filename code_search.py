import torch
import pandas as pd
from pathlib import Path
from itertools import chain

from date_utils import utc_ts
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util


def build_texts_from_repository(repo_dir):
    """Return a dataset of the non-blank lines of code
    """
    dataset = []

    for path in chain(
        Path(repo_dir).glob("**/*.py"),
        Path(repo_dir).glob("**/*.md"),
    ):
        assert path.is_file() and path.suffix
        lines = path.read_text().splitlines()

        dataset.extend(
            [
                {
                    "line_number": i,
                    "line": line,
                    "path": str(path.relative_to(repo_dir))}
                for i, line in enumerate(lines)
                if line.strip() != ""
            ]
        )
    return dataset


def build_query_dataset(queries, dataset):
    """
    Args:
        queries: plain list of strings
        dataset: list of dictionaries with ["line_number", "line", "path"]
            Example

            [{'line_number': 35,
              'line': '            name = "_"+name',
              'path': 'sentence_transformers/evaluation/MSEEvaluatorFromDataFrame.py'},
             {'line_number': 110,
              'line': 'if not os.path.exists(queries_filepath):',
              'path': 'examples/training/ms_marco/train_bi-encoder_mnrl.py'},
             {'line_number': 52,
              'line': "tracer = logging.getLogger('elasticsearch') ",
              'path': 'examples/training/data_augmentation/train_sts_indomain_bm25.py'}]

    """
    search_results = []
    for query in tqdm(queries):
        findings = [
            {"query": query, **x
             }
            for x in dataset if query in x["line"]
        ]
        search_results.extend(findings)
    return search_results


def run_semantic_search(embedder, dataset, queries, top_k):
    """

    Reference:
    https://www.sbert.net/examples/applications/semantic-search/README.html#python
    """

    # so I can do the evaluation later.
    corpus = [x["line"] for x in dataset]
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    truthdf = pd.DataFrame.from_records(dataset)

    results = []

    for query in tqdm(queries):
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        results.extend(
            [
                {
                    "score": score, "idx": idx, "query": query
                }
                for score, idx in zip(top_results[0], top_results[1])
            ]
        )
    resultsdf = pd.DataFrame.from_records(results).astype(
        {"idx": "int", "score": "float"})

    return resultsdf.merge(
        truthdf, left_on="idx", right_index=True, how="left"
    ).drop(["idx"], axis=1)
