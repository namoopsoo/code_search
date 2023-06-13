from pathlib import Path
from itertools import chain

from date_utils import utc_ts
from tqdm import tqdm


def build_texts_from_repository(repo_dir):
    """Return a dataset of the non-blank lines of code
    """
    dataset = []
    file_types = []
    for path in chain(
        Path(repo_dir).glob("**/*.py"),
        Path(repo_dir).glob("**/*.md"),
    ):
        assert path.is_file() and path.suffix
        lines = path.read_text().splitlines()
        
        dataset.extend(
            [{"line_number": i,
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


def run_search(dataset, queries):
    """

    Reference:
    https://www.sbert.net/examples/applications/semantic-search/README.html#python
    """

    # TODO let's make sure I tape back the line numbers from the dataset, after getting the corpus for the encoder, 
    # so I can do the evaluation later.
    corpus = [x["line"] for x in dataset]
    

    top_k = min(5, len(corpus))
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            print(corpus[idx].strip(), "(Score: {:.4f})".format(score))

        """
        # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
        hits = hits[0]      #Get the hits for the first query
        for hit in hits:
            print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
        """

