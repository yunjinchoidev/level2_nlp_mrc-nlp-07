import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
)
import os
import pickle
from typing import List, Tuple, NoReturn, Optional, Union
from datasets import Dataset
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
import argparse


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


set_seed(42)


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self):
        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        query_vec = self.tfidfv.transform([query])
        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--use_bulk", default=False, type=bool)
    parser.add_argument("--topk", default=2, type=int)
    parser.add_argument("--data_path", default="../../data/train_dataset", type=str)
    parser.add_argument("--save_path", default="../../data/train_with_neg", type=str)
    args = parser.parse_args()

    model_checkpoint = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenize_fn = lambda x: tokenizer.tokenize(x)
    retriever = SparseRetrieval(tokenize_fn=tokenize_fn)

    retriever.get_sparse_embedding()

    data_path = args.data_path
    datasets = load_from_disk(data_path)

    train = pd.DataFrame(
        {key: datasets["train"][key] for key in datasets["train"].features.keys()}
    )
    negative_examples = []
    for idx, query in tqdm(enumerate(train["question"])):
        doc_score, doc_indices = retriever.get_relevant_doc(query, k=args.topk + 1)
        ground_truth = train["context"][idx]
        relevant_doc = [retriever.contexts[idx] for idx in doc_indices]
        if ground_truth in relevant_doc:
            del relevant_doc[relevant_doc.index(ground_truth)]

        if args.use_bulk:
            negative_examples.append(relevant_doc[: args.topk])
        else:
            negative_examples.append(relevant_doc[0])
    train["negative_examples"] = negative_examples
    train = train.drop(
        columns=[
            "title",
            "document_id",
            "__index_level_0__",
        ]
    )
    f = Features(
        {
            "answers": Sequence(
                feature={
                    "text": Value(dtype="string", id=None),
                    "answer_start": Value(dtype="int32", id=None),
                },
                length=-1,
                id=None,
            ),
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
            "negative_examples": Sequence(
                feature=Value(dtype="string", id=None),
                length=-1,
                id=None,
            )
            if args.use_bulk
            else Value(dtype="string", id=None),
        }
    )
    new_datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(train, features=f),
            "validation": datasets["validation"],
        }
    )
    new_datasets.save_to_disk(args.save_path)
