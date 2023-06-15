from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
import os
import json
import pickle
import pandas as pd
from encoder import *
import wandb
import datetime
from pytz import timezone
import timer
from utils import retieval_logging
from datasets import Dataset, concatenate_datasets, load_from_disk

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)


class DenseRetriever:
    def __init__(
        self,
        data_path,
        context_path,
        model_name_or_path,
        p_encoder_ckpt=None,
        q_encoder_ckpt=None,
        stage="train",
        use_neg_sampling=False,
        num_neg=15,
    ):
        self.model_name_or_path = model_name_or_path
        self.stage = stage
        self.num_neg = num_neg
        self.use_neg_sampling = use_neg_sampling

        if p_encoder_ckpt == None:
            self.p_encoder_ckpt = model_name_or_path
            self.q_encoder_ckpt = model_name_or_path
        else:
            self.p_encoder_ckpt = p_encoder_ckpt
            self.q_encoder_ckpt = q_encoder_ckpt

        # 모델에 따라서 변수 세팅
        self.set_models()

        # 학습용 데이터 불러오기
        self.dataset = load_from_disk(data_path)

        # 만약 test 단계라면 validation 밖에 없으므로
        if stage == "test":
            training_dataset = self.dataset["validation"]
        else:
            training_dataset = self.dataset["train"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, model_max_length=512, use_fast=True
        )

        # 쿼리 시퀀스 불러오기
        q_seqs = self.tokenizer(
            training_dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # context는 학습 데이터만 있으므로
        if stage == "train":
            # negative sampling을 사용하기 위한 데이터셋 세팅
            if self.use_neg_sampling:
                corpus = training_dataset["context"]
                neg_ex = training_dataset["negative_examples"]

                p_with_neg = []
                for idx in tqdm(range(len(training_dataset))):
                    p_with_neg.append(corpus[idx])
                    p_with_neg.extend(neg_ex[idx])

                p_seqs = self.tokenizer(
                    p_with_neg,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                max_len = p_seqs["input_ids"].size(-1)
                p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
                p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
                    -1, num_neg + 1, max_len
                )
                p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
                    -1, num_neg + 1, max_len
                )

                self.train_dataset = TensorDataset(
                    p_seqs["input_ids"],
                    p_seqs["attention_mask"],
                    p_seqs["token_type_ids"],
                    q_seqs["input_ids"],
                    q_seqs["attention_mask"],
                    q_seqs["token_type_ids"],
                )

            # 일반 데이터셋 세팅
            else:
                p_seqs = self.tokenizer(
                    training_dataset["context"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                self.train_dataset = TensorDataset(
                    p_seqs["input_ids"],
                    p_seqs["attention_mask"],
                    p_seqs["token_type_ids"],
                    q_seqs["input_ids"],
                    q_seqs["attention_mask"],
                    q_seqs["token_type_ids"],
                )

        # 모델에 따라서 인코더 세팅
        if self.isRoberta:
            self.p_encoder = RoBertaEncoder.from_pretrained(self.p_encoder_ckpt)
            self.q_encoder = RoBertaEncoder.from_pretrained(self.q_encoder_ckpt)

        elif self.isBart:
            self.p_encoder = BartEncoder.from_pretrained(self.p_encoder_ckpt)
            self.q_encoder = BartEncoder.from_pretrained(self.q_encoder_ckpt)

        elif self.isT5:
            self.p_encoder = T5Encoder.from_pretrained(self.p_encoder_ckpt)
            self.q_encoder = T5Encoder.from_pretrained(self.q_encoder_ckpt)

        elif self.isElectra:
            self.p_encoder = ElectraEncoder.from_pretrained(self.p_encoder_ckpt)
            self.q_encoder = ElectraEncoder.from_pretrained(self.q_encoder_ckpt)

        else:
            self.p_encoder = BertEncoder.from_pretrained(self.p_encoder_ckpt)
            self.q_encoder = BertEncoder.from_pretrained(self.q_encoder_ckpt)

        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()

        # 전체 context 불러와서 세팅하기
        with open("../data/wikipedia_documents.json", "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

    def train(self, args):
        if not self.use_neg_sampling:
            train_sampler = RandomSampler(self.train_dataset)
        else:
            train_sampler = SequentialSampler(self.train_dataset)

        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=args.per_device_train_batch_size,
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )

        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_ratio * t_total),
            num_training_steps=t_total,
        )

        global_step = 0

        # 학습을 위한 인코더 초기화
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()
        Ebar = tqdm(range(args.num_train_epochs))
        for e in Ebar:
            Ebar.set_description(f"Epoch - {e}")
            Ibar = tqdm(train_dataloader)
            for batch in Ibar:
                self.q_encoder.train()
                self.p_encoder.train()
                if self.use_neg_sampling:
                    targets = torch.zeros(args.per_device_train_batch_size).long()
                else:
                    targets = torch.arange(0, args.per_device_train_batch_size).long()

                # 배치와 타켓 모두 GPU 올리기
                if torch.cuda.is_available():
                    targets = targets.to("cuda")
                    batch = tuple(t.cuda() for t in batch)

                # negative sampling을 사용할 경우의 input
                if self.use_neg_sampling:
                    p_inputs = {
                        "input_ids": batch[0].view(
                            args.per_device_train_batch_size * (self.num_neg + 1), -1
                        ),
                        "attention_mask": batch[1].view(
                            args.per_device_train_batch_size * (self.num_neg + 1), -1
                        ),
                        "token_type_ids": batch[2].view(
                            args.per_device_train_batch_size * (self.num_neg + 1), -1
                        ),
                    }
                    q_inputs = {
                        "input_ids": batch[3],
                        "attention_mask": batch[4],
                        "token_type_ids": batch[5],
                    }

                # 일반적인 input
                else:
                    p_inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }

                    q_inputs = {
                        "input_ids": batch[3],
                        "attention_mask": batch[4],
                        "token_type_ids": batch[5],
                    }

                # 인코더에 입력하기
                p_outputs = self.p_encoder(**p_inputs)  # (batch_size, emb_dim)
                q_outputs = self.q_encoder(**q_inputs)  # (batch_size, emb_dim)

                # negative sampling을 사용할 경우의 sim_scores
                if self.use_neg_sampling:
                    p_outputs = p_outputs.view(
                        args.per_device_train_batch_size, -1, self.num_neg + 1
                    )
                    q_outputs = q_outputs.view(args.per_device_train_batch_size, 1, -1)

                    sim_scores = torch.bmm(
                        q_outputs, p_outputs
                    ).squeeze()  # (batch_size, num_neg+1)
                    sim_scores = sim_scores.view(args.per_device_train_batch_size, -1)

                # 일반적인 경우의 sim_scores
                else:
                    # Calculate similarity score & loss
                    sim_scores = torch.matmul(
                        q_outputs, torch.transpose(p_outputs, 0, 1)
                    )  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = F.nll_loss(sim_scores, targets)

                # 학습하면서 loss 볼 수 있도록 tqdm postfix 추가
                Ibar.set_postfix(loss=loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()
                global_step += 1

                torch.cuda.empty_cache()

        return self.p_encoder, self.q_encoder

    def get_dense_embedding(self):
        wiki_emb_path = f"../data/wiki_emb.pkl"

        # 이미 임베딩이 존재하는지 확인
        if os.path.isfile(wiki_emb_path):
            with open(wiki_emb_path, "rb") as file:
                self.p_embs = pickle.load(file)
            print("Load Saved Wiki p embs.")

        # 없으면 임베딩 만들고 객체 저장
        else:
            print("Building New Wiki p embs.")
            eval_batch_size = 8
            valid_p_seqs = self.tokenizer(
                self.contexts,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            valid_dataset = TensorDataset(
                valid_p_seqs["input_ids"],
                valid_p_seqs["attention_mask"],
                valid_p_seqs["token_type_ids"],
            )
            valid_sampler = SequentialSampler(valid_dataset)
            valid_dataloader = DataLoader(
                valid_dataset, sampler=valid_sampler, batch_size=eval_batch_size
            )

            p_embs = []
            with torch.no_grad():
                epoch_iterator = tqdm(
                    valid_dataloader, desc="Iteration", position=0, leave=True
                )
                self.p_encoder.eval()

                for _, batch in enumerate(epoch_iterator):
                    batch = tuple(t.cuda() for t in batch)

                    p_inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }

                    outputs = self.p_encoder(**p_inputs).to("cpu").numpy()
                    p_embs.extend(outputs)

                p_embs = np.array(p_embs)
                print(f"(num_passage, emb_dim) : {p_embs.shape}")

            self.p_embs = p_embs
            with open(wiki_emb_path, "wb") as file:
                pickle.dump(p_embs, file)

    # wiki context에서 query와 유사한 k개의 문서를 반환해주는 함수
    def get_relevant_docs(self, query, k):
        with torch.no_grad():
            self.p_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query], padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = self.q_encoder(**q_seqs_val).to("cuda")  # (num_query, emb_dim)

        p_embs = (
            torch.Tensor(self.p_embs).squeeze().to("cuda")
        )  # (num_passage, emb_dim)
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))

        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        relevant_doc = [self.contexts[rank[i]] for i in range(k)]

        # 유사한 문서 list, 점수, 순위 반환
        return relevant_doc, dot_prod_scores, rank

    def retrieve(self, query_or_dataset, topk=5):
        total = []
        for example in query_or_dataset:
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context": " ".join(
                    self.get_relevant_docs(example["question"], topk)[
                        0
                    ]  # 쿼리와 관련한 문서 리스트 가져오는 부분
                ),
            }
            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]

            # 평가를 위한 df column 만들어주기
            relevant_doc_for_df = self.get_relevant_docs(example["question"], topk)[0]
            for i in range(len(relevant_doc_for_df)):
                tmp[f"context{i + 1}"] = relevant_doc_for_df[i]

        total.append(tmp)

        # inference를 위한 df 반환
        cqas = pd.DataFrame(total)
        return cqas

    # 추후에 인코더를 학습시키지 않고 바로 사용할 수 있도록 저장
    def save_trained_encoders(self, p_encoder_path, q_encoder_path):
        self.p_encoder.save_pretrained(p_encoder_path)
        self.q_encoder.save_pretrained(q_encoder_path)

    def set_models(self):
        if self.model_name_or_path.find("roberta") != -1:
            print("Using Roberta Model")
            self.isRoberta = True
        else:
            self.isRoberta = False

        if self.model_name_or_path.find("bart") != -1:
            print("Using Bart Model")
            self.isBart = True
        else:
            self.isBart = False

        if self.model_name_or_path.find("t5") != -1:
            print("Using T5 Model")
            self.isT5 = True
        else:
            self.isT5 = False

        if self.model_name_or_path.find("electra") != -1:
            print("Using Electra Model")
            self.isElectra = True
        else:
            self.isElectra = False

        if (
            self.model_name_or_path.find("SBERT") != -1
            or self.model_name_or_path.find("sbert") != -1
        ):
            print("Using SBert Model")
            self.isSBert = True
        else:
            self.isSBert = False


if __name__ == "__main__":
    wandb_name = "13_dense_validation_test"

    wandb.init(
        project="nlp07_mrc11",
        name=wandb_name
        + "_"
        + datetime.datetime.now(timezone("Asia/Seoul")).strftime("%m/%d %H:%M"),
    )

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="klue/bert-base",
        type=str,
        help="",
    )
    parser.add_argument(
        "--data_path", default="../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--context_path", default="../data/wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--use_faiss", default=False, type=bool, help="")
    parser.add_argument(
        "--p_encoder_ckpt",
        default="./encoders/p_encoder/",
        type=str,
        help="",
    )
    parser.add_argument(
        "--q_encoder_ckpt",
        default="./encoders/q_encoder/",
        type=str,
        help="",
    )

    args = parser.parse_args()
    print(args)
    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    retriever = DenseRetriever(
        data_path=args.data_path,
        context_path=os.path.join(args.data_path, args.context_path),
        model_name_or_path=args.model_name_or_path,
        p_encoder_ckpt=args.p_encoder_ckpt,
        q_encoder_ckpt=args.q_encoder_ckpt,
        stage="test",
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:
        # test single query
        retriever.get_sparse_embedding()
        retriever.build_faiss()
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            top_k = 50  # ground truth 를 확인할 passage 개수
            term = 5  # term 단위로 확인함

            # 10 에서 부터 1/2 씩 감소 시키면서 계산 check_passage_cnt, term
            weight = [(1 / 2**i) for i in range(0, top_k // term)]

            retriever.get_dense_embedding()
            df = retriever.retrieve(full_ds, top_k)

            df.to_csv("dense_result.csv", index=False)

            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

            # # check retrieval
            retieval_logging.retrieval_check(df, weight, top_k, term)

        # 단일 쿼리 에러가 나서 주석처리
        # with timer("single query by exhaustive search"):
        #     scores, indices = retriever.retrieve(query)
