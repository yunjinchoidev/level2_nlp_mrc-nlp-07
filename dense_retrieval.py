from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertPreTrainedModel,
    AdamW,
    RobertaModel,
    RobertaPreTrainedModel,
    BartModel,
    BartPretrainedModel,
    get_linear_schedule_with_warmup,
    T5Model,
    T5PreTrainedModel,
    ElectraPreTrainedModel,
    ElectraModel,
)
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
import faiss
import os
import json
import pickle
import pandas as pd

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)


class RoBertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RoBertaEncoder, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output


class BartEncoder(BartPretrainedModel):
    def __init__(self, config):
        super(BartEncoder, self).__init__(config)

        self.bart = BartModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bart(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output


class T5Encoder(T5PreTrainedModel):
    def __init__(self, config):
        super(T5Encoder, self).__init__(config)

        self.t5 = T5Model(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.t5(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output


class ElectraEncoder(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraEncoder, self).__init__(config)

        self.electra = ElectraModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output


class DenseRetriever:
    def __init__(
        self,
        data_path,
        context_path,
        model_name_or_path,
        p_encoder_ckpt,
        q_encoder_ckpt,
        stage="train",
        use_neg_sampling=False,
        num_neg=3,
    ):
        self.model_name_or_path = model_name_or_path
        self.stage = stage
        self.num_neg = num_neg
        self.use_neg_sampling = use_neg_sampling

        # 모델에 따라서 변수 세팅
        if self.model_name_or_path.find("roberta") != -1:
            self.isRoberta = True
        else:
            self.isRoberta = False

        if self.model_name_or_path.find("bart") != -1:
            self.isBart = True
        else:
            self.isBart = False

        if self.model_name_or_path.find("t5") != -1:
            self.isT5 = True
        else:
            self.isT5 = False

        if self.model_name_or_path.find("electra") != -1:
            self.isElectra = True
        else:
            self.isElectra = False

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
                corpus = list(set(training_dataset["context"]))
                corpus = np.array(corpus)
                p_with_neg = []
                for c in training_dataset["context"]:
                    while True:
                        neg_idxs = np.random.randint(len(corpus), size=num_neg)
                        if not c in corpus[neg_idxs]:
                            p_neg = corpus[neg_idxs]
                            p_with_neg.append(c)
                            p_with_neg.extend(p_neg)
                            break
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
            self.p_encoder = RoBertaEncoder.from_pretrained(p_encoder_ckpt)
            self.q_encoder = RoBertaEncoder.from_pretrained(q_encoder_ckpt)

        elif self.isBart:
            self.p_encoder = BartEncoder.from_pretrained(p_encoder_ckpt)
            self.q_encoder = BartEncoder.from_pretrained(q_encoder_ckpt)

        elif self.isT5:
            self.p_encoder = T5Encoder.from_pretrained(p_encoder_ckpt)
            self.q_encoder = T5Encoder.from_pretrained(q_encoder_ckpt)

        elif self.isElectra:
            self.p_encoder = ElectraEncoder.from_pretrained(p_encoder_ckpt)
            self.q_encoder = ElectraEncoder.from_pretrained(q_encoder_ckpt)

        else:
            self.p_encoder = BertEncoder.from_pretrained(p_encoder_ckpt)
            self.q_encoder = BertEncoder.from_pretrained(q_encoder_ckpt)

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
        train_sampler = RandomSampler(self.train_dataset)
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
            total.append(tmp)

        # inference를 위한 df 반환
        cqas = pd.DataFrame(total)
        return cqas

    # 추후에 인코더를 학습시키지 않고 바로 사용할 수 있도록 저장
    def save_trained_encoders(self, p_encoder_path, q_encoder_path):
        self.p_encoder.save_pretrained(p_encoder_path)
        self.q_encoder.save_pretrained(q_encoder_path)
