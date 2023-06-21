from transformers import AutoModelForQuestionAnswering, AutoTokenizer, HfArgumentParser
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import json
from tqdm import tqdm
import re
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="boostcamp-5th-nlp07/klue-roberta-large-reader-noNewline",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    tokenizer_name: Optional[str] = field(
        default="klue/roberta-large",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    retrieval_data_path: str = field(
        default="./retrieval.csv",
        metadata={"help": "Path to load retrieval result"},
    )
    output_dir: str = field(
        default="./outputs/split_retrieval_result",
        metadata={"help": "Path to save inference result"},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "Number of nbest"},
    )


class QADataset(Dataset):
    def __init__(self, p_inputs):
        self.input_ids = p_inputs["input_ids"]
        self.attention_mask = p_inputs["attention_mask"]
        self.offset_mapping = p_inputs["offset_mapping"]
        self.original_context_idx = p_inputs["original_context_idx"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index].to("cuda")
        attention_mask = self.attention_mask[index].to("cuda")
        offset_mapping = self.offset_mapping[index].to("cuda")
        original_context_idx = self.original_context_idx[index]
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "original_context_idx": original_context_idx,
        }
        return item


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # 모델과 토크나이저 불러오기
    model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = model.to("cuda").eval()

    # 결과 csv 가져오기
    df = pd.read_csv(data_args.retrieval_data_path)

    # 불러오는 속도를 위해 배열로 미리 불러오기
    questions = df["question"]
    contexts = [eval(c) for c in df["context"]]
    q_ids = df["id"]

    # 파라미터
    topk = len(contexts[0])
    batch_size = 8  # 16부터 터져요ㅜ
    os.makedirs(data_args.output_dir, exist_ok=True)
    answer_dict = {}
    n_best_dict = {}

    # 답변 구하기
    p_bar = tqdm(range(len(df)))
    for q_idx in p_bar:
        # 테스트할 문항 하나씩 가져오기
        answer = ""
        query = questions[q_idx]
        context = contexts[q_idx]  # [passage-1, passage-2 , ... , passage-k]

        input_ids = None
        attention_mask = None
        offset_mapping = None
        original_context_idx = []

        # query와 각 context를 합쳐 각각 토큰화 후 하나의 tensor로 concat
        # context끼리 concat 되는 일 없이 한 텐서 안에는 단일 passage만 존재
        for k in range(topk):
            tokens = tokenizer(
                query,
                context[k],
                truncation="only_second",
                max_length=data_args.max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            if k == 0:
                input_ids = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]
                offset_mapping = tokens["offset_mapping"]
                # truncation 되면 여러 passage가 생기므로 그 개수만큼 context_idx 추가
                original_context_idx.extend(
                    [k for _ in range(len(tokens["input_ids"]))]
                )
            else:
                input_ids = torch.concat((input_ids, tokens["input_ids"]))
                attention_mask = torch.concat(
                    (attention_mask, tokens["attention_mask"])
                )
                offset_mapping = torch.concat(
                    (offset_mapping, tokens["offset_mapping"])
                )
                original_context_idx.extend(
                    [k for _ in range(len(tokens["input_ids"]))]
                )

        # 입력 데이터 구성
        input_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "original_context_idx": original_context_idx,
        }

        # 데이터셋 및 데이터로더 구성
        valid_dataset = QADataset(input_data)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
        n_best_list = []
        max_logit = 0
        for batch_idx, batch in enumerate(valid_dataloader):
            # 모델에 안 들어가도 될 입력(답변 구할 때 필요한 정보)은 pop하여 빼주기
            original_context_idx = batch.pop("original_context_idx")
            offset_mapping = batch.pop("offset_mapping")

            # 모델 forward
            outputs = model(**batch)

            # 배치마다 담긴 길이가 다르므로 구해주기
            batch_len = len(outputs["start_logits"])

            # 각 query+passage 쌍에서의 start와 end의 max, argmax 구하기
            # 총 batch_len개의 max, argmax 존재
            s_max = outputs["start_logits"].max(dim=1)
            e_max = outputs["end_logits"].max(dim=1)

            # 각 query+passage 쌍에서 답변 확률과 위치 구하기
            for idx in range(batch_len):
                # 원래 토큰으로 돌리기 위한 offset
                offsets = offset_mapping[idx]

                # span의 확률
                start_logit = s_max.values[idx].item()
                end_logit = e_max.values[idx].item()
                logit = start_logit + end_logit
                s_pos = offsets[s_max.indices[idx].item()][0]
                e_pos = offsets[e_max.indices[idx].item()][1]
                original_context = context[original_context_idx[idx]]
                text = original_context[s_pos:e_pos]

                result = {
                    "start_logit": start_logit,
                    "end_logit": end_logit,
                    "text": text,
                    "score": start_logit + end_logit,
                }
                n_best_list.append(result)

                if max_logit < logit:
                    # 답변의 길이가 0이거나 [CLS]토큰이 답변이 된 케이스들 제외
                    if s_pos == e_pos:
                        continue

                    # 끝나는 위치가 시작점보다 앞에 위치한 케이스 제외
                    if e_pos < s_pos:
                        continue

                    # 너무 긴 답변 제외
                    if e_pos - s_pos > data_args.max_answer_length:
                        continue

                    max_logit = logit
                    answer = original_context[s_pos:e_pos]

        # GPU 공간을 위해 cache 비워주기
        torch.cuda.empty_cache()

        # answer 후처리
        answer = answer.strip()
        answer = re.sub(r"\\", "", answer)
        answer = re.sub(r'""?', '"', answer)
        answer = re.sub(r'^"|"$', "", answer)

        # 진행 상황 볼 수 있게 postfix로 답변 보여주기
        p_bar.set_postfix(answer=answer)
        # n_best_list.sort(key=lambda x: x["score"])
        # n_best_list = n_best_list[:n_best_size]
        predictions = sorted(n_best_list, key=lambda x: x["score"], reverse=True)[
            : data_args.n_best_size
        ]

        scores = np.array([x.pop("score") for x in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # 답변 추가하기
        answer_dict[q_ids[q_idx]] = answer
        n_best_dict[q_ids[q_idx]] = predictions

    # 답변 저장하기
    with open(
        os.path.join(data_args.output_dir, "predictions.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(answer_dict, f, indent=4, ensure_ascii=False)
    with open(
        os.path.join(data_args.output_dir, "nbest_predictions.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(n_best_dict, f, indent=4, ensure_ascii=False)
