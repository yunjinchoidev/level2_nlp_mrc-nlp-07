from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    BertModel,
    BertPreTrainedModel,
    get_linear_schedule_with_warmup,
)
import os
import pickle

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)


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


def hard_train(
    train_with_hard_example="../../data/train_with_neg",
    encoder_base_model="klue/bert-base",
):
    datasets = load_from_disk(train_with_hard_example)["train"]
    p_encoder = BertEncoder.from_pretrained(encoder_base_model).to("cuda")
    q_encoder = BertEncoder.from_pretrained(encoder_base_model).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(encoder_base_model)
    questions = datasets["question"]
    true_passage = datasets["context"]
    negative_passage = datasets["negative_examples"]
    N = len(true_passage)

    trainset_path = "../../data/hard_trainset.pkl"
    if os.path.isfile(trainset_path):
        with open(trainset_path, "rb") as file:
            trainset = pickle.load(file)
        print("Load Saved trainset.")

    else:
        print("Building New hard trainset")
        trainset = []
        for idx in tqdm(range(N)):
            concat = [true_passage[idx]] + negative_passage[idx]
            query = questions[idx]
            p_tokens = tokenizer(
                concat,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            q_tokens = tokenizer(
                query,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            trainset.append([q_tokens, p_tokens])

        with open(trainset_path, "wb") as file:
            pickle.dump(trainset, file)
        print("New trainset saved")

    # 여기부터 학습
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in p_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in p_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in q_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in q_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-08)

    epoch = 2
    t_total = len(trainset) * epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.3 * t_total), num_training_steps=t_total
    )

    p_encoder.zero_grad()
    q_encoder.zero_grad()
    torch.cuda.empty_cache()

    target = torch.Tensor([0]).long().to("cuda")
    random.shuffle(trainset)

    for e in range(epoch):
        print(f"Epoch {e}")
        Ibar = tqdm(trainset)
        loss_sum = 0
        cnt = 0
        for batch_q, batch_p in Ibar:
            q_outputs = q_encoder(**batch_q)
            p_outputs = p_encoder(**batch_p)

            sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
            sim_scores = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_scores, target)
            Ibar.set_postfix(loss=loss.item())
            loss_sum += loss.item()
            cnt += 1

            loss.backward()
            optimizer.step()
            scheduler.step()
            p_encoder.zero_grad()
            q_encoder.zero_grad()

            torch.cuda.empty_cache()

        print(f"Avg loss : {loss_sum / cnt:0.4f}")

    os.makedirs("hard4", exist_ok=True)
    p_encoder.save_pretrained("hard4/p_encoder")
    q_encoder.save_pretrained("hard4/q_encoder")


if __name__ == "__main__":
    hard_train()
