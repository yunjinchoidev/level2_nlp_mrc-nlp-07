from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import transformers
from transformers import (
    BertModel,
    BertPreTrainedModel,
    get_linear_schedule_with_warmup,
    BertConfig,
)
import os
import pickle

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler


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


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls,
        cfg_name: str = "klue/bert-base",
        projection_dim: int = 0,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        cfg = BertConfig.from_pretrained(cfg_name)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim)
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        representation_token_pos=0,
    ):
        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert (
                representation_token_pos.size(0) == bsz
            ), "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack(
                [
                    sequence_output[i, representation_token_pos[i, 1], :]
                    for i in range(bsz)
                ]
            )

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        # return sequence_output, pooled_output, hidden_states
        return pooled_output


def hard_train(
    train_with_hard_example="../../data/train_with_one_neg",
    encoder_base_model="klue/bert-base",
):
    datasets = load_from_disk(train_with_hard_example)
    print("Dataset Loaded")
    train_data = datasets["train"]
    valid_data = datasets["validation"]
    # p_encoder = BertEncoder.from_pretrained(encoder_base_model).to("cuda")
    # q_encoder = BertEncoder.from_pretrained(encoder_base_model).to("cuda")
    p_encoder = HFBertEncoder.init_encoder(cfg_name=encoder_base_model).to("cuda")
    q_encoder = HFBertEncoder.init_encoder(cfg_name=encoder_base_model).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(encoder_base_model)
    questions = train_data["question"]
    true_passage = train_data["context"]
    negative_passage = train_data["negative_examples"]
    N = len(true_passage)
    batch_size = 8
    num_neg = 1
    epoch = 10
    ckpt_save_dir = "../encoders/HFBert_HardRand_loss_norm2-e10"
    warmup_ratio = 0.3

    trainset = []
    for idx in tqdm(range(N)):
        trainset.append(true_passage[idx])
        trainset.append(negative_passage[idx])

    p_seqs = tokenizer(
        trainset,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    max_len = p_seqs["input_ids"].size(-1)
    p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
    p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg + 1, max_len)
    p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg + 1, max_len)

    q_seqs = tokenizer(
        questions,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    train_dataset = TensorDataset(
        p_seqs["input_ids"],
        p_seqs["attention_mask"],
        p_seqs["token_type_ids"],
        q_seqs["input_ids"],
        q_seqs["attention_mask"],
        q_seqs["token_type_ids"],
    )

    val_q_seqs = tokenizer(
        valid_data["question"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    val_p_seqs = tokenizer(
        valid_data["context"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    valid_dataset = TensorDataset(
        val_p_seqs["input_ids"],
        val_p_seqs["attention_mask"],
        val_p_seqs["token_type_ids"],
        val_q_seqs["input_ids"],
        val_q_seqs["attention_mask"],
        val_q_seqs["token_type_ids"],
    )
    print("DataLoader set")

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

    t_total = len(trainset) * epoch // batch_size

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * t_total),
        num_training_steps=t_total,
    )

    p_encoder.zero_grad()
    q_encoder.zero_grad()
    torch.cuda.empty_cache()

    # random.shuffle(trainset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=RandomSampler(valid_dataset),
        batch_size=8,
    )

    min_val_loss = 50
    for e in range(epoch):
        print(f"Epoch {e}")
        Ibar = tqdm(train_dataloader)
        hard_loss_sum = 0
        rand_loss_sum = 0
        cnt = 0
        for batch in Ibar:
            q_encoder.train()
            p_encoder.train()

            # Set Targets
            hard_targets = torch.zeros(batch_size).long()
            rand_targets = torch.arange(0, batch_size).long()

            if torch.cuda.is_available():
                hard_targets = hard_targets.to("cuda")
                rand_targets = rand_targets.to("cuda")
                batch = tuple(t.cuda() for t in batch)

            # Set inputs
            try:
                p_hard_inputs = {
                    "input_ids": batch[0].view(batch_size * (num_neg + 1), -1),
                    "attention_mask": batch[1].view(batch_size * (num_neg + 1), -1),
                    "token_type_ids": batch[2].view(batch_size * (num_neg + 1), -1),
                }
            except:
                break

            p_rand_inputs = {
                "input_ids": batch[0][:, 0],
                "attention_mask": batch[1][:, 0],
                "token_type_ids": batch[2][:, 0],
            }
            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }

            # Encoder forward
            q_outputs = q_encoder(**q_inputs)
            p_hard_outputs = p_encoder(**p_hard_inputs)
            p_rand_outputs = p_encoder(**p_rand_inputs)

            # Hard examples
            p_hard_outputs = p_hard_outputs.view(batch_size, num_neg + 1, -1)
            q_hard_outputs = q_outputs.view(batch_size, -1, 1)
            hard_sim_scores = torch.bmm(
                p_hard_outputs, q_hard_outputs
            ).squeeze()  # (batch_size, num_neg+1)
            hard_sim_scores = hard_sim_scores.view(batch_size, -1)

            hard_sim_scores = F.log_softmax(hard_sim_scores, dim=1)
            hard_loss = F.nll_loss(hard_sim_scores, hard_targets)

            # Random in-batch examples
            rand_sim_scores = torch.matmul(
                q_outputs, torch.transpose(p_rand_outputs, 0, 1)
            )
            rand_sim_scores = F.log_softmax(rand_sim_scores, dim=1)
            rand_loss = F.nll_loss(rand_sim_scores, rand_targets)

            # Sum Loss
            total = num_neg + batch_size - 1
            # loss = hard_loss * (num_neg + 1) / total + rand_loss * batch_size / total
            loss = hard_loss * (batch_size - 1) / total + rand_loss * num_neg / total
            # loss = hard_loss + rand_loss
            loss.backward()

            # Set progress bar description
            Ibar.set_postfix(hard_loss=hard_loss.item(), rand_loss=rand_loss.item())
            hard_loss_sum += hard_loss.item()
            rand_loss_sum += rand_loss.item()
            cnt += 1

            # Step
            optimizer.step()
            scheduler.step()
            p_encoder.zero_grad()
            q_encoder.zero_grad()

            torch.cuda.empty_cache()

        print(f"Avg hard loss : {hard_loss_sum / cnt:0.4f}")
        print(f"Avg rand loss : {rand_loss_sum / cnt:0.4f}")

        VIbar = tqdm(valid_dataloader)
        loss_sum = 0
        cnt = 0
        for batch in VIbar:
            q_encoder.eval()
            p_encoder.eval()

            targets = torch.arange(0, 8).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")
                batch = tuple(t.cuda() for t in batch)
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
            p_outputs = p_encoder(
                **p_inputs
            )  # (batch_size, emb_dim) or (batch_size*(num_neg+1), emb_dim)
            q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)

            sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
            sim_scores = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_scores, targets)
            VIbar.set_postfix(loss=loss.item())
            loss_sum += loss.item()
            cnt += 1

            torch.cuda.empty_cache()

        avg_loss = loss_sum / cnt
        print(f"Epoch {e} Val Avg loss : {avg_loss:0.4f}")
        if avg_loss < min_val_loss:
            min_val_loss = avg_loss
            os.makedirs(ckpt_save_dir, exist_ok=True)
            p_encoder.save_pretrained(os.path.join(ckpt_save_dir, "p_encoder"))
            q_encoder.save_pretrained(os.path.join(ckpt_save_dir, "q_encoder"))

    p_encoder.save_pretrained(os.path.join(ckpt_save_dir, "last/p_encoder"))
    q_encoder.save_pretrained(os.path.join(ckpt_save_dir, "last/q_encoder"))


if __name__ == "__main__":
    hard_train()
