import torch
import numpy as np
import random
from arguments import DataTrainingArguments, ModelArguments
from dense_retrieval import DenseRetriever
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
import os

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)

if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"Base model of encoders : {model_args.encoder_base}")

    data_path = "../data/train_dataset"
    context_path = "../data/wikipedia_documents.json"

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        warmup_ratio=0.2,
        weight_decay=0.01,
    )

    retriever = DenseRetriever(
        data_path=data_path,
        context_path=context_path,
        model_name_or_path=model_args.encoder_base,
        use_neg_sampling=False,
    )

    print("Training Encoders ...")
    retriever.train(args)
    print("Training Done!")

    print("Saving Encoders ...")
    retriever.save_trained_encoders(
        model_args.p_encoder_path, model_args.q_encoder_path
    )
    print("Encoders saved!")
