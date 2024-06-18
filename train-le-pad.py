# Code modified from: https://github.com/havenhq/mamba-chat/blob/main/train_mamba.py

import torch
import argparse
import transformers
import json
import os
import random
import dataset
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import Dataset

from transformers import AutoTokenizer, TrainingArguments
from transformers import Trainer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super(SFTDataset, self).__init__()
        data = []
        print(f"Reading in data from file: {data_path}")
        if data_path.split(".")[-1] == "jsonl":
            with open(data_path, "r") as file:
                for line in file:
                    try:
                        data.append(json.loads(line))
                    except Exception as e:
                        print("json processing exception", e)
                        continue
        elif data_path.split(".")[-1] == "json":
            with open(data_path, "r") as file:
                for line in file:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"JSON decoding error: {e}")

        print(f"Got {len(data)} examples, preprocess...")
        data_dict = self.preprocess(data, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.label_ids = data_dict["label_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], label_ids=self.label_ids[i])

    def preprocess(self, examples, tokenizer):
        """
        Preprocess the data by tokenizing.
        """
        all_input_ids, all_label_ids = [], []

        print("Tokenizing dataset...")
        for ex in tqdm(examples):
            # Add a positive example
            # text = f"{ex['context']}\n\nQ: {ex['prompt']}\nA: {ex['response']}\n"
            # Truncate the text if it's too long
            # mantain the context's head and tail
            context = ex['context']
            if len(context) > 1024*30:
                context = context[:1024*15] + context[-1024*15:]
            context = f"Summarize the content into a few short sentences. Content:\n{context}\n\nSummary:\n"
            context = "Summarize "
            label = f"{ex['response']}{tokenizer.eos_token}"
            label = "the"
            context_id = tokenizer.encode(context)
            label_id = tokenizer.encode(label)
            all_input_ids.append(torch.LongTensor(context_id))
            all_label_ids.append(torch.LongTensor(label_id))


        return dict(input_ids=all_input_ids, label_ids=all_label_ids)


@dataclass
class DataCollatorForSFTDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        instruction_ids, answer_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "label_ids"))
        input_ids = []
        label_ids = []
        for instruction, answer in zip(instruction_ids, answer_ids):
            input_ids.append(torch.cat([instruction, answer], dim=-1))
            label_ids.append(torch.cat([torch.tensor([-100]*instruction.shape[-1]), answer], dim=-1))
        #input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
        #                                            padding_value=self.tokenizer.pad_token_id)
        #label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)
        # max length of the input_ids
        max_len = max([len(ids) for ids in input_ids])
        # left padding
        for i in range(len(input_ids)):
            input_ids[i] = torch.cat([torch.tensor([self.tokenizer.pad_token_id] * (max_len - len(input_ids[i])), dtype=torch.long), input_ids[i]])
            label_ids[i] = torch.cat([torch.tensor([-100] * (max_len - len(label_ids[i])), dtype=torch.long), label_ids[i]])
        # convert to tensor
        input_ids = torch.stack(input_ids)
        label_ids = torch.stack(label_ids)
        return dict(
            input_ids=input_ids,
            label_ids=label_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SFTDataModule():
    def __init__(self, tokenizer, data_path: str):
        self.dataset = SFTDataset(tokenizer=tokenizer, data_path=data_path)
        self.data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)


class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # input_ids = inputs.pop("input_ids")
        # label_ids = inputs.pop("label_ids")
        # labels = label_ids.to(input_ids.device)
        # concat_ids = torch.cat([input_ids, label_ids], dim=-1)
        # lm_logits = model(concat_ids).logits
        # shift_logits = lm_logits[:, input_ids.shape(1):, :]
        # loss_fct = torch.nn.CrossEntropyLoss()
        # q = shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        # lm_loss = loss_fct(q)
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits
        labels = inputs.pop("label_ids").to(input_ids.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        return lm_loss

    def save_model(self, output_dir, _internal_call=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)

        # https://huggingface.co/state-spaces/mamba-130m/blob/main/config.json
        json_str = """
{
    "d_model": 768,
    "n_layer": 24,
    "vocab_size": 50277,
    "ssm_cfg": {},
    "rms_norm": true,
    "residual_in_fp32": true,
    "fused_add_norm": true,
    "pad_vocab_size_multiple": 8
}"""
        with open(f"{output_dir}/config.json", 'w') as f:
            f.write(json_str)


def run(args):
    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    data_module = SFTDataModule(
        tokenizer=tokenizer,
        data_path=args.data_path,
    )

    trainer = MambaTrainer(
        model=model,
        train_dataset=data_module.dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output,
            save_total_limit=2,
            logging_steps=50,
            save_steps=500,
        ),
        data_collator=data_module.data_collator,
    )

    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-1.4b")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="./dataset/SummScreen/TVMegaSite/tms_train.jsonl")
    parser.add_argument("--num_epochs", type=int, default=3)
    args = parser.parse_args()

    run(args)