# scripts/train_lora.py
import os
import re
import sys
import glob
import json
import time
import subprocess
import argparse
import gc

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from utils_traj import str2bool


# =========================
# Argument Parser
# =========================
def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--base_model_name", type=str, default="./Qwen3.5-4B")
    p.add_argument("--train_jsonl", type=str, default="./data/train.jsonl")
    p.add_argument("--test_jsonl", type=str, default="./data/test.jsonl")
    p.add_argument("--output_dir", type=str, default="./output")

    # ✅ 更稳的默认参数
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)

    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # ✅ 更稳 LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    # eval
    p.add_argument("--eval_num_samples", type=int, default=50)
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--eval_max_new_tokens", type=int, default=120)
    p.add_argument("--eval_save_text", type=str, default="false")
    p.add_argument("--eval_thresholds", type=str, default="1.0,2.0,3.0")

    # Early Stop 早停机制
    p.add_argument("--early_stop_patience", type=int, default=2)
    p.add_argument("--early_stop_metric", type=str, default="mean_fde")

    p.add_argument("--resume", type=str, default="true")

    return p


# =========================
# Checkpoint helper
# =========================
def latest_checkpoint(root):
    pattern = os.path.join(root, "checkpoint-*")
    paths = glob.glob(pattern)
    if not paths:
        return None

    def step_id(p):
        m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    paths = sorted(paths, key=step_id)
    return paths[-1]


# =========================
# Callback
# =========================
class SaveAdapterAndEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        base_model_name,
        test_jsonl,
        eval_script_path,
        adapters_dir,
        eval_dir,
        eval_cfg,
        output_dir,
    ):
        self.tokenizer = tokenizer
        self.base_model_name = base_model_name
        self.test_jsonl = test_jsonl
        self.eval_script_path = eval_script_path
        self.adapters_dir = adapters_dir
        self.eval_dir = eval_dir
        self.eval_cfg = eval_cfg
        self.output_dir = output_dir

        self.best_metric = None
        self.no_improve_epochs = 0
        self.patience = eval_cfg["early_stop_patience"]
        self.metric_name = eval_cfg["early_stop_metric"]

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return

        epoch_k = int(state.epoch)
        model = kwargs["model"]

        adapter_dir = os.path.join(self.adapters_dir, f"epoch_{epoch_k}")
        os.makedirs(adapter_dir, exist_ok=True)

        model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)

        print(f"[Epoch {epoch_k}] Adapter saved.")

        # 释放显存
        torch.cuda.empty_cache()
        gc.collect()

        csv_out = os.path.join(self.eval_dir, f"eval_epoch_{epoch_k}.csv")
        json_out = os.path.join(self.eval_dir, f"eval_epoch_{epoch_k}.json")

        cmd = [
            sys.executable,
            self.eval_script_path,
            "--base_model_name", self.base_model_name,
            "--adapter_dir", adapter_dir,
            "--jsonl_path", self.test_jsonl,
            "--csv_out", csv_out,
            "--json_out", json_out,
            "--num_samples", str(self.eval_cfg["num_samples"]),
            "--seed", str(self.eval_cfg["seed"]),
            "--max_new_tokens", str(self.eval_cfg["max_new_tokens"]),
            "--save_text", str(self.eval_cfg["save_text"]),
            "--thresholds", str(self.eval_cfg["thresholds"]),
        ]

        subprocess.run(cmd, check=True)

        with open(json_out, "r", encoding="utf-8") as f:
            summary = json.load(f)

        metric_value = summary.get(self.metric_name, None)
        parse_success = summary.get("parse_success_rate", 0)

        print(f"[Eval] {self.metric_name} = {metric_value}")

        # ===== 防爆保护 =====
        if metric_value is not None and metric_value > 100:
            print(" Metric exploded. Early stop.")
            control.should_training_stop = True
            return control

        if parse_success < 0.2:
            print(" Parse success too low. Early stop.")
            control.should_training_stop = True
            return control

        # ===== Early Stop 判断 =====
        if metric_value is not None:
            if self.best_metric is None or metric_value < self.best_metric:
                self.best_metric = metric_value
                self.no_improve_epochs = 0

                # 保存 best
                best_dir = os.path.join(self.output_dir, "adapter_best")
                model.save_pretrained(best_dir)
                self.tokenizer.save_pretrained(best_dir)

                print(f"New best model saved. {self.metric_name}={metric_value:.4f}")

            else:
                self.no_improve_epochs += 1
                print(f"No improvement ({self.no_improve_epochs}/{self.patience})")

            if self.no_improve_epochs >= self.patience:
                print("Early stopping triggered.")
                control.should_training_stop = True

        return control


# =========================
# Main
# =========================
def main():
    args = build_argparser().parse_args()

    output_dir = args.output_dir
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    adapters_dir = os.path.join(output_dir, "adapters")
    eval_dir = os.path.join(output_dir, "eval")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(adapters_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    target_modules = [x.strip() for x in args.target_modules.split(",")]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, peft_config)

    dataset = load_dataset("json", data_files={"train": args.train_jsonl})

    def build_prompt(inp):
        return f"### 输入:\n{inp}\n### 输出:\n"

    def format_fn(examples):
        texts = [
            build_prompt(i) + o
            for i, o in zip(examples["input"], examples["output"])
        ]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_train = dataset["train"].map(
        format_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    eval_cfg = {
        "num_samples": args.eval_num_samples,
        "seed": args.eval_seed,
        "max_new_tokens": args.eval_max_new_tokens,
        "save_text": args.eval_save_text,
        "thresholds": args.eval_thresholds,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_metric": args.early_stop_metric,
    }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        callbacks=[
            SaveAdapterAndEvalCallback(
                tokenizer,
                args.base_model_name,
                args.test_jsonl,
                os.path.join(SCRIPT_DIR, "eval_metrics.py"),
                adapters_dir,
                eval_dir,
                eval_cfg,
                output_dir,
            )
        ],
    )

    trainer.train()

    final_dir = os.path.join(output_dir, "adapter_final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Training finished.")


if __name__ == "__main__":
    main()