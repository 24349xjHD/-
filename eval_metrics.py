# scripts/eval_metrics.py
import os
import json
import csv
import random
import argparse
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from utils_traj import extract_coords, compute_ade_fde, coords_to_str, compute_recall_like_from_fde, str2bool

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_name", type=str, default="./Qwen3.5-4B")
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--jsonl_path", type=str, required=True)

    p.add_argument("--csv_out", type=str, required=True)
    p.add_argument("--json_out", type=str, required=True)

    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--save_text", type=str, default="false")  # true/false
    p.add_argument("--thresholds", type=str, default="1.0,2.0,3.0")
    return p

def main():
    args = build_argparser().parse_args()

    base_model_name = args.base_model_name
    adapter_dir = args.adapter_dir
    jsonl_path = args.jsonl_path
    CSV_OUT = args.csv_out
    JSON_OUT = args.json_out

    NUM_SAMPLES = args.num_samples
    SEED = args.seed
    MAX_NEW_TOKENS = args.max_new_tokens
    SAVE_TEXT = str2bool(args.save_text)

    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip() != "")

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    max_memory = {
        0: "20GiB",  # 4090 24GB：给点余量，先用 20GiB
        "cpu": "64GiB"
    }

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    if len(samples) < NUM_SAMPLES:
        raise ValueError(f"jsonl 样本数不足：len={len(samples)} < NUM_SAMPLES={NUM_SAMPLES}")

    picked = random.sample(samples, NUM_SAMPLES)

    rows = []
    ok_cnt = 0
    ade_list = []
    fde_list = []
    success_parse_cnt = 0

    for i, data in enumerate(picked):
        test_input = data["input"]
        true_traj = extract_coords(data["output"])

        # ground truth 异常：不满足 6 点
        if len(true_traj) != 6:
            row = {
                "idx": i,
                "ade": "",
                "fde": "",
                "pred_len": "",
                "true_len": len(true_traj),
                "pred_coords": "",
                "true_coords": coords_to_str(true_traj),
                "parse_ok": False,
            }
            if SAVE_TEXT:
                row["input"] = test_input
            rows.append(row)
            continue

        prompt = f"### 输入:\n{test_input}\n### 输出:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        pred_traj = extract_coords(pred_text)
        if len(pred_traj) > 6:
            pred_traj = pred_traj[:6]

        # prediction 解析不出 6 点
        if len(pred_traj) != 6:
            row = {
                "idx": i,
                "ade": "",
                "fde": "",
                "pred_len": len(pred_traj),
                "true_len": len(true_traj),
                "pred_coords": coords_to_str(pred_traj),
                "true_coords": coords_to_str(true_traj),
                "parse_ok": False,
            }
            if SAVE_TEXT:
                row["input"] = test_input
            rows.append(row)
            continue

        # 成功解析
        success_parse_cnt += 1
        ade, fde = compute_ade_fde(pred_traj, true_traj)
        ok_cnt += 1
        ade_list.append(ade)
        fde_list.append(fde)

        row = {
            "idx": i,
            "ade": float(ade),
            "fde": float(fde),
            "pred_len": len(pred_traj),
            "true_len": len(true_traj),
            "pred_coords": coords_to_str(pred_traj),
            "true_coords": coords_to_str(true_traj),
            "parse_ok": True,
        }
        if SAVE_TEXT:
            row["input"] = test_input
        rows.append(row)

        if (i + 1) % 10 == 0:
            print(f"[EVAL] {i+1}/{NUM_SAMPLES} ok | ADE={ade:.4f}, FDE={fde:.4f}")

    success_rate = float(ok_cnt / NUM_SAMPLES)
    mean_ade = float(np.mean(ade_list)) if len(ade_list) > 0 else None
    mean_fde = float(np.mean(fde_list)) if len(fde_list) > 0 else None

    recall_like = compute_recall_like_from_fde(fde_list, thresholds=thresholds)

    summary = {
        "num_samples": NUM_SAMPLES,
        "seed": SEED,
        "parse_success_rate": success_rate,  # 与 ok_cnt 对齐（ground truth 和 pred 都成功）
        "mean_ade": mean_ade,
        "mean_fde": mean_fde,
        **recall_like,
    }

    # 写 CSV
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    if SAVE_TEXT:
        fieldnames = ["idx", "input", "ade", "fde", "pred_len", "true_len", "pred_coords", "true_coords", "parse_ok"]
    else:
        fieldnames = ["idx", "ade", "fde", "pred_len", "true_len", "pred_coords", "true_coords", "parse_ok"]

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 写 JSON
    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("==== Eval Done ====")
    print("CSV:", CSV_OUT)
    print("JSON:", JSON_OUT)
    print("Summary:", summary)

if __name__ == "__main__":
    main()