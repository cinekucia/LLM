import os
import json
import re
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Configuration & Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Local Qwen/HuggingFace Scorer")
    parser.add_argument("--input", required=True, help="Input CSV or TSV file")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--rubric-keys", default="content,organization,language", help="Comma-separated keys")
    
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID or path")
    
    # Generation params
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    
    parser.add_argument("--max-rows", type=int, default=None)
    
    return parser.parse_args()

# -----------------------------
# Parsing Helper
# -----------------------------
def _extract_last_json_object(text: str) -> dict:
    text = (text or "").strip()
    try: return json.loads(text)
    except json.JSONDecodeError: pass

    # Regex for markdown JSON
    m = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try: return json.loads(m[-1])
        except: pass

    # Regex for braced block
    try:
        start_idx = text.rfind("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return json.loads(text[start_idx : end_idx + 1])
    except: pass

    return None # Return None if failed, handle later

def _validate_scores(obj: dict, required_keys: list) -> dict:
    if not obj: return {}
    out = {}
    for k in required_keys:
        if k in obj:
            try:
                # Force to float
                out[k] = float(obj[k])
            except:
                pass
    return out

# -----------------------------
# Prompt Construction
# -----------------------------
def build_messages(system_prompt, essay_text, prompt_text, rubric_keys):
    # Truncate safety
    if len(essay_text) > 250000:
        essay_text = essay_text[:250000] + "\n[TRUNCATED]"

    # 1. System: Inject Schema
    json_schema = json.dumps({k: "number (1.0-5.0)" for k in rubric_keys}, indent=2)
    sys_content = (
        f"{system_prompt}\n\n"
        f"STRICT OUTPUT RULES:\n"
        f"1. Output ONLY valid JSON.\n"
        f"2. Follow this schema:\n```json\n{json_schema}\n```"
    )

    # 2. User: Sandwich Method
    # Clean tags
    safe_essay = essay_text.replace("<essay_content>", "").replace("</essay_content>", "")
    
    if prompt_text:
        safe_prompt = prompt_text.replace("<prompt>", "").replace("</prompt>", "")
        user_content = (
            f"Please score this essay.\n\n"
            f"TOPIC:\n<prompt>\n{safe_prompt}\n</prompt>\n\n"
            f"ESSAY:\n<essay_content>\n{safe_essay}\n</essay_content>\n\n"
            f"REMINDER: Output strictly valid JSON. Keys: {', '.join(rubric_keys)}."
        )
    else:
        user_content = (
            f"Please score this essay.\n\n"
            f"ESSAY:\n<essay_content>\n{safe_essay}\n</essay_content>\n\n"
            f"REMINDER: Output strictly valid JSON. Keys: {', '.join(rubric_keys)}."
        )

    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content}
    ]

# -----------------------------
# Main Logic
# -----------------------------
def main():
    args = parse_args()
    rubric_keys = [k.strip() for k in args.rubric_keys.split(",")]

    # 1. Setup Paths
    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    prompt_stem = os.path.splitext(os.path.basename(args.prompt_file))[0]
    model_stem = args.model.split("/")[-1].replace(":", "-")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"{input_stem}_{model_stem}_{prompt_stem}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    # 2. Load Prompts
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        base_system_prompt = f.read().strip()

    # 3. Load Data & Normalize Columns
    try:
        if args.input.lower().endswith(".tsv"):
            df = pd.read_csv(args.input, sep="\t")
        else:
            df = pd.read_csv(args.input)
            
        # Normalization Logic (ASAP / DREsS / ELLIPSE)
        if "essay_id" in df.columns and "id" not in df.columns: df["id"] = df["essay_id"]
        if "full_text" in df.columns and "essay" not in df.columns: df["essay"] = df["full_text"]
        if "assignment" in df.columns and "prompt" not in df.columns: df["prompt"] = df["assignment"]
        
        if "essay" not in df.columns:
            raise ValueError(f"Input must contain 'essay' or 'full_text'. Found: {list(df.columns)}")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Slice if needed
    if args.max_rows:
        df = df.head(args.max_rows)

    # 4. Load Model (Transformer Logic)
    print(f"Loading Model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left" # Critical for batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded.")

    # 5. Prepare Results Holder
    # Check for existing
    completed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
            for r in existing_data.get("rows", []):
                if r.get("LLM_status") == "scored":
                    completed_ids.add(str(r["id"]))
    
    # Filter rows to process
    rows_to_process = []
    for i, row in enumerate(df.itertuples(index=False)):
        rid = str(getattr(row, "id", f"row_{i}"))
        if rid in completed_ids: continue
        rows_to_process.append(row)

    print(f"Processing {len(rows_to_process)} rows in batches of {args.batch_size}...")

    # Load existing results to append
    final_rows = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            final_rows = json.load(f).get("rows", [])

    # -----------------------------
    # BATCH LOOP
    # -----------------------------
    col_map = {c.lower(): c for c in df.columns}

    # Process chunks
    for i in tqdm(range(0, len(rows_to_process), args.batch_size), desc="LLM Scoring"):
        batch_rows = rows_to_process[i : i + args.batch_size]
        
        # A. Prepare Batch Prompts
        batch_prompts_formatted = []
        batch_metadata = []

        for row in batch_rows:
            # Prepare metadata for final result
            row_dict = row._asdict()
            res = { "id": str(getattr(row, "id", "unknown")), "essay": row.essay }
            if hasattr(row, 'prompt') and not pd.isna(row.prompt): res["prompt"] = row.prompt
            
            # Save original scores
            for k in rubric_keys:
                if k in col_map: res[k] = row_dict.get(col_map[k])
            if "total" in col_map: res["total"] = row_dict.get(col_map["total"])
            if "score" in col_map: res["score"] = row_dict.get(col_map["score"])

            # Prepare Messages
            p_text = str(row.prompt) if hasattr(row, 'prompt') and not pd.isna(row.prompt) else None
            messages = build_messages(base_system_prompt, str(row.essay), p_text, rubric_keys)
            
            # Apply Template
            text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            batch_prompts_formatted.append(text_prompt)
            batch_metadata.append(res)

        # B. Tokenize & Generate
        if not batch_prompts_formatted: continue

        inputs = tokenizer(batch_prompts_formatted, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=(args.temperature > 0),
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            )

        # C. Decode
        # Only decode the *new* tokens (slice input length)
        input_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # D. Parse & Update Results
        for res_obj, raw_text in zip(batch_metadata, decoded_responses):
            json_obj = _extract_last_json_object(raw_text)
            
            if json_obj:
                scores = _validate_scores(json_obj, rubric_keys)
                res_obj["LLM_scores"] = scores
                if scores:
                    res_obj["LLM_total"] = round(sum(scores.values()) / len(scores), 2)
                res_obj["LLM_status"] = "scored"
            else:
                res_obj["LLM_status"] = "failed_parse"
                res_obj["LLM_error_message"] = raw_text[:200] # Log snippet

            final_rows.append(res_obj)

        # E. Save Atomic (after every batch)
        temp_path = output_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({"meta": vars(args), "rows": final_rows}, f, indent=2)
        os.replace(temp_path, output_path)

    print(f"\nDone. Saved to {output_path}")

if __name__ == "__main__":
    main()