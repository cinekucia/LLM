import os
import json
import re
import time
import argparse
import logging
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm

# -----------------------------
# Configuration Class
# -----------------------------
class ScorerConfig:
    def __init__(self, args):
        self.input_csv = args.input
        self.model_id = args.model
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.save_every_n = args.save_every
        self.max_rows = args.max_rows
        
        self.rubric_keys = [k.strip() for k in args.rubric_keys.split(",")]

        input_stem = os.path.splitext(os.path.basename(self.input_csv))[0]
        prompt_stem = os.path.splitext(os.path.basename(args.prompt_file))[0]
        model_stem = self.model_id.split("/")[-1].replace(":", "-") 
        
        filename = f"results_{input_stem}_{model_stem}_{prompt_stem}.json"
        
        os.makedirs(args.output_dir, exist_ok=True)
        self.output_json = os.path.join(args.output_dir, filename)

        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            self.base_system_prompt = f.read().strip()

        self.json_schema_str = json.dumps({
            "type": "object",
            "properties": {k: {"type": "number", "minimum": 1.0, "maximum": 5.0} for k in self.rubric_keys},
            "required": self.rubric_keys,
            "additionalProperties": False
        }, indent=2)

        self.log_dir = args.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        log_base = f"{input_stem}_{model_stem}_{prompt_stem}"
        
        self.log_all_path = os.path.join(self.log_dir, f"{log_base}_conversation_logs.jsonl")
        self.log_error_path = os.path.join(self.log_dir, f"{log_base}_errors.jsonl")

# -----------------------------
# Detailed Logging Helpers
# -----------------------------
def _append_jsonl(path: str, event: Dict[str, Any]) -> None:
    event["ts"] = time.time()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def _log_all(config: ScorerConfig, event: Dict[str, Any]) -> None:
    _append_jsonl(config.log_all_path, event)

def _log_error(config: ScorerConfig, event: Dict[str, Any]) -> None:
    _append_jsonl(config.log_error_path, event)

# -----------------------------
# Robust JSON Parsing
# -----------------------------
def _extract_last_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try: return json.loads(text)
    except json.JSONDecodeError: pass

    m = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try: return json.loads(m[-1])
        except: pass

    try:
        start_idx = text.rfind("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return json.loads(text[start_idx : end_idx + 1])
    except: pass

    raise ValueError("No valid JSON object found in output.")

def _validate_scores(obj: Dict[str, Any], required_keys: List[str]) -> Dict[str, float]:
    out = {}
    errors = []
    
    for k in required_keys:
        if k not in obj:
            errors.append(f"Missing key: {k}")
            continue
        try:
            val = float(obj[k])
        except (ValueError, TypeError):
            errors.append(f"Key '{k}' is not a number: {obj[k]}")
            continue
            
        if not (1.0 <= val <= 5.0):
            errors.append(f"Key '{k}' out of range (1.0-5.0): {val}")
            
        out[k] = val
        
    if errors:
        raise ValueError("Validation failed: " + "; ".join(errors))
    return out

# -----------------------------
# Core Scoring Function (Transformers)
# -----------------------------
def score_row(pipeline, config: ScorerConfig, row_id: Any, essay_text: str) -> Tuple[Dict[str, float], str]:
    MAX_CHARS = 12000 
    if len(essay_text) > MAX_CHARS:
        essay_text = essay_text[:MAX_CHARS] + "\n[TRUNCATED]"

    system_content = (
        f"{config.base_system_prompt}\n\n"
        f"STRICT OUTPUT RULES:\n"
        f"1. You must output valid JSON only.\n"
        f"2. Do not write introductory text.\n"
        f"3. Follow this JSON schema exactly:\n"
        f"```json\n{config.json_schema_str}\n```"
    )

    safe_essay = essay_text.replace("<essay_content>", "").replace("</essay_content>", "")
    user_content = (
        f"Please score the following essay.\n\n"
        f"<essay_content>\n"
        f"{safe_essay}\n"
        f"</essay_content>\n\n"
        f"REMINDER: Output strictly valid JSON. "
        f"Keys: {', '.join(config.rubric_keys)}. "
        f"Values: 1.0 to 5.0 (0.5 increments)."
    )
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # -------------------------------------------------------
    # Dynamic Generation Args (Conditional)
    # -------------------------------------------------------
    do_sample = (config.temperature > 0)

    # Base arguments always present
    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": do_sample,
        # Belt-and-suspenders: explicitly pass pad token ID here too
        "pad_token_id": pipeline.tokenizer.eos_token_id
    }

    # Only add sampling parameters if sampling is enabled
    if do_sample:
        gen_kwargs["temperature"] = config.temperature
        gen_kwargs["top_p"] = config.top_p
    else:
        # Silence transformer warnings by explicitly unsetting sampling parameters
        # that might be present in the model's default configuration
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None

    last_error = None
    
    for attempt in range(1, 3):
        try:
            _log_all(config, {
                "event": "request", "row_id": row_id, "attempt": attempt
            })
            
            # Pass dictionary as kwargs
            outputs = pipeline(messages, **gen_kwargs)
            
            raw_text = outputs[0]["generated_text"][-1]["content"]
            
            _log_all(config, {
                "event": "response", "row_id": row_id, "attempt": attempt, 
                "raw_output": raw_text
            })
            
            json_obj = _extract_last_json_object(raw_text)
            scores = _validate_scores(json_obj, config.rubric_keys)
            
            req_id = f"local-{row_id}-{attempt}"
            return scores, req_id

        except Exception as e:
            last_error = e
            error_msg = str(e)
            _log_error(config, {"event": "error", "row_id": row_id, "attempt": attempt, "error": error_msg})
            
            if "CUDA out of memory" in error_msg:
                print("\n\nCRITICAL: CUDA Out of Memory.")
                raise e 
    
    raise RuntimeError(f"Failed after attempts. Last error: {last_error}")

# -----------------------------
# File I/O
# -----------------------------
def load_output(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {}, "rows": []}

def save_atomic(path: str, data: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# -----------------------------
# Main Execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Local LLaMA Scorer")
    
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--prompt-file", required=True, help="System Prompt txt file")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--rubric-keys", default="cohesion,syntax,vocabulary,phraseology,grammar,conventions")
    
    # Defaults
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=None)

    args = parser.parse_args()
    config = ScorerConfig(args)

    print(f"--- Local LLaMA Scorer ---")
    
    # ---------------------------------------------------------
    # GPU CHECK
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU DETECTED: {gpu_name}")
        # Force on GPU 0 (integer 0) to avoid accelerate offloading
        device_arg = 0
        # Use bfloat16 for LLaMA 3.1 on Ampere+ GPUs (usually faster/better), otherwise float16
        dtype_arg = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        print("⚠️  WARNING: NO GPU DETECTED!")
        print("    Running on CPU. This will be very slow.")
        device_arg = -1
        dtype_arg = torch.float32

    print(f"Model: {config.model_id}")
    print("Loading tokenizer and model...")

    try:
        # 1. Initialize Tokenizer Explicitly to handle Padding
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if tokenizer.pad_token_id is None:
            # LLaMA models usually have no pad token by default
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 2. Initialize Pipeline with Explicit Tokenizer
        pipe = transformers.pipeline(
            "text-generation",
            model=config.model_id,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": dtype_arg}, 
            device=device_arg,
        )
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR loading model: {e}")
        print("Ensure 'accelerate' is installed and you are logged in via 'huggingface-cli login'.")
        return

    print("✅ Model loaded successfully.")
    
    # Load Data
    try:
        df = pd.read_csv(config.input_csv)
        if "full_text" not in df.columns:
            raise ValueError("Input CSV must contain a 'full_text' column.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if config.max_rows:
        df = df.head(config.max_rows)
    
    existing = load_output(config.output_json)
    existing_rows = {str(r["id"]): r for r in existing.get("rows", []) if "id" in r}
    
    processed = 0
    
    for i, row in tqdm(enumerate(df.itertuples(index=False)), total=len(df), desc="Scoring"):
        row_id = getattr(row, "id", f"row_{i}")
        rid = str(row_id)
        
        if rid in existing_rows and existing_rows[rid].get("LLM_status") == "scored":
            continue
            
        res = { "id": row_id, "full_text": getattr(row, "full_text", "") }
        
        for k, v in row._asdict().items():
            if k.lower() in config.rubric_keys:
                res[k.lower()] = v

        res.update({
            "LLM_status": "pending",
            "LLM_scores": {},
            "LLM_total": None,
            "LLM_request_id": None,
            "LLM_error_message": None
        })

        essay_str = "" if pd.isna(row.full_text) else str(row.full_text).strip()

        if not essay_str:
            res["LLM_status"] = "empty_essay"
            _log_all(config, {"event": "empty_essay", "row_id": rid})
        else:
            try:
                scores, req_id = score_row(pipe, config, row_id, essay_str)
                res["LLM_scores"] = scores
                if scores:
                    res["LLM_total"] = round(sum(scores.values()) / len(scores), 2)
                res["LLM_request_id"] = req_id
                res["LLM_status"] = "scored"
            except Exception as e:
                res["LLM_status"] = "failed"
                res["LLM_error_message"] = str(e)

        existing_rows[rid] = res
        processed += 1
        
        if processed % config.save_every_n == 0:
            payload = {"meta": vars(args), "rows": list(existing_rows.values())}
            save_atomic(config.output_json, payload)

    payload = {"meta": vars(args), "rows": list(existing_rows.values())}
    save_atomic(config.output_json, payload)
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()