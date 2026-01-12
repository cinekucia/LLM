import os
import json
import re
import time
import argparse
import logging
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

# -----------------------------
# Configuration Class
# -----------------------------
class ScorerConfig:
    def __init__(self, args):
        self.input_csv = args.input
        self.model = args.model
        self.region = args.region
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.requests_per_min = args.rpm
        self.save_every_n = args.save_every
        self.max_rows = args.max_rows
        
        # 1. Parse Rubric Keys from CLI
        self.rubric_keys = [k.strip() for k in args.rubric_keys.split(",")]

        # 2. Dynamic Filename Generation
        input_stem = os.path.splitext(os.path.basename(self.input_csv))[0]
        prompt_stem = os.path.splitext(os.path.basename(args.prompt_file))[0]
        # Clean model name (e.g., meta.llama3-1-70b -> meta-llama3-1-70b)
        model_stem = self.model.replace(":", "-").replace(".", "-")
        
        filename = f"{input_stem}_{model_stem}_{prompt_stem}.json"
        
        os.makedirs(args.output_dir, exist_ok=True)
        self.output_json = os.path.join(args.output_dir, filename)

        # 3. Load Base System Prompt
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            self.base_system_prompt = f.read().strip()

        # 4. Construct Strict JSON Schema
        self.json_schema_str = json.dumps({
            "type": "object",
            "properties": {k: {"type": "number", "minimum": 1.0, "maximum": 5.0} for k in self.rubric_keys},
            "required": self.rubric_keys,
            "additionalProperties": False
        }, indent=2)

        # 5. Detailed Logs
        self.log_dir = args.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        log_base = f"{input_stem}_{model_stem}_{prompt_stem}"
        
        self.log_all_path = os.path.join(self.log_dir, f"{log_base}_conversation_logs.jsonl")
        self.log_error_path = os.path.join(self.log_dir, f"{log_base}_errors.jsonl")

# -----------------------------
# Logging Helpers
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
# Rate Limiting
# -----------------------------
_last_request_time = 0.0

def _throttle(rpm: float):
    global _last_request_time
    min_seconds = 60.0 / rpm
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < min_seconds:
        time.sleep(min_seconds - elapsed)
    _last_request_time = time.time()

# -----------------------------
# Parsing
# -----------------------------
def _extract_last_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try: return json.loads(text)
    except json.JSONDecodeError: pass

    # Find JSON block in markdown
    m = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try: return json.loads(m[-1])
        except: pass

    # Find last brace pair
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
        if abs((val * 2) - round(val * 2)) > 1e-6:
            errors.append(f"Key '{k}' not in 0.5 increments: {val}")
        out[k] = val
    if errors:
        raise ValueError("Validation failed: " + "; ".join(errors))
    return out

# -----------------------------
# Llama 3 Prompt Formatter
# -----------------------------
def format_llama3_prompt(system_text: str, user_text: str) -> str:
    """
    Formats the prompt using Llama 3 special tokens.
    """
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_text}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_text}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

# -----------------------------
# Core Scoring (Bedrock)
# -----------------------------
def score_row(client, config: ScorerConfig, row_id: Any, essay_text: str) -> Tuple[Dict[str, float], str]:
    # 1. Truncate
    MAX_CHARS = 100_000 
    if len(essay_text) > MAX_CHARS:
        essay_text = essay_text[:MAX_CHARS] + "\n[TRUNCATED]"

    # 2. Prepare Contents (Schema Injection)
    system_content = (
        f"{config.base_system_prompt}\n\n"
        f"STRICT OUTPUT RULES:\n"
        f"1. You must output valid JSON only.\n"
        f"2. Follow this JSON schema exactly:\n"
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

    # 3. Format for Llama 3
    final_prompt = format_llama3_prompt(system_content, user_content)

    # 4. Bedrock Request Body
    native_request = {
        "prompt": final_prompt,
        "max_gen_len": 1024,  # Enough for the JSON
        "temperature": config.temperature,
        "top_p": config.top_p,
    }

    last_error = None
    
    for attempt in range(1, 4):
        _throttle(config.requests_per_min)
        try:
            _log_all(config, {
                "event": "request", "row_id": row_id, "attempt": attempt, 
                "request": native_request
            })
            
            # --- BEDROCK CALL ---
            response = client.invoke_model(
                modelId=config.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(native_request),
            )
            
            # Parse Bedrock Response
            model_response = json.loads(response["body"].read())
            raw_text = (model_response.get("generation") or "").strip()
            
            req_id = response.get("ResponseMetadata", {}).get("RequestId")

            _log_all(config, {
                "event": "response", "row_id": row_id, "attempt": attempt, 
                "response": {"raw_text": raw_text}
            })
            
            # Extract & Validate
            json_obj = _extract_last_json_object(raw_text)
            scores = _validate_scores(json_obj, config.rubric_keys)
            
            return scores, req_id

        except ClientError as e:
            last_error = e
            error_code = e.response['Error']['Code']
            error_msg = str(e)
            
            # Throttling handling
            if error_code in ["ThrottlingException", "Throttling"]:
                _log_error(config, {"event": "rate_limit", "row_id": row_id, "error": error_msg})
                time.sleep(10)
            else:
                _log_error(config, {"event": "error", "row_id": row_id, "attempt": attempt, "error": error_msg})
                time.sleep(2 * attempt)
        except Exception as e:
            last_error = e
            _log_error(config, {"event": "error", "row_id": row_id, "attempt": attempt, "error": str(e)})
            time.sleep(2 * attempt)
    
    raise RuntimeError(f"Failed after 3 attempts. Last error: {last_error}")

# -----------------------------
# File I/O & Main
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

def main():
    parser = argparse.ArgumentParser(description="Bedrock Llama 3 Scorer")
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--rubric-keys", default="cohesion,syntax,vocabulary,phraseology,grammar,conventions")
    
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--log-dir", default="logs")
    
    # Bedrock specific args
    parser.add_argument("--model", default="meta.llama3-1-70b-instruct-v1:0")
    parser.add_argument("--region", default="us-west-2", help="AWS Region")
    
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--rpm", type=float, default=60)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=None)

    args = parser.parse_args()

    config = ScorerConfig(args)
    
    # Initialize Bedrock Client
    try:
        client = boto3.client("bedrock-runtime", region_name=config.region)
    except Exception as e:
        print(f"Error initializing Bedrock client: {e}")
        return

    print(f"--- Bedrock Scorer ---")
    print(f"Model: {config.model} ({config.region})")
    print(f"File:  {config.output_json}")

    try:
        df = pd.read_csv(config.input_csv)
        if "full_text" not in df.columns:
            raise ValueError("Input CSV must contain a 'full_text' column.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    col_map = {c.lower(): c for c in df.columns}

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
            
        row_dict = row._asdict()
        
        res = {
            "id": row_id,
            "full_text": row.full_text
        }

        for key in config.rubric_keys:
            if key in col_map:
                res[key] = row_dict.get(col_map[key])
        if "total" in col_map:
            res["total"] = row_dict.get(col_map["total"])

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
                scores, req_id = score_row(client, config, row_id, essay_str)
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
    print("\nDone.")

if __name__ == "__main__":
    main()