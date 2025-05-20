
import re, json, boto3, time, logging, sys
import subprocess
from build_templates import PromptRenderer
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
from pathlib import Path
home_dir = Path.home()

import os

aws_session = boto3.Session(profile_name="xxx", region_name="us-west-2")
bedrock_runtime = aws_session.client("bedrock-runtime", region_name="us-west-2")


model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

prompt_func = PromptRenderer("pv_cls_prompt_template_claude35v1.02.txt")


infile =  "/data/test/mmlu_scale_20250403.jsonl"

sync_data = []
with open(infile, 'r') as f:
    for line in f:
        sync_data.append(json.loads(line.strip()))


def invoke_with_retry(bedrock_runtime, request, max_retries=5, initial_delay=1):
    for attempt in range(max_retries):
        try:
            response = bedrock_runtime.invoke_model(**request)
            return response
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                if attempt == max_retries - 1:
                    raise
                delay = initial_delay * (2 ** attempt)  
                time.sleep(delay)
            else:
                raise
    else:
        raise RuntimeError("Failed to invoke model after multiple attempts.")


def build_payload_claude37(prompt, max_tokens=6000, temperature=1.0, budget_tokens=2000):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["computer-use-2025-01-24"],
        "max_tokens": max_tokens,
        "thinking": {
            "type": "enabled",
            "budget_tokens": budget_tokens
        },
        "temperature": temperature,
        "messages": [
            {"role": "user",
            "content": [{ "type": "text", "text": prompt}]
            }
        ],
    }
    return payload

results = []
final_results = []
starttime = time.time()


for i, item in enumerate(sync_data):
    dialogue = item['metadata']['dialogue']
    true_cat = item['answer']
    true_label = 'no' if true_cat=='Non-PV' else 'yes'
    prompt = prompt_func(dialogue=dialogue, question="Based on the multi-turn dialogue above, analyze whether the final response violates any of the provided policies.")

    payload = build_payload_claude37(prompt)

    request = {
        "modelId": model_id,
        "body": json.dumps(payload),
        "contentType": "application/json"
    }

    try:
        response = invoke_with_retry(bedrock_runtime, request)

        response_body = json.loads(response["body"].read().decode("utf-8"))
        response_text = response_body['content'][-1]['text']
        response_json = json.loads(response_text.replace('```json\n', '').replace('\n```', '').strip())


        results.append({
            "dialogue": dialogue,
            "true_cat": true_cat,
            "true_label": true_label,
            "response_json": response_json,
            "response_body": response_body,
            "pred_cat": response_json['pv_category'],
            'pred_label': response_json['pv_pred'],
            "prompt": prompt
        })

    except ClientError as e:
        print(f"ClientError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


