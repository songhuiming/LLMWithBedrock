"""
Invoke claude 3.7 sonnet V1 from bedrock to make inference based on the prompt.
"""

import re, json, boto3, time

model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0" 

aws_session = boto3.Session(profile_name="xxx", region_name="us-west-2")
bedrock_runtime = aws_session.client("bedrock-runtime", region_name="us-west-2")


system_prompt="""
Human: can you write the python code for two sum?
"""

def build_payload(prompt, max_tokens=6000, temperature=1.0, budget_tokens=2000):
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

def inference(model_id, payload):
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json"
    )
    return response

payload = build_payload(system_prompt)
response = inference(model_id, payload)

response_body = json.loads(response["body"].read().decode("utf-8"))

response_text = response_body['content'][-1]['text']
thinking = response_body['content'][0]['thinking']

print(f"The output is: \n {response_text} \n\n")
print(f"The reasoning process is: \n {thinking}")
