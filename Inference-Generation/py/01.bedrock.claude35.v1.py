"""
Invoke claude 3.5 sonnet from bedrock to make inference based on the prompt.
"""

import re, json, boto3, time

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

aws_session = boto3.Session(profile_name="xxx", region_name="us-west-2")
bedrock_runtime = aws_session.client("bedrock-runtime", region_name="us-west-2")

system_prompt="""
Human: can you write the python code for two sum?
"""

def build_payload(prompt, max_tokens=2000, temperature=.6):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
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

print(f"The output is: \n {response_text}")
