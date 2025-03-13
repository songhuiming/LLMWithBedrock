"""
Deepseek R1 generation with prompt, output the response text and the reasoning
"""

import re, json, time, boto3
from botocore.exceptions import ClientError

model_id = "us.deepseek.r1-v1:0"

aws_session = boto3.Session(profile_name="xxx", region_name="us-west-2")
bedrock_runtime = aws_session.client("bedrock-runtime", region_name="us-west-2")

system_prompt="""
Human: can you write the python code for two sum?
"""

def build_payload(prompt, max_tokens=2000, temperature=0.7):
    inference_config = {"maxTokens": max_tokens, "temperature": temperature} 
    payload = {
        "inferenceConfig": inference_config,
        "messages": [
            {"role": "user",
            "content": [{"text": prompt}]
            }
        ],
    }
    return payload

def inference(model_id, payload):
    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=payload['messages'],
        inferenceConfig=payload['inferenceConfig'],
    )
    return response

payload = build_payload(system_prompt)
response = inference(model_id, payload)

response_body = response
response_text = response_body['output']['message']['content'][0]['text']
reasoning = response_body['output']['message']['content'][1]['reasoningContent']['reasoningText']['text']

print(f"The output is: \n {response_text} \n\n")
print(f"The reason is: \n {reasoning} \n")
