"""
nova pro model form amaozn
"""

import re, json, boto3, time
from botocore.exceptions import ClientError

model_id = "us.amazon.nova-pro-v1:0"  

aws_session = boto3.Session(profile_name="xxx", region_name="us-west-2")
bedrock_runtime = aws_session.client("bedrock-runtime", region_name="us-west-2")


def build_payload(prompt, max_new_tokens=2000, temperature=1.0):
    inference_config = {
        "max_new_tokens": max_new_tokens, 
        "temperature": temperature
    } 
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
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json"
    )
    return response


payload = build_payload(system_prompt)
response = inference(model_id, payload)

response_body = json.loads(response["body"].read())
response_text = response_body['output']['message']['content'][-1]['text']

print(f"The output is: \n {response_text} \n\n")
