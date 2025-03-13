"""
Invoke Llama 3.1 from bedrock to make inference based on the prompt.
"""

import re, json, boto3, time
from botocore.exceptions import ClientError

model_id = "meta.llama3-1-70b-instruct-v1:0" 


aws_session = boto3.Session(profile_name="xxx", region_name="us-west-2")
bedrock_runtime = aws_session.client("bedrock-runtime", region_name="us-west-2")


system_prompt="""
Human: can you write the python code for two sum?
"""

def build_prompt(prompt):
    formatted_prompt = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

def build_payload(prompt, max_gen_len=6000, temperature=0.7):
    payload = {
        "prompt": prompt,
        "max_gen_len": max_gen_len,
        "temperature": temperature,
    }
    return payload

def inference(model_id, payload):
    response = bedrock_runtime.invoke_model(
        modelId = model_id,
        body=json.dumps(payload),
    )
    return response
 
prompt = build_prompt(system_prompt)
payload = build_payload(prompt)
response = inference(model_id, payload)

response_body = json.loads(response["body"].read().decode("utf-8"))
response_text = response_body["generation"]

print(f"The output is: \n {response_text} \n\n")
