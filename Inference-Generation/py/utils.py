from botocore.exceptions import ClientError

def invoke_with_retry(bedrock_runtime, request, max_retries=5, initial_delay=1):
    for attempt in range(max_retries):
        try:
            response = bedrock_runtime.invoke_model(**request)
            return response
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                if attempt == max_retries - 1:
                    logger.error("Maximum retry attempts reached. Raising exception.")
                    raise
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Rate limited. Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                logger.error(f"An unexpected error occurred: {e}")
                raise
    else:
        logger.error("Exceeded maximum retries without success.")
        raise RuntimeError("Failed to invoke model after multiple attempts.")

# response = invoke_with_retry(bedrock_runtime, request)
