from flask import Flask, request, jsonify,render_template
import os
import os
import sys
from langchain.llms.bedrock import Bedrock
from pathlib import Path
import os
from typing import Optional
import boto3
from botocore.config import Config

app = Flask(__name__)

def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


def process_file_content(book_question_email):

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # E.g. "us-east-1"
    os.environ["AWS_PROFILE"] = "default"


    boto3_bedrock = get_bedrock_client(
        region=os.environ.get("AWS_DEFAULT_REGION", None),
    )

    llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=boto3_bedrock,
        model_kwargs={
            "max_tokens_to_sample": 400,
            "temperature": 0, # Using 0 to get reproducible results
            "stop_sequences": ["\n\nHuman:"]
        }
    )

    query = f"""

    Human: Given the AWS API Gateway logs of multiple apis 
    extract all relevant information related to the configure the same api like
    api Name
    api method
    api query parameters
    api json body data
    api url/ api endpoint
    authorization paramters like username and password or token



    Api Logs: ```
    {book_question_email}
    ```

    Assistant:"""

    result = llm(query)
    print(result.strip())
    return result.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    file_data= (file.read())
    api_configs = process_file_content(file_data)
    return render_template('index.html', output_data=api_configs)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
