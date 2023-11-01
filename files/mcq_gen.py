from flask import Flask, request, jsonify,render_template, jsonify
import os
import os
from langchain.llms.bedrock import Bedrock
from typing import Optional
import boto3
from botocore.config import Config
from PyPDF2 import PdfReader
from tqdm import tqdm

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


def process_file_content(payslip_path):

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
 

    # payslip_path = Path(".") / "payslip" / 'Payslip_124_Aug_2023.pdf'
    # payslip_path = Path(".") / "payslip" / "Samples B" / 'B (1).pdf'
    # payslip_path = Path(".") / "payslip" / "Samples A" / 'A (1).pdf'
    pdf_reader = PdfReader(payslip_path)
    total_pages = len(pdf_reader.pages)

    raw_text = ''
    with tqdm(total=total_pages, desc="Processing PDF") as pbar:
        for i, page in enumerate(pdf_reader.pages):
            content = page.extract_text()
            if content:
                raw_text += content

    query = f"""

    Human: Given the data for a well know book.please read it and analyse the contents.
    Generate 3 possible MCQ (Multiple Choice Questions) from the given book data

    Next Generate a answer key with below format for above generated questions
    Question:
    Correct Answer:

    Book data: ```
    {raw_text}
    ```

    Assistant:"""

    result = llm(query)
    payslip_information = (result.strip())
    print(f"hello")
    if payslip_information:
        print("has data")
    return payslip_information

@app.route('/')
def index():
    return render_template('index-mcq-css.html')

@app.route('/upload', methods=['POST'])
def upload_file():

    uploaded_file = request.files['pdf']
    payslip_information = process_file_content(uploaded_file)
            
    return render_template('index-mcq-css.html', output_data=payslip_information)
    
    

if __name__ == '__main__':
    app.run(debug=True, port=5002)
