
from flask import Flask, request, jsonify,render_template, jsonify
import os
import os
import sys
from langchain.llms.bedrock import Bedrock
from pathlib import Path
import os
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


def read_pdf(payslip_path):
    pdf_reader = PdfReader(payslip_path)
    total_pages = len(pdf_reader.pages)

    raw_text = ''
    with tqdm(total=total_pages, desc="Processing PDF") as pbar:
        for i, page in enumerate(pdf_reader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
    return raw_text

def llm_caller(query):
    result = llm(query)
    insights = (result.strip())
    print(f"hello")
    if insights:
        print("has data")
    return insights

@app.route('/dashboard')
def dashboard():
    return render_template('template.html')

@app.route('/')
def index():
    return render_template('template.html')

@app.route('/option1')
def option1():
    # Add your code here
    return render_template('index-summary-css.html')

@app.route('/option2')
def option2():
    # Add your code here
    return render_template('index-mcq-css.html')

@app.route('/option3')
def option3():
    return render_template('index-payslip-css.html')


@app.route('/option4')
def option4():
    # Add your code here
    return render_template('index-api.html')

@app.route('/upload', methods=['POST'])
def upload_file():

    uploaded_file = request.files['pdf']
    raw_text = read_pdf(uploaded_file)
    query = f"""

    Human: Given is payslip information for employee please read it and analyse the contents.
    if a name of employee is mentioned return it, otherwise return nothing.
    if the net salary of employee is mentioned return it, otherwise return nothing.
    if the designation of employee is mentioned return it, otherwise return nothing.
    if the Income Tax of employee is mentioned return it, otherwise return nothing.
    if the Basic Amount of employee is mentioned return it, otherwise return nothing.
    if the Basic Amount YTD of employee is mentioned return it, otherwise return nothing.
    if the month for which the payslip is mentioned return it, otherwise return nothing.
    if the Comp Off amount of employee is mentioned return it, otherwise return nothing.
    if the Comp Off YTD of employee is mentioned return it, otherwise return nothing.

    paylip data: ```
    {raw_text}
    ```

    Assistant:"""

    payslip_information = llm_caller(query)
            
    return render_template('index-payslip-css.html', output_data=payslip_information)

@app.route('/upload-mcq', methods=['POST'])
def upload_file_mcq():

    uploaded_file = request.files['pdf']
    raw_text = read_pdf(uploaded_file)
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

    payslip_information = llm_caller(query)
            
    return render_template('index-mcq-css.html', output_data=payslip_information)

@app.route('/upload-api', methods=['POST'])
def upload_file_api():

    file = request.files['file']
    
    file_data= (file.read())
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
    {file_data}
    ```

    Assistant:"""

    payslip_information = llm_caller(query)
            
    return render_template('index-api.html', output_data=payslip_information)

@app.route('/upload-summary', methods=['POST'])
def upload_file_summary():

    uploaded_file = request.files['pdf']
    raw_text = read_pdf(uploaded_file)
    raw_query = request.form['text']
    
    query = f"""

    Human: Given the loan report document
    Give answer for: {raw_query} 
    
    report: ```
    {raw_text}
    
    Assistant:"""

    payslip_information = llm_caller(query)
    
    return render_template('index-summary-css.html', output_data=payslip_information, query=raw_query )





if __name__ == '__main__':
    app.run(debug=True)
