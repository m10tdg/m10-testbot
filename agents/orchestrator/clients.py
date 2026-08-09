import os
from dotenv import load_dotenv
load_dotenv()

import boto3
import openai
import psycopg2
from qdrant_client import QdrantClient

pg_conn = psycopg2.connect(os.environ["DATABASE_URL"])
pg_conn.autocommit = True

openai.api_key = os.environ["OPENAI_API_KEY"]

qdrant = QdrantClient(url=os.environ["QDRANT_URL"])

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["AWS_REGION"],
)

ARTIFACTS_BUCKET = os.environ["S3_ARTIFACTS_BUCKET"]


def embed(text: str) -> list[float]:
    resp = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding
