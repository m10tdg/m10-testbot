#!/bin/bash
# infra/init-localstack/ready.sh
# LocalStack runs scripts in /etc/localstack/init/ready.d/ automatically after startup.

set -e

echo "[localstack-init] Creating S3 buckets..."

awslocal s3 mb s3://m10-documents  --region eu-central-1 2>/dev/null || echo "m10-documents already exists"
awslocal s3 mb s3://m10-artifacts  --region eu-central-1 2>/dev/null || echo "m10-artifacts already exists"

echo "[localstack-init] Buckets ready:"
awslocal s3 ls