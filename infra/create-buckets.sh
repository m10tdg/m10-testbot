#!/bin/bash
# Creates the S3 buckets TESTBOT needs, inside Localstack's fake S3.
# Run this ONCE, after `docker compose up -d`, before uploading any documents.
#
# Usage:  bash create-buckets.sh

set -e

echo "Creating m10-documents bucket..."
curl -s -X PUT http://localhost:4566/m10-documents
echo ""

echo "Creating m10-artifacts bucket..."
curl -s -X PUT http://localhost:4566/m10-artifacts
echo ""

echo "Done. Verify with:  curl http://localhost:4566/"
