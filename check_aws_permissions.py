"""
check_aws_permissions.py
Quick check for S3 and Transcribe access, using the same credentials
already working for Bedrock in this project.
"""
import boto3
import os
from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv("AWS_REGION", "us-east-1")

print(f"Checking permissions in region: {REGION}\n")

# ── Check S3 ──────────────────────────────────────────────────────────
print("── S3 ──")
try:
    s3 = boto3.client('s3', region_name=REGION)
    response = s3.list_buckets()
    buckets = [b['Name'] for b in response.get('Buckets', [])]
    print(f"✅ S3 access works. Found {len(buckets)} bucket(s):")
    for b in buckets:
        print(f"   - {b}")
except Exception as e:
    print(f"❌ S3 access failed: {e}")

# ── Check Transcribe ──────────────────────────────────────────────────
print("\n── Transcribe ──")
try:
    transcribe = boto3.client('transcribe', region_name=REGION)
    response = transcribe.list_transcription_jobs(MaxResults=1)
    print(f"✅ Transcribe access works.")
except Exception as e:
    print(f"❌ Transcribe access failed: {e}")

print("\nDone.")
