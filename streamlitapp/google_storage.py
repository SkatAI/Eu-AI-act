"""
"""

import os
import json
import base64
from google.oauth2 import service_account
from google.cloud import storage


class StorageWrap(object):
    @classmethod
    def get_credentials_from_base64(cls) -> service_account.Credentials:
        encoded_credentials = os.environ.get("GOOGLE_CLOUD_CREDENTIALS_BASE64")
        if encoded_credentials is None:
            raise ValueError("The GOOGLE_CLOUD_CREDENTIALS_BASE64 environment variable is not set.")
        credentials_json = base64.b64decode(encoded_credentials).decode("utf-8")
        return service_account.Credentials.from_service_account_info(json.loads(credentials_json))

    def __init__(self) -> None:
        self.client = storage.Client(credentials=StorageWrap.get_credentials_from_base64())
        assert self.client.project == "skatai", "Storage Client can't access project name"

    def set_bucket(self, bucket_name: str = "ragtime-ai-act") -> None:
        self.bucket_name = bucket_name
        self.bucket = self.client.get_bucket(self.bucket_name)

    def save_json(self, json_data, filename):
        blob = self.bucket.blob(filename)
        blob.upload_from_string(json_data, content_type="application/json")

    def file_info(self, filename):
        self.bucket = self.client.get_bucket(self.bucket_name)
        blob = self.bucket.blob(filename)
        if blob.exists():
            print(f"{blob.id} {blob.size}")
        else:
            print(f"no blob for {filename}")
        return blob

    def list_buckets(self):
        return [bucket.name for bucket in self.client.list_buckets()]

    def list_blobs(self):
        return [blob.name for blob in self.bucket.list_blobs()]


if __name__ == "__main__":
    sw = StorageWrap()
    buckets = sw.list_buckets()
    print(buckets)
    sw.set_bucket("ragtime-ai-act")
    blobs = sw.list_blobs()
    print(blobs)
