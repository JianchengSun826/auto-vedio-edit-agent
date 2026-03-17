from pathlib import Path
from storage.base import StorageBackend


class S3Storage(StorageBackend):
    """
    Stub — not implemented in v1.
    To implement: pip install boto3, configure AWS credentials,
    implement download-to-temp on read(), upload on write().
    Change STORAGE_BACKEND=s3 in settings to activate.
    """

    def __init__(self, bucket: str, region: str = "us-east-1"):
        raise NotImplementedError(
            "S3Storage is not implemented in v1. "
            "Set STORAGE_BACKEND=local in your .env file."
        )

    def read(self, path: str) -> Path:
        raise NotImplementedError

    def write(self, local_path: Path, dest: str) -> None:
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:
        raise NotImplementedError
