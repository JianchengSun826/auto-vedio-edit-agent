from storage.base import StorageBackend
from storage.local import LocalStorage
from config.settings import settings


def get_storage_backend() -> StorageBackend:
    if settings.storage_backend == "local":
        return LocalStorage(root=settings.local_storage_root)
    raise ValueError(
        f"Unknown storage backend: {settings.storage_backend}. "
        "Only 'local' is supported in v1."
    )
