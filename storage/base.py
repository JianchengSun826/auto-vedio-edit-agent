from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    @abstractmethod
    def read(self, path: str) -> Path:
        """Return a local Path to the file (downloads if remote)."""

    @abstractmethod
    def write(self, local_path: Path, dest: str) -> None:
        """Write local_path to dest (uploads if remote)."""

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """List files under prefix."""
