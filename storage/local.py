import shutil
from pathlib import Path
from storage.base import StorageBackend


class LocalStorage(StorageBackend):
    def __init__(self, root: str = "./data"):
        self.root = Path(root)

    def read(self, path: str) -> Path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p

    def write(self, local_path: Path, dest: str) -> None:
        dest_path = self.root / dest
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest_path)

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix if prefix else self.root
        if not base.exists():
            return []
        return [str(p) for p in base.rglob("*") if p.is_file()]
