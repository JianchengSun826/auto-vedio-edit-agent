import pytest
from pathlib import Path
from storage.local import LocalStorage
from storage.factory import get_storage_backend


def test_local_storage_read_existing_file(tmp_path):
    test_file = tmp_path / "video.mp4"
    test_file.write_bytes(b"fake video data")
    storage = LocalStorage(root=str(tmp_path))
    result = storage.read(str(test_file))
    assert result == test_file
    assert result.exists()


def test_local_storage_write(tmp_path):
    storage = LocalStorage(root=str(tmp_path))
    src = tmp_path / "source.mp4"
    src.write_bytes(b"data")
    storage.write(src, "output/result.mp4")
    assert (tmp_path / "output" / "result.mp4").exists()


def test_local_storage_list(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    storage = LocalStorage(root=str(tmp_path))
    results = storage.list("")
    assert len(results) == 3


def test_local_storage_read_missing_file(tmp_path):
    storage = LocalStorage(root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        storage.read("/nonexistent/path/video.mp4")


def test_factory_returns_local(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    backend = get_storage_backend()
    assert isinstance(backend, LocalStorage)
