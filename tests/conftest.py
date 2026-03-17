# tests/conftest.py
import os
import pytest

# Set at module level so it's available during collection (before fixtures run)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Ensure required env vars are set for all tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
