# tests/conftest.py
import pytest

@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Ensure required env vars are set for all tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
