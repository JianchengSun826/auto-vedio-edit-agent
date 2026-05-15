# tests/conftest.py
import os
import sys
import pytest
from unittest.mock import MagicMock

# Set at module level so it's available during collection (before fixtures run)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

# Mock whisperx so tests can import transcriber.py without the real package
if 'whisperx' not in sys.modules:
    sys.modules['whisperx'] = MagicMock()

# Mock gradio so tests can import app.main without a working gradio installation
if 'gradio' not in sys.modules:
    sys.modules['gradio'] = MagicMock()


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Ensure required env vars are set for all tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
