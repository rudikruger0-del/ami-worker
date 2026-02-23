import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MODULE_NAME = "services.supabase_client"


def _reload_supabase_client(monkeypatch, *, url, service_role_key, mock_create_client):
    monkeypatch.setenv("SUPABASE_URL", url)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", service_role_key)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)

    monkeypatch.setattr("supabase.create_client", mock_create_client)

    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_supabase_client_uses_service_role_key(monkeypatch):
    mock_create_client = MagicMock(return_value="client")

    module = _reload_supabase_client(
        monkeypatch,
        url="https://example.supabase.co",
        service_role_key="service-role-key",
        mock_create_client=mock_create_client,
    )

    assert module.supabase == "client"
    mock_create_client.assert_called_once_with(
        "https://example.supabase.co",
        "service-role-key",
    )


def test_supabase_client_disabled_without_service_role_key(monkeypatch):
    mock_create_client = MagicMock(return_value="client")

    module = _reload_supabase_client(
        monkeypatch,
        url="https://example.supabase.co",
        service_role_key="",
        mock_create_client=mock_create_client,
    )

    assert module.supabase is None
    mock_create_client.assert_not_called()
