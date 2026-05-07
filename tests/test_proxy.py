"""Tests for llm_api_logger.proxy — LLMAPIProxy and LLM_HOSTS."""

from llm_api_logger.proxy import LLMAPIProxy, LLM_HOSTS
from llm_api_logger.store import LogStore


def test_llm_hosts_non_empty():
    assert len(LLM_HOSTS) >= 5


def test_llm_hosts_known_providers():
    assert "api.openai.com"     in LLM_HOSTS
    assert "api.anthropic.com"  in LLM_HOSTS
    assert "api.mistral.ai"     in LLM_HOSTS


def test_proxy_instantiation():
    store = LogStore(":memory:", backend="sqlite")
    proxy = LLMAPIProxy(store=store, host="localhost", port=18080)
    assert proxy.host == "localhost"
    assert proxy.port == 18080


def test_proxy_default_store():
    proxy = LLMAPIProxy()
    assert isinstance(proxy.store, LogStore)


def test_proxy_start_stop():
    store = LogStore(":memory:", backend="sqlite")
    proxy = LLMAPIProxy(store=store, port=18081)
    proxy.start()
    assert proxy._server is not None
    proxy.stop()
    assert proxy._server is None


def test_proxy_context_manager():
    store = LogStore(":memory:", backend="sqlite")
    with LLMAPIProxy(store=store, port=18082) as proxy:
        assert proxy._server is not None
    assert proxy._server is None


def test_proxy_different_ports_independent():
    """Two proxy instances on different ports must not share state."""
    s1 = LogStore(":memory:", backend="sqlite")
    s2 = LogStore(":memory:", backend="sqlite")
    with LLMAPIProxy(s1, port=18083) as p1:
        with LLMAPIProxy(s2, port=18084) as p2:
            assert p1.port != p2.port
