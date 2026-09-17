import asyncio
import sys
import types

from schemas.requests import TranslateRequest, TranslateSegment


class DummyHTTPError(RuntimeError):
    def __init__(self, status_code, message="provider error"):
        self.status_code = status_code
        self.message = message
        super().__init__(f"{status_code} {message}")


class DummyGoogleTranslator:
    def __init__(self, *args, **kwargs):
        pass

    def translate(self, text):
        raise DummyHTTPError(429, "Too Many Requests")


class DummyMyMemoryTranslator:
    def __init__(self, *args, **kwargs):
        pass

    def translate(self, text):
        return f"mymemory:{text}"


class DummyLLMClient:
    calls = 0

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                DummyLLMClient.calls += 1
                raise DummyHTTPError(500, "router overloaded")


async def _passthrough_cinematic(rows, req, src_lang, loop, **kwargs):
    return rows


def _setup(monkeypatch, llm_handle):
    from api.routers import dub_translate

    fake_deep_translator = types.ModuleType("deep_translator")
    fake_deep_translator.GoogleTranslator = DummyGoogleTranslator
    fake_deep_translator.MyMemoryTranslator = DummyMyMemoryTranslator
    monkeypatch.setitem(sys.modules, "deep_translator", fake_deep_translator)
    monkeypatch.setattr(dub_translate, "_maybe_cinematic", _passthrough_cinematic)
    monkeypatch.setattr(dub_translate.time, "sleep", lambda s: None)
    monkeypatch.setattr("services.llm_skills.resolve_skill_client", lambda skill_id: llm_handle)
    DummyLLMClient.calls = 0
    return dub_translate


def _req():
    return TranslateRequest(
        segments=[
            TranslateSegment(id="2", text="hello", target_lang="ne"),
            TranslateSegment(id="1", text="world", target_lang="ne"),
        ],
        target_lang="ne",
        provider="google",
    )


def _llm():
    return types.SimpleNamespace(
        client=DummyLLMClient(), model="meta-llama/Llama-3.3-70B-Instruct", timeout=30,
    )


def test_opt_in_fallback_uses_llm_then_mymemory(monkeypatch):
    dub_translate = _setup(monkeypatch, _llm())
    monkeypatch.setenv("OMNIVOICE_TRANSLATE_FALLBACK", "1")

    result = asyncio.run(dub_translate.dub_translate(_req()))

    assert [r["id"] for r in result] == ["1", "2"]
    assert result[0]["text"] == "mymemory:world"
    assert result[1]["text"] == "mymemory:hello"
    assert all("error" not in r for r in result)
    # One LLM attempt per failed segment, never duplicated.
    assert DummyLLMClient.calls == 2


def test_fallback_is_off_by_default(monkeypatch):
    """Without the explicit opt-in no text reaches a provider the user did not pick."""
    dub_translate = _setup(monkeypatch, _llm())
    monkeypatch.delenv("OMNIVOICE_TRANSLATE_FALLBACK", raising=False)

    result = asyncio.run(dub_translate.dub_translate(_req()))

    assert DummyLLMClient.calls == 0
    assert all("429" in r["error"] for r in result)
    assert [r["text"] for r in result] == ["world", "hello"]


def test_fallback_without_llm_client_keeps_provider_error(monkeypatch):
    """A skill handle with no client must not surface AttributeError."""
    handle = types.SimpleNamespace(client=None, model="m", timeout=30)
    dub_translate = _setup(monkeypatch, handle)
    monkeypatch.setenv("OMNIVOICE_TRANSLATE_FALLBACK", "1")
    monkeypatch.setattr(DummyMyMemoryTranslator, "translate", lambda self, t: "")

    result = asyncio.run(dub_translate.dub_translate(_req()))

    assert all("429" in r["error"] for r in result)
    assert not any("AttributeError" in r["error"] for r in result)
