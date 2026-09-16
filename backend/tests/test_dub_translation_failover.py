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
        self.calls = 0

    def translate(self, text):
        self.calls += 1
        raise DummyHTTPError(429, "Too Many Requests")


class DummyMyMemoryTranslator:
    def __init__(self, *args, **kwargs):
        pass

    def translate(self, text):
        return f"mymemory:{text}"


class DummyLLMClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                raise DummyHTTPError(500, "router overloaded")


def test_google_rate_limits_fall_back_to_llm_then_mymemory(monkeypatch):
    from api.routers import dub_translate

    fake_deep_translator = types.ModuleType("deep_translator")
    fake_deep_translator.GoogleTranslator = DummyGoogleTranslator
    fake_deep_translator.MyMemoryTranslator = DummyMyMemoryTranslator
    monkeypatch.setitem(sys.modules, "deep_translator", fake_deep_translator)

    fake_llm = types.SimpleNamespace(
        client=DummyLLMClient(),
        model="meta-llama/Llama-3.3-70B-Instruct",
        timeout=30,
    )
    monkeypatch.setattr(dub_translate, "_maybe_cinematic", lambda rows, req, src_lang, loop, **kwargs: rows)
    monkeypatch.setattr("services.llm_skills.resolve_skill_client", lambda skill_id: fake_llm)

    req = TranslateRequest(
        segments=[
            TranslateSegment(id="2", text="hello", target_lang="ne"),
            TranslateSegment(id="1", text="world", target_lang="ne"),
        ],
        target_lang="ne",
        provider="google",
    )

    result = asyncio.run(dub_translate.dub_translate(req))

    assert [r["id"] for r in result] == ["1", "2"]
    assert result[0]["text"] == "mymemory:world"
    assert result[1]["text"] == "mymemory:hello"
    assert all("error" not in r for r in result)
