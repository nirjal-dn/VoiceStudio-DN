"""/gallery/download must never let video_url reach yt-dlp as an option.

yt-dlp parses options interspersed with URLs, so an unchecked value such as
``--config-locations=...`` or ``--exec=...`` would be an argument injection.
"""
import asyncio

import pytest
from fastapi import HTTPException

from api.routers import gallery


class _FailedProc:
    returncode = 1

    async def communicate(self):
        return b"", b"stub"


@pytest.fixture
def spawned(monkeypatch):
    calls = []

    async def fake_spawn(*argv, **kwargs):
        calls.append(list(argv))
        return _FailedProc()

    monkeypatch.setattr(gallery, "spawn_subprocess", fake_spawn)
    return calls


def _download(url):
    return asyncio.run(gallery.download_youtube_clip(
        video_url=url, start_time=0, duration=10,
        character_name="x", category="import", description="",
    ))


@pytest.mark.parametrize("url", [
    "--config-locations=/tmp/evil.conf",
    "--exec=touch /tmp/pwned",
    "file:///etc/passwd",
    "ytsearch1:anything",
    "https://",
    "",
])
def test_non_http_urls_are_rejected_before_spawning(spawned, url):
    with pytest.raises(HTTPException) as exc:
        _download(url)
    assert exc.value.status_code == 400
    assert spawned == []


def test_url_is_passed_after_end_of_options_marker(spawned):
    with pytest.raises(HTTPException):  # the stub process "fails"
        _download(" https://www.youtube.com/watch?v=abc ")
    argv = spawned[0]
    assert argv[-2:] == ["--", "https://www.youtube.com/watch?v=abc"]
