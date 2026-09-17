"""Uploads are persisted in bounded chunks: peak Python memory stays about one
chunk however large the file (a dub video, a long recording)."""
import asyncio
import tempfile
import tracemalloc

import pytest
from starlette.datastructures import UploadFile

from core import uploads

SIZE = 64 * 1024 * 1024


def _upload(tmp_path):
    spool = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, dir=tmp_path)
    block = bytes(range(256)) * 4096  # 1 MiB, non-zero pattern
    for _ in range(SIZE // len(block)):
        spool.write(block)
    spool.seek(0)
    return UploadFile(file=spool, filename="talk.webm")


def _peak(coro):
    tracemalloc.start()
    try:
        asyncio.run(coro)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_save_upload_copies_everything_with_bounded_memory(tmp_path):
    dest = tmp_path / "original.webm"
    peak = _peak(uploads.save_upload(_upload(tmp_path), str(dest)))
    assert dest.stat().st_size == SIZE
    with open(dest, "rb") as fh:
        assert fh.read(256) == bytes(range(256))
    assert peak < 8 * 1024 * 1024, f"peak {peak / 1e6:.1f} MB for a {SIZE / 1e6:.0f} MB upload"


def test_full_read_baseline_materializes_the_upload(tmp_path):
    """The pattern this replaced: documents the cost the helper removes."""
    async def old(upload):
        with open(tmp_path / "old.webm", "wb") as fh:
            fh.write(await upload.read())

    assert _peak(old(_upload(tmp_path))) >= SIZE


def test_failed_save_removes_the_partial_file(tmp_path):
    class Broken:
        calls = 0

        async def read(self, size):
            Broken.calls += 1
            if Broken.calls > 2:
                raise ConnectionResetError("client went away")
            return b"x" * size

    dest = tmp_path / "partial.bin"
    with pytest.raises(ConnectionResetError):
        asyncio.run(uploads.save_upload(Broken(), str(dest)))
    assert not dest.exists()
