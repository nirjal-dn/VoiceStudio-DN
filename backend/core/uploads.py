"""Persist multipart uploads to disk in bounded chunks.

``await UploadFile.read()`` with no size materializes the whole upload (a
multi-GB video, a long recording) in process memory before writing it back
out. Starlette already spools the request body to a temp file, so copying it
in 1 MB reads keeps peak memory at one chunk regardless of upload size.
"""
from __future__ import annotations

import logging

from core.file_cleanup import FileCleanupError, unlink_if_present

logger = logging.getLogger("omnivoice.uploads")

UPLOAD_CHUNK_BYTES = 1024 * 1024


async def copy_upload(upload, destination) -> int:
    """Stream ``upload`` into the open binary file ``destination``; returns bytes copied."""
    total = 0
    while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
        destination.write(chunk)
        total += len(chunk)
    return total


async def save_upload(upload, path: str) -> int:
    """Stream ``upload`` to ``path``; a partial file is removed on failure."""
    try:
        with open(path, "wb") as output:
            return await copy_upload(upload, output)
    except BaseException:
        try:
            unlink_if_present(path)
        except FileCleanupError:
            logger.warning("Could not remove incomplete upload %s", path, exc_info=True)
        raise
