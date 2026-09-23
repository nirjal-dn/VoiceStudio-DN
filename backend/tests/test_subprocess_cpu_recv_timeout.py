"""Regression: on a CPU-only host, a subprocess TTS synth must not be killed at
the tight accelerator recv deadline.

The generate recv watchdog re-arms only when a frame arrives. Sidecars heartbeat
during model *load* (#1367) but emit no frame during the *synthesis* forward
pass — fine on a GPU (synthesis is fast), fatal on a CPU-only host where a heavy
TTS forward pass runs for minutes: the tight deadline hard-killed a healthy
synth ("sidecar exceeded recv timeout; killing"). The hard-kill exists to
reclaim an *accelerator device*, which a CPU-only host doesn't have, so on CPU
the deadline is floored at the CPU generate budget instead.
"""
from __future__ import annotations

import services.subprocess_backend as sb


def _clear_cache():
    sb._host_has_accelerator.cache_clear()


def test_cpu_only_floors_recv_at_cpu_budget(monkeypatch):
    _clear_cache()
    monkeypatch.setattr(sb, "_host_has_accelerator", lambda: False)
    from services.model_manager import CPU_JOB_TIMEOUT_S

    # A tight per-engine deadline (e.g. the base 60s, or omnivoice's 300s) is
    # lifted to the CPU generate budget so a slow-but-healthy CPU synth survives.
    assert sb._effective_generate_recv_timeout(60.0) == CPU_JOB_TIMEOUT_S
    assert sb._effective_generate_recv_timeout(300.0) == CPU_JOB_TIMEOUT_S


def test_cpu_only_never_lowers_a_generous_engine_deadline(monkeypatch):
    _clear_cache()
    monkeypatch.setattr(sb, "_host_has_accelerator", lambda: False)
    from services.model_manager import CPU_JOB_TIMEOUT_S

    # An engine that already asks for longer than the budget keeps its own value.
    generous = CPU_JOB_TIMEOUT_S + 100.0
    assert sb._effective_generate_recv_timeout(generous) == generous


def test_accelerator_host_keeps_tight_deadline(monkeypatch):
    _clear_cache()
    monkeypatch.setattr(sb, "_host_has_accelerator", lambda: True)
    # A GPU/MPS host must keep the tight deadline — a wedge there really does
    # hold VRAM and must be reclaimed promptly.
    assert sb._effective_generate_recv_timeout(60.0) == 60.0
    assert sb._effective_generate_recv_timeout(300.0) == 300.0


def test_host_accelerator_probe_never_raises(monkeypatch):
    _clear_cache()

    class _Boom:
        @staticmethod
        def is_available():
            raise RuntimeError("driver exploded")

    monkeypatch.setattr(sb.torch, "cuda", _Boom)
    # A broken CUDA probe must degrade to "no accelerator", never crash generate.
    assert sb._host_has_accelerator() is False
    _clear_cache()
