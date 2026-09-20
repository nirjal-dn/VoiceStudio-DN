"""Every builtin translation engine's package must be a declared dependency.

``TRANSLATE_PROVIDER`` defaults to ``"google"``
(``api/routers/dub_translate.py``), which is backed by ``deep_translator`` —
and that package was never declared in ``pyproject.toml``. The router imports
cleanly at startup and only pulls the provider in per-request, so a fresh
install did not fail loudly at boot: every Dub translation returned
``ImportError`` while the backend looked healthy.

The registry already carries the fact that decides this. ``builtin: True``
means "ships with the app, no user install step" — argos, nllb and openai are
all marked and all declared. A reviewer adding a fifth builtin has no way to
remember the pyproject half, so it is checked here rather than remembered.

Failing here means one of two edits: declare the package in
``[project].dependencies``, or drop ``builtin: True`` and let the engine
catalogue offer its install command like any other optional engine.
"""
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYPROJECT = os.path.join(_ROOT, "pyproject.toml")

sys.path.insert(0, os.path.join(_ROOT, "backend"))


def _declared_dependencies() -> set:
    """Distribution names in ``[project].dependencies``, normalised.

    PEP 503 treats ``-``/``_``/``.`` as equivalent and names as
    case-insensitive, so ``deep-translator`` in pyproject and
    ``deep_translator`` as an import both land on the same key.
    """
    with open(_PYPROJECT, encoding="utf-8") as fh:
        src = fh.read()
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", src, re.S | re.M)
    assert block, "could not find [project].dependencies in pyproject.toml"
    names = set()
    for raw in re.findall(r'"([^"]+)"', block.group(1)):
        # Strip extras, markers and version specifiers: "pkg[x]>=1 ; sys=='y'".
        name = re.split(r"[<>=!~;\[]", raw, 1)[0].strip()
        if name:
            names.add(re.sub(r"[-_.]+", "-", name).lower())
    return names


def test_every_builtin_engine_has_its_package_declared():
    from services import translation_engines as te

    declared = _declared_dependencies()
    missing = []
    for eid, spec in te.REGISTRY.items():
        if not spec.get("builtin"):
            continue
        pkg = spec.get("pip_package")
        if pkg is None:
            continue  # rides a package another entry already declares
        if re.sub(r"[-_.]+", "-", pkg).lower() not in declared:
            missing.append(f"{eid} needs {pkg!r}")

    assert not missing, (
        "builtin translation engines whose package is not in "
        f"[project].dependencies: {missing}. A builtin engine must work on a "
        "fresh install with no extra pip step."
    )


def test_the_default_provider_is_a_builtin():
    """Whatever TRANSLATE_PROVIDER falls back to must ship with the app.

    A default that needs a user install is the ImportError above, rearranged.
    """
    from services import translation_engines as te

    default = "google"  # api/routers/dub_translate.py's fallback
    spec = te.REGISTRY.get(default)
    assert spec is not None, f"default provider {default!r} is not in the registry"
    assert spec.get("builtin"), (
        f"default provider {default!r} is not builtin — either mark it builtin "
        "and declare its package, or change the default in dub_translate.py"
    )
