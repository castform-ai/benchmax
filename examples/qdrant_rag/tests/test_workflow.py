from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

from benchmax.bundle import dump_bundle, load_bundle

_MAIN_PATH = Path(__file__).parents[1] / "main.py"
_SPEC = importlib.util.spec_from_file_location("qdrant_rag_example_main", _MAIN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LOCAL_MODULE_NAMES = ("data", "qdrant_rag_env", "search")
_PREVIOUS = {name: sys.modules.pop(name, None) for name in _LOCAL_MODULE_NAMES}
sys.path.insert(0, str(_MAIN_PATH.parent))
try:
    main = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = main
    _SPEC.loader.exec_module(main)
    _RUNTIME_MODULES = {
        name: sys.modules[name] for name in _LOCAL_MODULE_NAMES if name in sys.modules
    }
finally:
    sys.path.remove(str(_MAIN_PATH.parent))
    for _name, _module in _PREVIOUS.items():
        if _module is not None:
            sys.modules[_name] = _module
        else:
            sys.modules.pop(_name, None)


def test_runtime_bundle_excludes_castform_and_data_pipeline() -> None:
    secret = "qdrant_test_secret"
    with (
        patch.dict(sys.modules, _RUNTIME_MODULES),
        patch.object(sys, "path", [str(_MAIN_PATH.parent), *sys.path]),
    ):
        bundle = dump_bundle(
            main.QdrantRagEnv,
            constructor_args={
                "judge_base_url": "https://models.example/v1",
                "embedding_base_url": "https://models.example/v1",
                "url": "https://qdrant.example:6333",
                "api_key": secret,
            },
            pip_dependencies=main.RUNTIME_DEPENDENCIES,
        )
    env = load_bundle(bundle)

    assert env._search.available_modes == ["hybrid", "lexical", "vector"]
    assert secret.encode() in bundle.pickled
    assert b"castform.rag" not in bundle.pickled
    assert b"QdrantChunkSource" not in bundle.pickled
    assert bundle.metadata.pip_dependencies == ("qdrant-client<2,>=1.17.1",)


def test_runtime_dependency_list_does_not_include_castform() -> None:
    assert all("castform" not in dependency for dependency in main.RUNTIME_DEPENDENCIES)
