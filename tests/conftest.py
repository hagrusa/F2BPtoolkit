"""pytest configuration for F2BPtoolkit tests."""

import pytest


def pytest_collection_modifyitems(config, items):
    """
    Skip tests marked @pytest.mark.integration if the C++ extension
    is not importable (i.e., the package has not been built yet).
    """
    try:
        from f2bptoolkit import _core  # noqa: F401
        _core_available = True
    except ImportError:
        _core_available = False

    if not _core_available:
        skip_marker = pytest.mark.skip(
            reason="C++ extension '_core' not built — run: pip install -e ."
        )
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip_marker)
