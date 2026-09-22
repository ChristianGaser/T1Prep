"""``t1prep.cat_surf`` re-exports everything the installed cat-surf exports.

The wrapper's explicit import list went stale as cat-surf grew:
``vol_pbt_barrier_reference`` and 13 other public functions were missing, so
code that probed the wrapper concluded the installed cat-surf lacked them.
"""

import pytest

cat_surf = pytest.importorskip("cat_surf")

import t1prep.cat_surf as wrapper  # noqa: E402


def test_every_public_symbol_is_reexported():
    missing = [n for n in getattr(cat_surf, "__all__", ()) if not hasattr(wrapper, n)]
    assert not missing


def test_reexports_are_the_same_objects():
    for name in getattr(cat_surf, "__all__", ()):
        assert getattr(wrapper, name) is getattr(cat_surf, name), name
