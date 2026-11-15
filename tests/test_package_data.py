"""What the wheel ships of ``src/t1prep/data``.

The data folders are listed one by one in ``pyproject.toml``.  A folder that
is renamed there but not here is silently left out of the wheel: the pattern
``data/templates_surfaces/**`` matched nothing for months, so the 164k
templates ``CAT_SurfView`` matches overlays against were missing from every
installation.
"""

import re
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "src" / "t1prep" / "data"
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#: Left out of the wheel on purpose: the weights are downloaded on first use
#: (they would take the wheel over PyPI's size limit)
_NOT_SHIPPED = {"models"}


def shipped_folders():
    """The ``data/<folder>/**`` patterns of the package data in pyproject.toml."""
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text[text.index("[tool.setuptools.package-data]"):]
    return set(re.findall(r'"data/([\w.-]+)/\*\*"', block))


@unittest.skipUnless(_DATA.is_dir(), "no source tree")
class TestTheWheelShipsTheDataFolders(unittest.TestCase):
    def folders(self):
        return {path.name for path in _DATA.iterdir()
                if path.is_dir() and path.name != "__pycache__"
                and any(item.is_file() for item in path.iterdir())}

    def test_every_data_folder_is_shipped(self):
        self.assertEqual(self.folders() - _NOT_SHIPPED - shipped_folders(), set())

    def test_every_pattern_names_a_folder_that_exists(self):
        self.assertEqual(shipped_folders() - self.folders(), set())

    def test_the_viewer_s_template_folders_are_among_them(self):
        from t1prep.gui import cat_surf_view

        for folder in cat_surf_view.TEMPLATE_HEMI_POINTS.values():
            with self.subTest(folder=folder):
                self.assertIn(folder, shipped_folders())


@unittest.skipUnless((_DATA / "models").is_dir(), "no model archives in this tree")
class TestOnlyTheNetworksInUseAreKept(unittest.TestCase):
    def test_no_archive_of_a_network_t1prep_does_not_run(self):
        # warp_model.pt (90 MB) and segmentation_nogm_model.pt stayed in the
        # repository long after shooting and the nogm rule had replaced them
        from t1prep.nn.models import NETWORKS

        archives = {path.stem for path in (_DATA / "models").glob("*.pt")}
        self.assertEqual(archives - set(NETWORKS), set())


if __name__ == "__main__":
    unittest.main()
