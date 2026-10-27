import sys
import tempfile
import unittest
from pathlib import Path

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import deepmriprep.preprocess  # noqa: F401 - loads every DATA_PATH binder
    import deepmriprep.register as _register
except ImportError:  # pragma: no cover - deepmriprep is optional for the test env
    _register = None

if _register is not None:
    from t1prep import _models


def _bound_modules():
    """deepmriprep modules that hold their own ``DATA_PATH`` binding."""
    return {
        name: module
        for name, module in list(sys.modules.items())
        if name.split(".")[0] == "deepmriprep" and hasattr(module, "DATA_PATH")
    }


@unittest.skipIf(_register is None, "deepmriprep not installed")
class TestRedirectDeepmriprepPaths(unittest.TestCase):
    """A model directory outside deepmriprep (``T1PREP_MODEL_DIR``, user cache)."""

    def setUp(self):
        self._saved_paths = {n: m.DATA_PATH for n, m in _bound_modules().items()}
        self._saved_bet = dict(sys.modules["deepmriprep.preprocess"].BET_MODEL_PATHS)
        self._saved_model_dir = _models.MODEL_DIR
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name)
        _models.MODEL_DIR = self.data / "models"

    def tearDown(self):
        for name, path in self._saved_paths.items():
            sys.modules[name].DATA_PATH = path
        sys.modules["deepmriprep.preprocess"].BET_MODEL_PATHS = self._saved_bet
        _models.MODEL_DIR = self._saved_model_dir

    def test_every_data_path_binding_is_redirected(self):
        _models._redirect_deepmriprep_paths()
        bound = _bound_modules()
        for name in ("utils", "preprocess", "segment", "register", "atlas"):
            self.assertIn(f"deepmriprep.{name}", bound)
        for name, module in bound.items():
            self.assertEqual(module.DATA_PATH, str(self.data), name)

    def test_default_location_is_left_alone(self):
        _models.MODEL_DIR = _models._DEEPMRIPREP_MODELS
        _models._redirect_deepmriprep_paths()
        for name, module in _bound_modules().items():
            self.assertEqual(module.DATA_PATH, self._saved_paths[name], name)


if __name__ == "__main__":
    unittest.main()
