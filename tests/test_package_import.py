"""``import t1prep`` does not load the networks' framework.

The package used to import ``utils`` -- and with it torch -- in its
``__init__``, so every tool paid about a second for it on start: the viewers,
the launcher and each ``--help``, none of which runs a network.  What the
package exports is now imported on first use (PEP 562), and the name table the
launcher needs lives in a module of its own (``t1prep._names``).

Each case runs in a fresh interpreter: what is loaded is a property of the
process, and this one has long imported everything.
"""

import subprocess
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def in_a_fresh_interpreter(program):
    """Run *program* with the checkout on the path; return its output."""
    done = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(_SRC)!r})\n{program}"],
        capture_output=True, text=True, timeout=180)
    if done.returncode != 0:
        raise AssertionError(done.stderr[-2000:])
    return done.stdout.strip()


class TestWhatAStartLoads(unittest.TestCase):
    #: Modules a tool imports before it does anything; none of them runs a network
    LIGHT = [
        "t1prep",
        "t1prep.cli_help",
        "t1prep._names",
        "t1prep.t1prep",                 # the launcher asks it where results are
        "t1prep.launch.options",
        "t1prep.launch.results",
        "t1prep.launch.progress",
        "t1prep.plot_histogram",
        "t1prep.plot_boxplot",
        "t1prep.gui.make_apps",
    ]
    #: The same for the windows, where Qt and VTK are there to import them
    WINDOWS = ["t1prep.gui.launcher", "t1prep.gui.cat_vol_view", "t1prep.gui.cat_surf_view"]

    def loads_torch(self, module):
        return in_a_fresh_interpreter(
            f"import {module}\nprint('torch' in sys.modules)") == "True"

    def test_no_tool_loads_torch_to_start(self):
        for module in self.LIGHT:
            with self.subTest(module=module):
                self.assertFalse(self.loads_torch(module))

    def test_no_window_loads_torch_to_open(self):
        try:
            import PySide6  # noqa: F401
            import vtkmodules  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional GUI stack
            self.skipTest(f"no Qt/VTK: {exc}")
        for module in self.WINDOWS:
            with self.subTest(module=module):
                self.assertFalse(self.loads_torch(module))

    def test_the_mps_settings_are_made_before_torch_can_be_imported(self):
        # torch reads them when it is loaded, so they belong to the package
        # import, which now comes long before
        if sys.platform != "darwin":
            self.skipTest("macOS only")
        out = in_a_fresh_interpreter(
            "import os\n"
            "os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)\n"
            "import t1prep\n"
            "print(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'), 'torch' in sys.modules)")
        self.assertEqual(out, "1 False")


class TestTheExportsAreStillThere(unittest.TestCase):
    def test_every_exported_name_resolves(self):
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional heavy deps
            self.skipTest(f"torch unavailable: {exc}")
        out = in_a_fresh_interpreter(
            "import t1prep\n"
            "from t1prep import run_t1prep, get_filenames, compute_dice_nifti\n"
            "import t1prep.t1prep, t1prep.utils, t1prep.metrics\n"
            "assert run_t1prep is t1prep.t1prep.run_t1prep\n"
            "assert t1prep.run_t1prep is run_t1prep        # not the submodule\n"
            "assert get_filenames is t1prep.utils.get_filenames\n"
            "missing = [n for n in t1prep.__all__ if not hasattr(t1prep, n)]\n"
            "print(missing, sorted(set(t1prep.__all__) - set(dir(t1prep))))")
        self.assertEqual(out, "[] []")

    def test_a_submodule_is_reached_as_an_attribute(self):
        # As after the eager imports: t1prep.utils without importing it by name
        out = in_a_fresh_interpreter(
            "import t1prep\n"
            "print(t1prep.cli_help.__name__, t1prep._names.__name__)")
        self.assertEqual(out, "t1prep.cli_help t1prep._names")

    def test_a_name_that_does_not_exist_is_an_attribute_error(self):
        out = in_a_fresh_interpreter(
            "import t1prep\n"
            "print(hasattr(t1prep, 'no_such_thing'), hasattr(t1prep, '__wrapped__'))\n"
            "try:\n"
            "    from t1prep import no_such_thing\n"
            "except ImportError as exc:\n"
            "    print(type(exc).__name__)")
        self.assertEqual(out.split(), ["False", "False", "ImportError"])

    def test_the_name_table_is_one_object_under_both_names(self):
        try:
            from t1prep import _names, utils
        except Exception as exc:  # pragma: no cover - optional heavy deps
            self.skipTest(f"t1prep.utils unavailable: {exc}")
        for name in ("NameTable", "name_file", "DATA_PATH_T1PREP", "_resolve_names_tsv"):
            with self.subTest(name=name):
                self.assertIs(getattr(utils, name), getattr(_names, name))
        self.assertTrue(Path(_names.name_file).is_file())


if __name__ == "__main__":
    unittest.main()
