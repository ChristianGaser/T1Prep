"""The web UI form starts from what T1Prep itself would do.

Up to 0.8.2 the form read ``T1Prep_defaults.txt`` from ``BASE_DIR``, which an
installed ``t1prep-ui`` sets to the working directory, so it never found the
file next to the ``T1Prep`` it runs and started from fallbacks of its own:
no pial/white surfaces (the default since March 2026) and a "24" GB minimum
memory (T1Prep's is 10).  The default volume atlases were never ticked either,
because the defaults file names them in lower case; T1Prep computed them all
the same, as an empty selection sent no ``--atlas``.
"""

import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from t1prep.webui import app as webui
except ImportError:  # pragma: no cover - flask is a dependency
    webui = None


def _checked(page: str, name: str, value: str = "1") -> bool:
    """Whether the checkbox *name* with *value* is ticked on *page*."""
    tag = re.search(
        rf'<input type="checkbox" name="{re.escape(name)}" '
        rf'value="{re.escape(value)}"[^>]*>',
        page,
    )
    if tag is None:
        raise AssertionError(f"no checkbox {name}={value} on the page")
    return "checked" in tag.group(0)


def _value(page: str, name: str) -> str:
    """The value the input *name* starts with on *page*."""
    return re.search(rf'name="{re.escape(name)}" value="([^"]*)"', page).group(1)


def _selected(page: str) -> str:
    """The network backend selected on *page*."""
    return re.search(r'<option value="([^"]+)" selected', page).group(1)


@unittest.skipIf(webui is None, "web UI not importable")
class TestTheFormStartsFromTheDefaultsFile(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        patcher = mock.patch.dict(os.environ, {webui.BACKEND_ENV: ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def t1prep(self, folder: str, defaults: str = None, beside: bool = False) -> Path:
        """A T1Prep in tmp/*folder*, with *defaults* next to it or one up."""
        script = self.tmp / folder / "T1Prep"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/usr/bin/env bash\n")
        if defaults is not None:
            where = script.parent.parent if beside else script.parent
            (where / "T1Prep_defaults.txt").write_text(defaults)
        return script

    def page(self, script: Path) -> str:
        with mock.patch.object(webui, "T1PREP_CMD", str(script)):
            return webui.app.test_client().get("/").get_data(as_text=True)

    def test_installed_next_to_t1prep_whatever_the_working_directory(self):
        # A decoy where the form used to look
        decoy = self.tmp / "cwd"
        decoy.mkdir()
        (decoy / "T1Prep_defaults.txt").write_text("save_pial_white=1\nmin_memory=24\n")
        script = self.t1prep(
            "venv/bin", "save_pial_white=0\nmin_memory=7\nnn_backend='tinygrad'\n"
        )
        with mock.patch.object(webui, "BASE_DIR", decoy):
            page = self.page(script)
        self.assertFalse(_checked(page, "--pial-white"))
        self.assertEqual(_value(page, "--min-memory"), "7")
        self.assertEqual(_selected(page), "tinygrad")

    def test_a_source_checkout_keeps_it_at_the_root(self):
        script = self.t1prep("repo/scripts", "min_memory=12\n", beside=True)
        self.assertEqual(_value(self.page(script), "--min-memory"), "12")

    def test_the_shipped_defaults(self):
        script = self.t1prep(
            "venv/bin", (_ROOT / "T1Prep_defaults.txt").read_text()
        )
        page = self.page(script)
        self.assertTrue(_checked(page, "--pial-white"))
        self.assertTrue(_checked(page, "save_mwp"))
        self.assertEqual(_value(page, "--min-memory"), "10")
        self.assertEqual(_value(page, "--multi"), "-1")
        self.assertEqual(_selected(page), "torch")
        # Named in lower case in the defaults file, looked up case-insensitively
        self.assertTrue(_checked(page, "atlas_choice", "Neuromorphometrics"))
        self.assertTrue(_checked(page, "atlas_choice", "CoBra"))
        self.assertFalse(_checked(page, "atlas_choice", "SUIT"))
        for name in ("aparc_DK40.freesurfer", "aparc_a2009s.freesurfer"):
            self.assertTrue(_checked(page, "surface_atlas_choice", name))
        self.assertFalse(_checked(page, "surface_atlas_choice", "aparc_HCP_MMP1"))

    def test_without_a_defaults_file_it_falls_back_to_the_shipped_one(self):
        shipped = self.t1prep(
            "venv/bin", (_ROOT / "T1Prep_defaults.txt").read_text()
        )
        missing = self.t1prep("empty/bin")
        self.assertEqual(self.page(missing), self.page(shipped))

    def test_the_environment_overrides_the_defaults_file(self):
        script = self.t1prep("venv/bin", "nn_backend='torch'\n")
        with mock.patch.dict(os.environ, {webui.BACKEND_ENV: "tinygrad"}):
            self.assertEqual(_selected(self.page(script)), "tinygrad")


@unittest.skipIf(webui is None, "web UI not importable")
class TestTheCommandLine(unittest.TestCase):
    def cmd(self, atlas_override=None, **form):
        return webui.build_command(
            dict(save_mwp="1", **form), [Path("x.nii")], atlas_override=atlas_override
        )

    def test_the_network_backend_is_forwarded(self):
        cmd = self.cmd(**{"--nn-backend": "tinygrad"})
        self.assertEqual(cmd[cmd.index("--nn-backend") + 1], "tinygrad")
        with self.assertRaises(ValueError):
            self.cmd(**{"--nn-backend": "jax"})

    def test_no_atlas_ticked_means_no_atlases(self):
        self.assertIn("--no-atlas", self.cmd())
        cmd = self.cmd(atlas_override="'CoBra'")
        self.assertNotIn("--no-atlas", cmd)
        self.assertEqual(cmd[cmd.index("--atlas") + 1], "'CoBra'")


if __name__ == "__main__":
    unittest.main()
