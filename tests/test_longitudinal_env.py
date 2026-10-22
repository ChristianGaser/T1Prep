"""How the bash wrappers find the Python environment they run in.

``T1Prep_utils.sh`` resolves two layouts: a source checkout, where the package
sits at ``<repo>/src/t1prep`` and a project-managed venv is expected at
``<repo>/env``, and an installed tree, where pip put the scripts in
``<venv>/bin`` next to the interpreter that has t1prep.

The longitudinal wrappers used to hard-code ``<repo>/env`` and abort with
"Virtual environment not found" whenever it was absent -- which is every
pip-installed tree, and every checkout of someone who runs the pip-installed
pipeline.  ``activate_t1prep_env`` is what they share instead.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_UTILS = _ROOT / "scripts" / "T1Prep_utils.sh"

WRAPPERS = [
    "realign_longitudinal.sh",
    "warp_longitudinal.sh",
    "modulate_longitudinal.sh",
]


def probe(script_dir, snippet, env=None):
    """Source T1Prep_utils.sh from *script_dir* and run *snippet* after it."""
    return subprocess.run(
        ["bash", "-c", f'source "{script_dir}/T1Prep_utils.sh"; {snippet}'],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


class TestASourceCheckoutWithoutAVenv(unittest.TestCase):
    """A checkout whose <repo>/env was never created still has to work."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        # A minimal source tree: scripts/ next to src/t1prep/, and no env/.
        (self.tmp / "scripts").mkdir()
        package = self.tmp / "src" / "t1prep"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('__version__ = "9.9.9"\n')
        (package / "data").mkdir()
        shutil.copy(_UTILS, self.tmp / "scripts" / "T1Prep_utils.sh")

        # An inherited $T1PREP_PYTHON is an explicit choice, which bypasses
        # both the venv and the search these tests are about.
        self.env = {k: v for k, v in os.environ.items() if k != "T1PREP_PYTHON"}

    def runner_path(self):
        """Put a GitHub runner's interpreters on PATH: a system python3.12
        without T1Prep's dependencies ahead of the python3 that has them."""
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        bare = bin_dir / "python3.12"
        bare.write_text(
            '#!/bin/sh\n[ "$1" = "-V" ] && echo "Python 3.12.0" && exit 0\nexit 1\n'
        )
        good = bin_dir / "python3"
        good.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        for exe in (bare, good):
            exe.chmod(0o755)
        site = self.tmp / "site"
        site.mkdir()
        (site / "cat_surf.py").write_text("")

        self.env["PATH"] = os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"])
        self.env["PYTHONPATH"] = str(site)
        return bare

    def fake_venv(self):
        """A <repo>/env whose activation leaves a mark in the environment."""
        env_dir = self.tmp / "env" / "bin"
        env_dir.mkdir(parents=True)
        (env_dir / "activate").write_text("export T1PREP_TEST_ACTIVATED=1\n")
        (env_dir / "python").write_text("")
        return env_dir / "python"

    def test_the_missing_venv_is_not_fatal(self):
        done = probe(self.tmp / "scripts", 'activate_t1prep_env; echo "OK ${python}"')
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertNotIn("Virtual environment not found", done.stdout + done.stderr)
        self.assertIn("OK ", done.stdout)

    def test_the_checkout_is_put_first_on_pythonpath(self):
        # PYTHONPATH is searched before site-packages, so this is what keeps a
        # checkout from silently running stale code out of a pip install.
        done = probe(self.tmp / "scripts", 'activate_t1prep_env; echo "PP=${PYTHONPATH}"')
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"PP={self.tmp / 'src'}", done.stdout)

    def test_an_interpreter_without_the_dependencies_is_passed_over(self):
        # The name order of check_python_cmd picks the system python3.12.
        self.runner_path()
        done = probe(
            self.tmp / "scripts",
            'activate_t1prep_env; echo "P=${python}"; '
            '"${python}" -c "import t1prep; print(t1prep.__version__)"',
            env=self.env,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertNotIn("P=python3.12", done.stdout)
        self.assertIn("9.9.9", done.stdout)

    def test_an_explicit_interpreter_is_not_second_guessed(self):
        # The search replaces only a pick by name: a $T1PREP_PYTHON without
        # the dependencies stays, so the import error names the interpreter
        # that was asked for.
        bare = self.runner_path()
        self.env["T1PREP_PYTHON"] = str(bare)
        done = probe(self.tmp / "scripts", 'activate_t1prep_env; echo "P=${python}"', env=self.env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"P={bare}", done.stdout)

    def test_an_existing_venv_is_activated(self):
        venv_python = self.fake_venv()
        done = probe(
            self.tmp / "scripts",
            'activate_t1prep_env; echo "A=${T1PREP_TEST_ACTIVATED:-} P=${python}"',
            env=self.env,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("A=1", done.stdout)
        self.assertIn(f"P={venv_python}", done.stdout)

    def test_t1prep_python_wins_over_the_venv(self):
        # Wrappers that do not parse --python themselves used to activate the
        # venv on top of it.
        self.fake_venv()
        self.env["T1PREP_PYTHON"] = sys.executable
        done = probe(
            self.tmp / "scripts",
            'activate_t1prep_env; echo "A=${T1PREP_TEST_ACTIVATED:-} P=${python}"',
            env=self.env,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"A= P={sys.executable}", done.stdout)


class TestAnInstalledTree(unittest.TestCase):
    """<venv>/bin: the interpreter next door already has t1prep."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        # bin/ without a ../src/t1prep sibling is what marks installed mode.
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        shutil.copy(_UTILS, self.bin / "T1Prep_utils.sh")

        # The probe needs *an* interpreter that can import t1prep; point it at
        # this checkout rather than at whatever the machine has installed.
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(_ROOT / "src")
        self.env["T1PREP_PYTHON"] = sys.executable

    def test_nothing_is_prepended_to_pythonpath(self):
        # The venv already resolves t1prep; prepending would be meaningless at
        # best and would shadow the installed package at worst.
        done = probe(self.bin, 'activate_t1prep_env; echo "PP=${PYTHONPATH}"', env=self.env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"PP={_ROOT / 'src'}", done.stdout)

    def test_the_venv_interpreter_is_used_as_is(self):
        done = probe(
            self.bin,
            'activate_t1prep_env; echo "MODE=${T1PREP_INSTALLED} P=${python}"',
            env=self.env,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("MODE=1", done.stdout)
        self.assertIn(f"P={sys.executable}", done.stdout)


class TestTheWrappersUseIt(unittest.TestCase):
    """No wrapper may go back to hard-coding <repo>/env."""

    def test_no_wrapper_hardcodes_the_env_directory(self):
        for name in WRAPPERS:
            with self.subTest(script=name):
                text = (_ROOT / "scripts" / name).read_text()
                self.assertIn("activate_t1prep_env", text)
                self.assertNotIn('ENV_DIR="$PROJECT_DIR/env"', text)

    def test_every_wrapper_answers_version_with_the_release(self):
        from t1prep import __version__

        for name in WRAPPERS:
            with self.subTest(script=name):
                done = subprocess.run(
                    [str(_ROOT / "scripts" / name), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertEqual(done.stdout.strip(), f"{name} {__version__}")


class TestDiceChecksTheImportItRuns(unittest.TestCase):
    """dice.sh's dependency check is the import of t1prep.dice itself.

    t1prep.dice needs only numpy/nibabel/scipy, but importing it runs the
    package __init__, which needs cat_surf and torch too -- the module list
    dice.sh used to check passed, and the import then ended in a traceback.
    """

    def run_dice(self, script_dir, *args):
        return subprocess.run(
            [str(script_dir / "dice.sh"), *args],
            capture_output=True,
            text=True,
            timeout=180,
            env=dict(os.environ, T1PREP_PYTHON=sys.executable),
        )

    def test_a_missing_dependency_of_the_package_is_reported(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "scripts").mkdir()
        package = tmp / "src" / "t1prep"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            '__version__ = "9.9.9"\nimport t1prep_test_missing_dependency\n'
        )
        for name in ("T1Prep_utils.sh", "dice.sh"):
            shutil.copy(_ROOT / "scripts" / name, tmp / "scripts" / name)

        done = self.run_dice(tmp / "scripts", "--gt", "gt.nii.gz", "--pred", "pred.nii.gz")
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("cannot import t1prep", done.stderr)
        self.assertIn("t1prep_test_missing_dependency", done.stderr)
        self.assertNotIn("Traceback", done.stderr)

    def test_the_metric_and_version_match_the_module(self):
        import nibabel as nib
        import numpy as np

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rng = np.random.default_rng(0)
        gt = rng.integers(0, 4, size=(12, 10, 8)).astype(np.int16)
        pred = np.where(rng.random(gt.shape) < 0.1, 0, gt).astype(np.int16)
        for name, data in (("gt", gt), ("pred", pred)):
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(tmp / f"{name}.nii.gz"))
        files = ["--gt", str(tmp / "gt.nii.gz"), "--pred", str(tmp / "pred.nii.gz")]

        module = subprocess.run(
            [sys.executable, "-m", "t1prep.dice", *files],
            capture_output=True,
            text=True,
            timeout=180,
            env=dict(os.environ, PYTHONPATH=str(_ROOT / "src")),
        )
        self.assertEqual(module.returncode, 0, module.stdout + module.stderr)
        done = self.run_dice(_ROOT / "scripts", *files)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(done.stdout, module.stdout)

        from t1prep import __version__

        done = self.run_dice(_ROOT / "scripts", "--version")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(done.stdout.strip(), f"dice.sh {__version__}")


if __name__ == "__main__":
    unittest.main()
