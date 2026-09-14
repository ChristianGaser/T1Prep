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

    def test_an_existing_venv_is_activated(self):
        env_dir = self.tmp / "env" / "bin"
        env_dir.mkdir(parents=True)
        (env_dir / "activate").write_text("export T1PREP_TEST_ACTIVATED=1\n")
        (env_dir / "python").write_text("")

        done = probe(
            self.tmp / "scripts",
            'activate_t1prep_env; echo "A=${T1PREP_TEST_ACTIVATED:-} P=${python}"',
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("A=1", done.stdout)
        self.assertIn(f"P={self.tmp / 'env' / 'bin' / 'python'}", done.stdout)


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


if __name__ == "__main__":
    unittest.main()
