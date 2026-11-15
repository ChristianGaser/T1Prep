"""The source-tree installer judges its success by what it installs.

``scripts/T1Prep`` in a checkout (and ``scripts/install.sh``, which runs
``T1Prep --install``) builds ``env/`` from ``requirements.txt`` and then
installs T1Prep into it.  It used to decide whether that had worked with
``import deepmriprep``.  deepmriprep left the requirements in 0.9.0 — T1Prep
runs its networks itself — so from then on no fresh environment passed the
check: the installer stopped with "Installation of deepmriprep not
successful" before T1Prep was installed, and every later run of the checkout
tried the installation again.

The installer is driven here with a stand-in interpreter that records what it
is asked to do, so no environment is built and nothing is downloaded.
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

# Stands in for the interpreter: "-m venv" makes the folder, "-m pip" does
# nothing, an import check fails for the modules named in STUB_MISSING and
# anything else is left to the real interpreter.
_STUB = '''\
#!/bin/sh
echo "$*" >> "$STUB_LOG"
if [ "$1" = "-m" ]; then
  if [ "$2" = "venv" ]; then
    mkdir -p "$3/bin"
    : > "$3/bin/activate"
    ln -sf "$STUB_SELF" "$3/bin/python"
  fi
  exit 0
fi
if [ "$1" = "-c" ]; then
  for module in $STUB_MISSING; do
    case "$2" in *"$module"*) exit 1 ;; esac
  done
  case "$2" in "import "*|pass) exit 0 ;; esac
fi
exec "$REAL_PYTHON" "$@"
'''


@unittest.skipUnless(shutil.which("bash"), "bash not available")
class TestTheSourceTreeInstaller(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "bin").mkdir()
        self.stub = self.tmp / "bin" / "python3"
        self.stub.write_text(_STUB)
        self.stub.chmod(0o755)
        self.root = self.tmp / "checkout"
        self.root.mkdir()
        (self.root / "requirements.txt").write_text("cat-surf\n")
        self.env = self.root / "env"
        self.log = self.tmp / "calls.log"

    def run_function(self, function, missing, re_install=0):
        """Call *function* of T1Prep_utils.sh; return (process, what the interpreter was asked)."""
        snippet = (
            f'source "{_UTILS}" >/dev/null 2>&1; '
            f'root_dir="{self.root}"; T1prep_env="{self.env}"; python=python3; '
            f'T1PREP_INSTALLED=0; re_install={re_install}; os_type=Linux; '
            f'{function}; echo "returned $?"'
        )
        environment = dict(
            os.environ, STUB_LOG=str(self.log), STUB_MISSING=missing,
            STUB_SELF=str(self.stub), REAL_PYTHON=sys.executable,
            PATH=os.pathsep.join([str(self.tmp / "bin"), "/usr/bin", "/bin"]))
        self.log.write_text("")
        done = subprocess.run(["bash", "-c", snippet], capture_output=True,
                              text=True, timeout=180, env=environment)
        return done, self.log.read_text().splitlines()

    def test_a_fresh_environment_has_no_deepmriprep_and_installs_all_the_same(self):
        done, asked = self.run_function("install_python_libraries", "deepmriprep")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("returned 0", done.stdout)
        self.assertNotIn("deepmriprep", done.stdout + done.stderr)
        self.assertIn(f"-m pip install --no-compile -r {self.root}/requirements.txt", asked)
        # T1Prep itself comes last: that is what the old check never reached
        self.assertEqual(asked[-1], f"-m pip install --no-compile --no-deps {self.root}")

    def test_it_asks_for_the_libraries_it_installed(self):
        _, asked = self.run_function("install_python_libraries", "deepmriprep")
        checks = [line for line in asked if line.startswith("-c import")]
        self.assertEqual(checks, ["-c import cat_surf, torch"])

    def test_missing_libraries_stop_it_before_t1prep_is_installed(self):
        for missing in ("cat_surf", "torch"):
            with self.subTest(missing=missing):
                done, asked = self.run_function("install_python_libraries", missing)
                self.assertEqual(done.returncode, 1)
                self.assertIn("Installation of the Python libraries not successful",
                              done.stderr)
                self.assertIn("requirements.txt", done.stderr)
                self.assertFalse([line for line in asked if "--no-deps" in line])

    def test_an_environment_that_has_the_libraries_is_left_alone(self):
        # As every start of T1Prep from a checkout checks it
        self.run_function("install_python_libraries", "deepmriprep")
        done, asked = self.run_function("check_python_libraries", "deepmriprep")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("returned 0", done.stdout)
        self.assertFalse([line for line in asked if "pip" in line], asked)

    def test_an_environment_without_them_is_installed_again(self):
        self.run_function("install_python_libraries", "deepmriprep")
        done, asked = self.run_function("check_python_libraries", "cat_surf")
        self.assertEqual(done.returncode, 1)
        self.assertTrue([line for line in asked if "requirements.txt" in line])

    def test_install_keeps_a_complete_environment_and_re_install_replaces_it(self):
        # T1Prep --install used to be --re-install: it removed env/ first
        self.run_function("install_python_libraries", "deepmriprep")
        kept = self.env / "kept"
        kept.write_text("")
        done, asked = self.run_function("check_python_libraries", "deepmriprep")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue(kept.exists())
        self.assertFalse([line for line in asked if "pip" in line])

        done, asked = self.run_function(
            "check_python_libraries", "deepmriprep", re_install=1)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertFalse(kept.exists())
        self.assertTrue([line for line in asked if "requirements.txt" in line])

    def test_the_two_options_are_told_apart(self):
        show = ('source "$T1PREP" && parse_args "$@" && '
                'printf "%s %s\\n" "$install_only" "$re_install"')
        for option, expected in (("--install", "1 0"), ("--re-install", "0 1")):
            with self.subTest(option=option):
                done = subprocess.run(
                    ["bash", "-c", show, "bash", option],
                    capture_output=True, text=True, timeout=120,
                    env=dict(os.environ, T1PREP=str(_ROOT / "scripts" / "T1Prep")))
                self.assertEqual(done.stdout.strip().splitlines()[-1], expected,
                                 done.stderr)

    def test_nothing_in_the_scripts_asks_for_deepmriprep(self):
        for name in ("T1Prep", "T1Prep_utils.sh", "install.sh"):
            text = (_ROOT / "scripts" / name).read_text()
            with self.subTest(script=name):
                self.assertNotIn("import deepmriprep", text.replace(
                    'The check used to be "import deepmriprep"', ""))


if __name__ == "__main__":
    unittest.main()
