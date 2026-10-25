"""How the bash orchestrator reads its command line and hands file names on.

Up to 0.8.0 ``scripts/T1Prep`` matched options by prefix (``--p*`` took
``--pial-white``, ``--profile`` and ``--python=...`` for "save native p
maps"; ``--out-dir=/r in.nii`` used the next word as the folder and lost the
input), sourced the word after any ``--defaults`` on the line (even when it was
another option's value), re-split the arguments it passed to the ``--multi``
jobs, and ran the segmentation and every ``parallelize`` job through ``eval``,
so a file name with ``$(...)`` in it was executed.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_T1PREP = _ROOT / "scripts" / "T1Prep"
_UTILS = _ROOT / "scripts" / "T1Prep_utils.sh"
_PARALLELIZE = _ROOT / "scripts" / "parallelize"

# Sourcing T1Prep defines its functions; main only runs when it is executed
_SHOW = (
    'source "$T1PREP" && parse_args "$@" && '
    'for v in save_p save_pial_white outdir multi python save_hemi '
    'save_lesions min_memory initial_surface atlas_surf seed; do '
    'printf "%s=%s\\n" "$v" "${!v}"; done && '
    'printf "file=%s\\n" "${ARRAY[@]}"'
)


class _InATempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "f.nii").write_bytes(b"")

    def parse(self, *args):
        """Run parse_args on *args*; return (returncode, {var: value}, files)."""
        done = subprocess.run(
            ["bash", "-c", _SHOW, "bash", *args],
            capture_output=True, text=True, timeout=120, cwd=self.tmp,
            env=dict(os.environ, T1PREP=str(_T1PREP)),
        )
        values, files = {}, []
        for line in done.stdout.splitlines():
            key, _, value = line.partition("=")
            if key == "file":
                files.append(value)
            else:
                values[key] = value
        return done, values, files


class TestOptionsMatchExactly(_InATempDir):
    def test_pial_white_is_not_native_p_maps(self):
        _, v, _ = self.parse("--pial-white", "f.nii")
        self.assertEqual((v["save_pial_white"], v["save_p"]), ("1", "0"))

    def test_python_with_equals_sign(self):
        _, v, _ = self.parse("--python=/usr/bin/python3", "f.nii")
        self.assertEqual((v["python"], v["save_p"]), ("/usr/bin/python3", "0"))

    def test_value_after_equals_sign_keeps_the_input(self):
        _, v, files = self.parse("--out-dir=/r", "--multi=4", "f.nii")
        self.assertEqual((v["outdir"], v["multi"], files), ("/r", "4", ["f.nii"]))

    def test_unknown_option_is_an_error(self):
        done, _, _ = self.parse("--profile", "f.nii")
        self.assertEqual(done.returncode, 1)
        self.assertIn("Unrecognized option", done.stderr)

    def test_the_old_long_spellings_still_work(self):
        _, v, _ = self.parse("--hemisphere", "--lesion", "--min-mem", "8",
                             "--initial-surf", "s.gii", "f.nii")
        self.assertEqual(
            (v["save_hemi"], v["save_lesions"], v["min_memory"], v["initial_surface"]),
            ("1", "1", "8", "s.gii"),
        )


class TestTheDefaultsFile(_InATempDir):
    def test_a_value_reading_defaults_is_not_sourced(self):
        marker = self.tmp / "sourced"
        (self.tmp / "evil.sh").write_text(f'touch "{marker}"\n')
        self.parse("--atlas-surf", "--defaults", "evil.sh", "f.nii")
        self.assertFalse(marker.exists())

    def test_defaults_with_equals_sign(self):
        defaults = (_ROOT / "T1Prep_defaults.txt").read_text()
        (self.tmp / "my.txt").write_text(defaults + "\nseed=7\n")
        _, v, _ = self.parse("--defaults=my.txt", "f.nii")
        self.assertEqual(v["seed"], "7")


class TestMultiKeepsTheArguments(unittest.TestCase):
    def test_filter_arguments_returns_whole_words(self):
        snippet = (
            f'source "{_UTILS}"; ARRAY=(f.nii); '
            "filter_arguments --atlas \"'a', 'b'\" f.nii --out-dir '/x y'; "
            'printf "<%s>\\n" "${FILTERED_ARGS[@]}"'
        )
        done = subprocess.run(["bash", "-c", snippet], capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(done.stdout.split("\n")[:4],
                         ["<--atlas>", "<'a', 'b'>", "<--out-dir>", "</x y>"])


class TestT1PrepRunParser(unittest.TestCase):
    def test_pial_white_is_on_unless_switched_off(self):
        import sys
        sys.path.insert(0, str(_ROOT / "src"))
        from t1prep.t1prep import _build_parser

        parser = _build_parser()
        self.assertTrue(parser.parse_args(["x.nii"]).pial_white)
        self.assertTrue(parser.parse_args(["x.nii", "--pial-white"]).pial_white)
        self.assertFalse(parser.parse_args(["x.nii", "--no-pial-white"]).pial_white)


class TestParallelizeNeverRunsAFileName(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # Runs in self.tmp, so the command substitution would create it there
        self.marker = self.tmp / "PWNED"
        self.evil = self.tmp / "x$(touch PWNED).nii"
        self.evil.write_bytes(b"")
        (self.tmp / "y.nii").write_bytes(b"")

    def run_parallelize(self, processes):
        return subprocess.run(
            [str(_PARALLELIZE), "--processes", str(processes),
             "--logdir", str(self.tmp), "--command", "printf '%s\\n'",
             str(self.evil), str(self.tmp / "y.nii")],
            capture_output=True, text=True, timeout=300, cwd=self.tmp,
        )

    def test_serially(self):
        done = self.run_parallelize(1)
        self.assertFalse(self.marker.exists())
        self.assertIn(str(self.evil), done.stdout)

    def test_in_parallel(self):
        self.run_parallelize(2)
        self.assertFalse(self.marker.exists())
        logs = "".join(p.read_text() for p in self.tmp.glob("parallelize_*.log"))
        self.assertIn(str(self.evil), logs)


if __name__ == "__main__":
    unittest.main()
