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
    'for v in save_p save_pial_white outdir multi python save_hemi keep_hemi '
    'save_lesions min_memory initial_surface atlas_surf seed long_data nn_backend; do '
    'printf "%s=%s\\n" "$v" "${!v}"; done && '
    'printf "file=%s\\n" "${ARRAY[@]}"'
)


class _InATempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "f.nii").write_bytes(b"")

    def parse(self, *args, env=None, show=_SHOW):
        """Run parse_args on *args*; return (returncode, {var: value}, files)."""
        done = subprocess.run(
            ["bash", "-c", show, "bash", *args],
            capture_output=True, text=True, timeout=120, cwd=self.tmp,
            env=dict(os.environ, T1PREP=str(_T1PREP), **(env or {})),
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

    def test_no_atlas_surf_empties_the_surface_list(self):
        _, v, _ = self.parse("f.nii")
        self.assertIn("aparc_DK40.freesurfer", v["atlas_surf"])
        _, v, _ = self.parse("--no-atlas-surf", "f.nii")
        self.assertEqual(v["atlas_surf"], "")
        # The last word counts, as for every option
        _, v, _ = self.parse("--no-atlas-surf", "--atlas-surf", "'aparc_HCP_MMP1'", "f.nii")
        self.assertEqual(v["atlas_surf"], "'aparc_HCP_MMP1'")

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


class TestHemisphereMaps(_InATempDir):
    """Surfaces always need the hemisphere maps (save_hemi); only --hemi
    keeps them afterwards (keep_hemi)."""

    def test_surfaces_write_but_do_not_keep_them(self):
        _, v, _ = self.parse("f.nii")
        self.assertEqual((v["save_hemi"], v["keep_hemi"]), ("1", "0"))

    def test_hemi_keeps_them(self):
        _, v, _ = self.parse("--hemi", "f.nii")
        self.assertEqual((v["save_hemi"], v["keep_hemi"]), ("1", "1"))


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


class TestNetworkBackend(_InATempDir):
    """--nn-backend over T1PREP_NN_BACKEND over nn_backend in the defaults
    file; the choice is exported for the Python steps and the --multi jobs."""

    def test_the_defaults_file(self):
        _, v, _ = self.parse("f.nii", env={"T1PREP_NN_BACKEND": ""})
        self.assertEqual(v["nn_backend"], "torch")

    def test_the_environment_overrides_the_defaults_file(self):
        _, v, _ = self.parse("f.nii", env={"T1PREP_NN_BACKEND": "tinygrad"})
        self.assertEqual(v["nn_backend"], "tinygrad")

    def test_the_option_overrides_both(self):
        _, v, _ = self.parse("--nn-backend", "torch", "f.nii",
                             env={"T1PREP_NN_BACKEND": "tinygrad"})
        self.assertEqual(v["nn_backend"], "torch")

    def test_it_is_exported(self):
        show = 'source "$T1PREP" && parse_args "$@" && printenv T1PREP_NN_BACKEND'
        done, _, _ = self.parse("--nn-backend=tinygrad", "f.nii", show=show)
        self.assertEqual(done.stdout.strip(), "tinygrad")


class TestLongData(_InATempDir):
    """--long-data names the file processed in place of the one input."""

    def setUp(self):
        super().setUp()
        (self.tmp / "r.nii").write_bytes(b"")
        (self.tmp / "g.nii").write_bytes(b"")

    def test_a_file_for_one_input(self):
        done, v, files = self.parse("--long-data", "r.nii", "f.nii")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual((v["long_data"], files), ("r.nii", ["f.nii"]))

    def test_anything_else_is_an_error(self):
        (self.tmp / "folder").mkdir()
        for args, message in (
            (("--long-data", "missing.nii", "f.nii"), "is not a file"),
            (("--long-data", "folder", "f.nii"), "is not a file"),
            (("--long-data", "r.nii", "f.nii", "g.nii"), "stands for one input"),
        ):
            with self.subTest(args=args):
                done, _, _ = self.parse(*args)
                self.assertEqual(done.returncode, 1)
                self.assertIn(message, done.stderr)


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


class TestParallelizeReportsFailures(unittest.TestCase):
    """The exit status says whether every item succeeded.

    With one process ``parallelize`` runs the items one after the other, and
    that branch returned 0 whatever had happened: ``T1Prep --multi 1`` (and
    any batch on a machine whose memory allows one job only) exited 0 with
    failed images.  Only the branch for several processes named them.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.files = []
        for name in ("a", "bad one", "c"):
            path = self.tmp / f"{name}.nii"
            path.write_bytes(b"")
            self.files.append(str(path))
        self.step = self.tmp / "step.sh"
        self.step.write_text(
            '#!/bin/bash\ncase "$1" in *bad*) exit 3;; esac\ntouch "$1.done"\n')
        self.step.chmod(0o755)

    def run_parallelize(self, processes, files):
        return subprocess.run(
            [str(_PARALLELIZE), "--processes", str(processes),
             "--logdir", str(self.tmp), "--command", str(self.step), *files],
            capture_output=True, text=True, timeout=300, cwd=self.tmp,
        )

    def check_a_failure(self, processes):
        done = self.run_parallelize(processes, self.files)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("with 1 error(s)", done.stdout)
        listed = done.stdout.split("Failed items:", 1)[1]
        self.assertIn(self.files[1], listed)
        self.assertNotIn(self.files[0], listed)
        # The others were still processed
        self.assertTrue(Path(self.files[0] + ".done").exists())
        self.assertTrue(Path(self.files[2] + ".done").exists())

    def test_one_process(self):
        self.check_a_failure(1)

    def test_several_processes(self):
        self.check_a_failure(2)

    def test_success_is_still_zero(self):
        good = [self.files[0], self.files[2]]
        for processes in (1, 2):
            with self.subTest(processes=processes):
                done = self.run_parallelize(processes, good)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertNotIn("Failed items", done.stdout)


if __name__ == "__main__":
    unittest.main()
