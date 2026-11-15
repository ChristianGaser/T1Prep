"""The bash helpers around cat_surf: what they hand on, where they write, what they return.

Each helper (``CAT_SurfResampleMulti_ui``, ``CAT_Surf2ROIMulti_ui``,
``CAT_SurfParameters_ui``, ``CAT_VolSmooth_ui``) builds its job in bash and
hands it to Python (``cat_surf.cli``).  The interpreter is replaced here by a
stand-in that records what it is handed, so the tests need neither cat_surf nor
real surfaces and look at exactly the part that was wrong:

* a call ended with 0 whatever happened to its files,
* ``CAT_SurfResampleMulti_ui --out`` did not create the folder and ``--res``
  only renamed the output,
* ``CAT_Surf2ROIMulti_ui`` wrote below the current directory,
* ``CAT_SurfParameters_ui`` computed gyrification with absolute values for
  several files and without them for one,
* ``CAT_GrepJson`` printed its help and went on without a field or a file.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_DATA = _ROOT / "src" / "t1prep" / "data"

# Stands in for the interpreter: keeps the program it is given on stdin and
# the job it is given in the environment, and ends with $STUB_STATUS.
_STUB = '''\
#!/bin/sh
n=$(ls "$STUB_DIR" | grep -c '\\.job$')
cat > "$STUB_DIR/$n.program"
{
  echo "OUT_FILE=$OUT_FILE"
  echo "LH_UNIT=$LH_UNIT"
  echo "RH_UNIT=$RH_UNIT"
  echo "IN=$IN"
  echo "OUT=$OUT"
} > "$STUB_DIR/$n.job"
exit "${STUB_STATUS:-0}"
'''


@unittest.skipUnless(shutil.which("bash"), "bash not available")
class _WithAStandInInterpreter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.calls = self.tmp / "calls"
        self.calls.mkdir()
        self.stub = self.tmp / "python-stand-in"
        self.stub.write_text(_STUB)
        self.stub.chmod(0o755)
        self.cwd = self.tmp / "elsewhere"
        self.cwd.mkdir()

    def touch(self, *names, folder="sub/surf"):
        """Empty files in *folder*; returns the first."""
        base = self.tmp / folder
        base.mkdir(parents=True, exist_ok=True)
        for name in names:
            (base / name).write_bytes(b"")
        return base / names[0]

    def run_script(self, script, *args, status=0):
        """Run *script*; return (process, the jobs the interpreter was given)."""
        shutil.rmtree(self.calls)
        self.calls.mkdir()
        done = subprocess.run(
            [str(_SCRIPTS / script), *map(str, args)],
            capture_output=True, text=True, timeout=120, cwd=self.cwd,
            # The helpers run "${python:-python3}"
            env=dict(os.environ, python=str(self.stub), STUB_DIR=str(self.calls),
                     STUB_STATUS=str(status)))
        jobs = []
        for path in sorted(self.calls.glob("*.job"), key=lambda p: int(p.stem)):
            job = dict(line.split("=", 1) for line in path.read_text().splitlines())
            job["program"] = path.with_suffix(".program").read_text()
            jobs.append(job)
        return done, jobs


class TestResampling(_WithAStandInInterpreter):
    SCRIPT = "CAT_SurfResampleMulti_ui"

    def surface_data(self):
        return self.touch("lh.thickness.s", "rh.thickness.s")

    def test_the_output_folder_is_created(self):
        out = self.tmp / "new" / "deep"
        done, jobs = self.run_script(self.SCRIPT, "--out", out, self.surface_data())
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue(out.is_dir())
        self.assertEqual(Path(jobs[0]["OUT_FILE"]).parent, out)

    def test_the_resolution_picks_the_templates_not_only_the_name(self):
        for res in ("32k", "4k"):
            with self.subTest(res=res):
                _, jobs = self.run_script(self.SCRIPT, "--res", res, self.surface_data())
                templates = _DATA / f"templates_surfaces_{res}"
                self.assertIn(f"resampled_{res}", jobs[0]["OUT_FILE"])
                for hemi, unit in (("lh", "LH_UNIT"), ("rh", "RH_UNIT")):
                    self.assertIn(f"trg_sphere={templates}/{hemi}.sphere.freesurfer.gii",
                                  jobs[0][unit])
                    self.assertIn(f"mask={templates}/{hemi}.mask", jobs[0][unit])
                    self.assertTrue((templates / f"{hemi}.mask").is_file())

    def test_a_sphere_that_was_given_stays(self):
        _, jobs = self.run_script(self.SCRIPT, "--res", "4k", "--trg-sphere",
                                  "/x/lh.mine.gii", self.surface_data())
        self.assertIn("trg_sphere=/x/lh.mine.gii,", jobs[0]["LH_UNIT"])
        self.assertIn("trg_sphere=/x/rh.mine.gii,", jobs[0]["RH_UNIT"])

    def test_a_resolution_without_templates_is_an_error(self):
        done, jobs = self.run_script(self.SCRIPT, "--res", "99k", self.surface_data())
        self.assertEqual(done.returncode, 1)
        self.assertIn("99k", done.stderr)
        self.assertEqual(jobs, [])

    def test_a_file_that_failed_fails_the_call(self):
        done, _ = self.run_script(self.SCRIPT, self.surface_data(), status=3)
        self.assertEqual(done.returncode, 1)

    def test_a_file_it_cannot_use_fails_the_call(self):
        done, jobs = self.run_script(self.SCRIPT, self.touch("thickness.s"))
        self.assertEqual(done.returncode, 1)
        self.assertEqual(jobs, [])


class TestRegionTables(_WithAStandInInterpreter):
    SCRIPT = "CAT_Surf2ROIMulti_ui"
    ATLAS = "aparc_DK40.freesurfer"

    def surface_data(self, folder="sub/surf"):
        return self.touch("lh.thickness.s", "rh.thickness.s",
                          "lh.sphere.reg.s.gii", "rh.sphere.reg.s.gii", folder=folder)

    def test_the_tables_of_a_surf_folder_go_to_the_label_folder_next_to_it(self):
        done, jobs = self.run_script(self.SCRIPT, "--annot", self.ATLAS,
                                     self.surface_data())
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(
            Path(jobs[0]["OUT_FILE"]),
            self.tmp / "sub" / "label" / f"ROI.thickness.{self.ATLAS}_32k.s.json")
        self.assertTrue((self.tmp / "sub" / "label").is_dir())
        # ... and nothing below the directory the helper was called from
        self.assertEqual(list(self.cwd.iterdir()), [])

    def test_other_inputs_keep_their_tables_next_to_them(self):
        _, jobs = self.run_script(self.SCRIPT, "--annot", self.ATLAS,
                                  self.surface_data("plain"))
        self.assertEqual(Path(jobs[0]["OUT_FILE"]).parent, self.tmp / "plain")
        self.assertEqual(list(self.cwd.iterdir()), [])

    def test_the_atlas_comes_from_the_data_of_this_t1prep(self):
        _, jobs = self.run_script(self.SCRIPT, "--annot", self.ATLAS,
                                  self.surface_data())
        atlas = _DATA / "atlases_surfaces_32k" / f"lh.{self.ATLAS}.annot"
        self.assertIn(f"annot={atlas},", jobs[0]["LH_UNIT"])

    def test_a_file_that_failed_fails_the_call(self):
        done, _ = self.run_script(self.SCRIPT, "--annot", self.ATLAS,
                                  self.surface_data(), status=1)
        self.assertEqual(done.returncode, 1)

    def test_an_atlas_that_is_missing_fails_the_call(self):
        done, jobs = self.run_script(self.SCRIPT, "--annot", f"{self.ATLAS},no_such_atlas",
                                     self.surface_data())
        self.assertEqual(done.returncode, 1)
        self.assertEqual(len(jobs), 1)          # the atlas that exists is still done


class TestSurfaceParameters(_WithAStandInInterpreter):
    SCRIPT = "CAT_SurfParameters_ui"

    def surface(self):
        return self.touch("lh.central.s.gii", "rh.central.s.gii")

    def settings(self, *options):
        """(method, use_absolute) each hemisphere is computed with."""
        _, jobs = self.run_script(self.SCRIPT, *options, self.surface())
        found = set()
        for job in jobs:
            lines = {line.split("=")[0].strip(): line.split("=")[1].strip()
                     for line in job["program"].splitlines()
                     if line.startswith(("method", "use_absolute"))}
            found.add((lines["method"], lines["use_absolute"]))
        self.assertEqual(len(jobs), 2)          # left and right
        self.assertEqual(len(found), 1)
        return found.pop()

    def test_gyrification_is_absolute_with_and_without_its_option(self):
        # Several files are run through "--gy"; one file without an option
        # used to give the signed curvature under the same name
        self.assertEqual(self.settings(), ("0", "bool(1)"))
        self.assertEqual(self.settings("--gy"), ("0", "bool(1)"))

    def test_the_other_measures_keep_their_sign(self):
        self.assertEqual(self.settings("--mc"), ("4", "bool(0)"))
        self.assertEqual(self.settings("--gy", "--mc"), ("4", "bool(0)"))

    def test_a_file_that_failed_fails_the_call(self):
        done, _ = self.run_script(self.SCRIPT, self.surface(), status=1)
        self.assertEqual(done.returncode, 1)
        done, _ = self.run_script(self.SCRIPT, self.surface())
        self.assertEqual(done.returncode, 0)


class TestSmoothing(_WithAStandInInterpreter):
    SCRIPT = "CAT_VolSmooth_ui"

    def test_the_status_is_that_of_the_files(self):
        image = self.touch("t1.nii", folder="mri")
        done, jobs = self.run_script(self.SCRIPT, "--fwhm", "8", image)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(Path(jobs[0]["OUT"]), image.with_name("s8t1.nii"))
        done, _ = self.run_script(self.SCRIPT, image, status=1)
        self.assertEqual(done.returncode, 1)


@unittest.skipUnless(shutil.which("bash"), "bash not available")
class TestGrepJson(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run([str(_SCRIPTS / "CAT_GrepJson"), *args],
                              capture_output=True, text=True, timeout=60)

    def test_without_a_field_or_a_file_it_stops(self):
        for args, missing in ((("report.json",), "No field"),
                              (("--field", "subjectmeasures.vol_TIV"), "No JSON file")):
            with self.subTest(args=args):
                done = self.run_script(*args)
                self.assertEqual(done.returncode, 1)
                self.assertIn(missing, done.stderr)
                self.assertIn("USAGE:", done.stderr)
                self.assertEqual(done.stdout, "")       # jq was not run


@unittest.skipUnless(shutil.which("bash"), "bash not available")
class TestTheEvaluationScriptsNameNoMachine(unittest.TestCase):
    """``run_ants_batch.sh`` and ``run_ants_antspy.py`` had the folders of the
    machine they were run on written into them, one a scratch folder long gone."""

    TOOLS = _ROOT / "evaluation" / "tools"

    def test_no_path_of_a_home_or_a_scratch_folder(self):
        for name in ("run_ants_batch.sh", "run_ants_antspy.py"):
            with self.subTest(script=name):
                text = (self.TOOLS / name).read_text()
                self.assertNotIn("/private/tmp/", text)
                self.assertNotIn("/Users/", text)

    def test_the_folders_are_asked_for(self):
        env = {key: value for key, value in os.environ.items()
               if key not in ("SP", "MB", "ANTSPATH")}
        done = subprocess.run(["bash", str(self.TOOLS / "run_ants_batch.sh")],
                              capture_output=True, text=True, timeout=60, env=env)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("SP", done.stderr)
        self.assertTrue(os.access(self.TOOLS / "run_ants_antspy.py", os.X_OK))


class TestTheEnvironmentWrapper(unittest.TestCase):
    def test_the_quarantine_path_names_the_environment_once(self):
        text = (_SCRIPTS / "run_with_env.sh").read_text()
        self.assertNotIn('"$ENV_DIR""$ENV_DIR"', text)
        self.assertIn('"$ENV_DIR"/lib/python*/site-packages/PySide6', text)


class TestTheMakefile(unittest.TestCase):
    def test_no_target_works_on_a_folder_that_is_gone(self):
        text = (_ROOT / "Makefile").read_text()
        for folder in ("src/t1prep/bin",):
            self.assertFalse((_ROOT / folder).exists())
            self.assertNotIn(folder, text)


if __name__ == "__main__":
    unittest.main()
