"""The Python orchestrator (t1prep-run, run_t1prep, Docker) behaves like bash.

Up to 0.8.0 ``t1prep.py`` resolved symlinks before naming outputs (on a
DataLad/git-annex dataset the results went to ``.git/annex/objects``, named
after the hash), ignored the presets of ``--fmriprep``, kept the atlases on
under ``--fast`` (the argparse defaults were strings, so the ``None`` checks
never fired), and its ``--no-overwrite`` globbed ``*PATTERN*`` over the whole
folder, so every subject was skipped once the first one was done.  Both bash
checks of ``--no-overwrite`` were wrong as well (one had the patterns
swapped, the other a stray ``/``).
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep import t1prep as orchestrator  # noqa: E402

_UTILS = _ROOT / "scripts" / "T1Prep_utils.sh"


class _InATempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


class TestSymlinkedInputs(_InATempDir):
    def test_an_annexed_input_keeps_its_dataset_layout(self):
        objects = self.tmp / "ds" / ".git" / "annex" / "objects" / "Xk"
        objects.mkdir(parents=True)
        (objects / "MD5E-s1--abc.nii.gz").write_bytes(b"")
        anat = self.tmp / "ds" / "sub-01" / "anat"
        anat.mkdir(parents=True)
        link = anat / "sub-01_T1w.nii.gz"
        link.symlink_to(objects / "MD5E-s1--abc.nii.gz")

        outdir0, use_subfolder, bname = orchestrator._resolve_output_dirs(
            str(link), None, "0.8.0"
        )
        self.assertEqual(bname, "sub-01_T1w")
        self.assertFalse(use_subfolder)
        self.assertEqual(
            outdir0, self.tmp / "ds" / "derivatives" / "T1Prep-v0.8.0" / "sub-01" / "anat"
        )


class TestNoOverwriteLooksForThisSubject(_InATempDir):
    def bash(self, outdir0, bname, pattern, bids):
        snippet = f'source "{_UTILS}"; already_processed "$@" && echo yes || echo no'
        done = subprocess.run(
            ["bash", "-c", snippet, "bash", str(outdir0), bname, pattern, str(int(bids))],
            capture_output=True, text=True, timeout=120,
        )
        return done.stdout.strip() == "yes"

    def check(self, outdir0, bname, pattern, bids, expected):
        for how, found in (
            ("python", orchestrator._already_processed(outdir0, bname, pattern, bids)),
            ("bash", self.bash(outdir0, bname, pattern, bids)),
        ):
            with self.subTest(how=how, bname=bname):
                self.assertIs(found, expected)

    def test_cat12_names(self):
        (self.tmp / "surf").mkdir()
        (self.tmp / "surf" / "lh.thickness.subA").write_bytes(b"")
        self.check(self.tmp, "subA", "surf/lh.thickness.", False, True)
        self.check(self.tmp, "subB", "surf/lh.thickness.", False, False)

    def test_bids_names(self):
        (self.tmp / "sub-01_label-GM_probseg.nii.gz").write_bytes(b"")
        self.check(self.tmp, "sub-01_T1w", "label-GM_probseg", True, True)
        self.check(self.tmp, "sub-02_T1w", "label-GM_probseg", True, False)


class TestPresets(_InATempDir):
    def run_main(self, *argv):
        (self.tmp / "x.nii").write_bytes(b"")
        with mock.patch.object(orchestrator, "_process_single", return_value=0) as single:
            orchestrator.main([str(self.tmp / "x.nii"), *argv])
        return single.call_args.kwargs

    def test_fmriprep_switches_on_what_fmriprep_reads(self):
        kw = self.run_main("--fmriprep")
        self.assertTrue(kw["bids"] and kw["gz"] and kw["csf"] and kw["p"])
        self.assertFalse(kw["save_mwp"])
        self.assertFalse(kw["hemisphere"])
        self.assertEqual(kw["atlas"], "")

    def test_fast_switches_the_atlases_off(self):
        kw = self.run_main("--fast")
        self.assertEqual((kw["atlas"], kw["atlas_surf"]), ("", ""))

    def test_an_explicit_atlas_survives_fast(self):
        kw = self.run_main("--fast", "--atlas", "'cobra'")
        self.assertEqual(kw["atlas"], "'cobra'")


class TestLongData(_InATempDir):
    """--long-data is a file, as in bash.  It used to be searched as a folder,
    and a file silently fell back to the non-realigned input."""

    def setUp(self):
        super().setUp()
        for name in ("a.nii", "b.nii", "ra.nii"):
            (self.tmp / name).write_bytes(b"")

    def test_the_file_is_what_gets_segmented(self):
        with mock.patch.object(orchestrator.subprocess, "call", return_value=0) as call:
            rc = orchestrator._process_single(
                str(self.tmp / "a.nii"), out_dir=str(self.tmp / "out"),
                long_data=str(self.tmp / "ra.nii"), skullstrip_only=True,
            )
        self.assertEqual(rc, 0)
        cmd = call.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--input") + 1], str(self.tmp / "ra.nii"))

    def test_one_input_and_an_existing_file_only(self):
        with self.assertRaises(ValueError):
            orchestrator.run_t1prep([str(self.tmp / "a.nii"), str(self.tmp / "b.nii")],
                                    long_data=str(self.tmp / "ra.nii"))
        with self.assertRaises(ValueError):
            orchestrator.run_t1prep(str(self.tmp / "a.nii"), long_data=str(self.tmp))
        with self.assertRaises(SystemExit):
            orchestrator.main([str(self.tmp / "a.nii"), "--long-data",
                               str(self.tmp / "missing.nii")])


if __name__ == "__main__":
    unittest.main()
