"""Regression tests for the longitudinal rigid realignment CLI.

These cover output-path handling rather than the registration itself: three
separate failures all came from the same place, and all of them only appeared
after the (slow) registration had already run.

* ``--use-skullstrip`` imported ``segment`` as a top-level module even though it
  uses package-relative imports, so it died with "attempted relative import with
  no known parent package".
* Only the ``--out-subfolders`` branch created its destination, so a plain
  ``--out-dir`` that did not exist yet failed at ``nib.save``.
* The in-place safety check named in the comment was never actually written --
  the loop computed both paths and compared nothing -- so an ``--out-dir`` equal
  to the input folder silently overwrote the inputs.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep import realign_longitudinal as R  # noqa: E402


def _write_pair(directory: Path):
    """Two small volumes with a little structure, offset from each other."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -20.0
    paths = []
    for idx in range(2):
        data = np.zeros((20, 22, 20), dtype=np.float32)
        start = 5 + idx  # a one-voxel shift, so alignment has work to do
        data[start : start + 9, 6:16, 5:14] = 1.0
        data[start + 2 : start + 6, 9:13, 8:11] = 0.5
        path = directory / f"tp{idx + 1}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(path))
        paths.append(str(path))
    return paths


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "t1prep.realign_longitudinal", *args],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_SRC)},
    )


class TestOutputPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.inputs = _write_pair(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_update_headers_creates_a_missing_out_dir(self):
        out_dir = self.tmp / "does" / "not" / "exist"
        result = _run(
            ["--inputs", *self.inputs, "--out-dir", str(out_dir), "--update-headers"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("tp1.nii.gz", "tp2.nii.gz"):
            self.assertTrue((out_dir / name).is_file(), f"missing {name}")

    def test_save_resampled_creates_a_missing_out_dir(self):
        out_dir = self.tmp / "resampled"
        result = _run(
            [
                "--inputs", *self.inputs,
                "--out-dir", str(out_dir),
                "--save-resampled",
                "--save-template",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((out_dir / "tp1_desc-realigned.nii.gz").is_file())
        self.assertTrue((out_dir / "reference.nii.gz").is_file())

    def test_update_headers_refuses_to_overwrite_the_inputs(self):
        # The destination equals the inputs' own folder, so writing would
        # replace the originals with header-modified copies.
        result = _run(
            ["--inputs", *self.inputs, "--out-dir", str(self.tmp), "--update-headers"]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("overwrite the input", result.stderr)

    def test_out_subfolders_still_work(self):
        out_dir = self.tmp / "sub"
        result = _run(
            [
                "--inputs", *self.inputs,
                "--out-dir", str(out_dir),
                "--out-subfolders", "a", "b",
                "--update-headers",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((out_dir / "a" / "tp1.nii.gz").is_file())
        self.assertTrue((out_dir / "b" / "tp2.nii.gz").is_file())

    def test_rejects_mismatched_subfolder_count(self):
        result = _run(
            [
                "--inputs", *self.inputs,
                "--out-dir", str(self.tmp / "out"),
                "--out-subfolders", "only-one",
                "--update-headers",
            ]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("out-subfolders", result.stderr)


class TestSkullstripImport(unittest.TestCase):
    """The import path that ``--use-skullstrip`` takes.

    The models themselves are stubbed out: what is under test is that
    ``segment`` is reached through the package, not that skull-stripping works.
    """

    def test_reaches_segment_through_the_package(self):
        import t1prep.segment as segment

        originals = {
            name: getattr(segment, name)
            for name in (
                "prepare_model_files",
                "setup_device",
                "CustomPreprocess",
                "skull_strip",
            )
        }
        segment.prepare_model_files = lambda *a, **k: None
        segment.setup_device = lambda *a, **k: (None, True)
        segment.CustomPreprocess = lambda *a, **k: object()
        segment.skull_strip = lambda prep, img, **k: (img, None, None)
        try:
            images = [
                nib.Nifti1Image(np.zeros((6, 6, 6), dtype=np.float32), np.eye(4))
                for _ in range(2)
            ]
            stripped = R._skullstrip_for_realign(images, verbose=False)
            self.assertEqual(len(stripped), 2)
        finally:
            for name, value in originals.items():
                setattr(segment, name, value)

    def test_module_does_not_put_its_own_directory_on_sys_path(self):
        # That is what made a bare ``import segment`` resolve to the package
        # module without its package, which is the bug this guards against.
        package_dir = str(Path(R.__file__).resolve().parent)
        self.assertNotIn(package_dir, sys.path)


if __name__ == "__main__":
    unittest.main()
