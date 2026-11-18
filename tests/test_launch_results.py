"""Where the results of an image are, and how they are looked at.

The front ends ask ``t1prep.launch.results`` which phases of a subject are
complete, what its report says and which viewer call shows it.  The answers
come from ``Names.tsv`` and the folder rules of ``scripts/T1Prep``; these
tests lay out result folders the way T1Prep does and ask.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep.launch import results as res  # noqa: E402
from t1prep.launch.inputs import find_images  # noqa: E402

REPORT = {
    "software": {"version_t1prep": "0.9.0", "date": "20261102-151301"},
    "subjectmeasures": {
        "vol_TIV": {"value": 1433.02},
        "vol_abs_CGW": {"value": [348.4, 572.53, 512.09]},
        "vol_rel_CGW": {"value": [0.24312, 0.39953, 0.35735]},
        "vol_WMH": {"value": 0.0},
    },
    "qualitymeasures": {
        "IQR": {"value": 2.08, "grade": "A-"},
        "res_RMS": {"value": 1.0324},
        "euler_lh": {"value": 88}, "euler_rh": {"value": 176},
    },
}


class _Folders(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def touch(self, path: Path, text: str = "") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def cat12(self, name="sub1", surfaces=True, compressed=False) -> Path:
        """An image with CAT12-style results next to it; returns the image."""
        ext = "nii.gz" if compressed else "nii"
        image = self.touch(self.tmp / "data" / f"{name}.nii.gz")
        base = image.parent
        self.touch(base / "report" / f"log_{name}.json", json.dumps(REPORT))
        self.touch(base / "report" / f"T1Preplog_{name}.txt")
        self.touch(base / "mri" / f"p0{name}.{ext}")
        self.touch(base / "mri" / f"m{name}.{ext}")
        if surfaces:
            for hemi in ("lh", "rh"):
                self.touch(base / "surf" / f"{hemi}.central.{name}.gii")
                self.touch(base / "surf" / f"{hemi}.thickness.{name}")
        return image


class TestWhereTheResultsAre(_Folders):
    def test_next_to_the_image_in_cat12_folders(self):
        subject = res.subject_for_input(self.cat12())
        self.assertEqual(subject.name, "sub1")
        self.assertEqual(subject.folder, self.tmp / "data")
        self.assertTrue(subject.subfolders)
        self.assertEqual(subject.find("Label_volume"),
                         self.tmp / "data" / "mri" / "p0sub1.nii")
        self.assertEqual(subject.find("Mid_surface", "right"),
                         self.tmp / "data" / "surf" / "rh.central.sub1.gii")
        self.assertEqual(subject.log_file().name, "T1Preplog_sub1.txt")

    def test_compressed_volumes_are_found_as_well(self):
        subject = res.subject_for_input(self.cat12(compressed=True))
        self.assertEqual(subject.find("Label_volume").name, "p0sub1.nii.gz")

    def test_an_output_folder_takes_the_cat12_folders(self):
        image = self.touch(self.tmp / "in" / "s.nii")
        subject = res.subject_for_input(image, out_dir=self.tmp / "out")
        self.assertEqual(subject.expected("Report_file"),
                         self.tmp / "out" / "report" / "log_s.json")

    def test_a_bids_dataset_gets_a_derivatives_folder(self):
        image = self.touch(
            self.tmp / "study" / "sub-01" / "ses-1" / "anat" / "sub-01_ses-1_T1w.nii.gz")
        subject = res.subject_for_input(image, bids=True)
        self.assertFalse(subject.subfolders)
        self.assertEqual(subject.folder.parts[-3:], ("sub-01", "ses-1", "anat"))
        self.assertEqual(subject.folder.parents[3], self.tmp / "study" / "derivatives")
        # BIDS names drop the _T1w of the image and say L/R for the hemisphere
        self.assertEqual(subject.expected("Report_file").name, "sub-01_ses-1_log.json")
        self.assertEqual(subject.expected("GMT_shape", "left").name,
                         "sub-01_ses-1_hemi-L_thickness.shape.gii")
        self.assertEqual(subject.expected("Label_volume", nii_ext="nii.gz").parent,
                         subject.folder)

    def test_the_underlay_is_the_image_or_what_t1prep_kept_of_it(self):
        image = self.cat12()
        subject = res.subject_for_input(image)
        self.assertEqual(subject.underlay(), image)
        found = res.find_results(self.tmp / "data")[0]
        self.assertIsNone(found.input)
        self.assertEqual(found.underlay().name, "msub1.nii")


class TestWhichPhasesAreComplete(_Folders):
    def test_the_report_ends_the_volumes_the_thickness_the_surfaces(self):
        subject = res.subject_for_input(self.cat12(surfaces=False))
        self.assertTrue(subject.stage_done("volume"))
        self.assertFalse(subject.stage_done("surface"))
        self.touch(subject.expected("GMT_shape", "left"))
        self.assertFalse(subject.stage_done("surface"))       # one hemisphere
        self.touch(subject.expected("GMT_shape", "right"))
        self.assertTrue(subject.stage_done("surface"))

    def test_results_of_an_earlier_run_do_not_count_for_this_one(self):
        subject = res.subject_for_input(self.cat12())
        report = subject.find("Report_file")
        os.utime(report, (1000, 1000))
        self.assertTrue(subject.stage_done("volume"))
        self.assertFalse(subject.stage_done("volume", since=2000))
        self.assertTrue(subject.stage_done("volume", since=500))

    def test_skull_stripping_alone_writes_the_stripped_image_and_no_report(self):
        # save_skullstrip_only_outputs() in segment.py writes it, with the mask
        image = self.touch(self.tmp / "s.nii")
        subject = res.subject_for_input(image)
        self.assertEqual(subject.expected("skullstripped_volume").name, "ms.nii")
        self.touch(subject.expected("skullstripped_volume"))
        self.assertFalse(subject.stage_done("volume"))
        self.assertTrue(subject.stage_done("volume", skullstrip_only=True))


class TestTheResultsOfAnImageWhereverTheyAre(_Folders):
    """An image added to the table showed results only where a run with the
    current options would write them.  In a BIDS dataset that is the folder
    of the running T1Prep version, and with either layout it is one of the
    two naming schemes -- so the results of an earlier version, or of a run
    with the other names, did not count as results of the image."""

    def dataset(self, version="0.6.0", bids_names=False, amap=False):
        """A BIDS dataset with one image and its results of *version*."""
        root = self.tmp / "study"
        image = self.touch(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
        anat = (root / "derivatives" / f"T1Prep{'Amap' if amap else ''}-v{version}"
                / "sub-01" / "anat")
        if bids_names:
            self.touch(anat / "sub-01_log.json", json.dumps(REPORT))
            self.touch(anat / "sub-01_label-seg.nii.gz")
        else:
            self.touch(anat / "log_sub-01.json", json.dumps(REPORT))
            self.touch(anat / "p0sub-01.nii")
        return image, anat

    def test_where_a_run_writes_is_looked_at_first(self):
        image = self.cat12()
        subject = res.subject_for_input(image)
        self.assertEqual(res.results_on_disk(subject), subject)

    def test_nothing_found_still_says_where_they_will_be(self):
        subject = res.subject_for_input(self.touch(self.tmp / "new.nii"))
        self.assertIs(res.results_on_disk(subject), subject)

    def test_an_earlier_version_of_a_bids_dataset(self):
        image, anat = self.dataset("0.6.0")
        subject = res.subject_for_input(image)
        self.assertNotEqual(subject.folder, anat)          # the current version's
        self.assertFalse(subject.stage_done("volume"))
        found = res.results_on_disk(subject)
        self.assertEqual(found.folder, anat)
        self.assertTrue(found.stage_done("volume"))
        self.assertEqual(found.find("Label_volume").name, "p0sub-01.nii")
        # The picture is still drawn on the image itself
        self.assertEqual(found.underlay(), image)
        self.assertEqual(res.quality(found.report()), ("A-", 2.08))

    def test_the_newest_of_several_versions(self):
        image, _old = self.dataset("0.6.0")
        _image, newer = self.dataset("0.8.10")
        _image, _amap = self.dataset("0.8.99", amap=True)
        found = res.results_on_disk(res.subject_for_input(image))
        # 0.8.10 is newer than 0.6.0 -- as numbers, not as text -- and the
        # folders of the other kind of run (AMAP) come after those of this one
        self.assertEqual(found.folder, newer)
        with_amap = res.results_on_disk(res.subject_for_input(image, amap=True))
        self.assertEqual(with_amap.folder.parents[1].name, "T1PrepAmap-v0.8.99")

    def test_the_other_names(self):
        image, anat = self.dataset("0.6.0", bids_names=True)
        for chosen in (False, True):
            with self.subTest(bids_option=chosen):
                found = res.results_on_disk(res.subject_for_input(image, bids=chosen))
                self.assertTrue(found.bids)
                self.assertEqual(found.find("Label_volume"), anat / "sub-01_label-seg.nii.gz")

    def test_the_other_names_next_to_the_image(self):
        image = self.cat12()
        subject = res.subject_for_input(image, bids=True)      # BIDS names ticked
        self.assertFalse(subject.stage_done("volume"))
        found = res.results_on_disk(subject)
        self.assertFalse(found.bids)
        self.assertTrue(found.stage_done("volume") and found.stage_done("surface"))


class TestFindingResults(_Folders):
    def test_both_layouts_below_a_folder(self):
        self.cat12("a")
        self.cat12("b", surfaces=False)
        anat = self.tmp / "data" / "derivatives" / "T1Prep-v0.9.0" / "sub-01" / "anat"
        self.touch(anat / "sub-01_log.json", json.dumps(REPORT))
        self.touch(self.tmp / "data" / "report" / "a_glued-lh.json", "{}")   # not a report
        found = {subject.name: subject for subject in res.find_results(self.tmp)}
        self.assertEqual(sorted(found), ["a", "b", "sub-01"])
        self.assertTrue(found["a"].subfolders and not found["a"].bids)
        self.assertEqual(found["a"].folder, self.tmp / "data")
        self.assertTrue(found["sub-01"].bids)
        self.assertEqual(found["sub-01"].folder, anat)
        self.assertTrue(found["a"].stage_done("surface"))
        self.assertFalse(found["b"].stage_done("surface"))

    def test_a_result_folder_stands_for_the_one_above_it(self):
        self.cat12("a")
        for folder in ("mri", "report", "surf"):
            found = res.find_results(self.tmp / "data" / folder)
            self.assertEqual([subject.name for subject in found], ["a"])

    def test_nothing_to_find(self):
        self.assertEqual(res.find_results(self.tmp), [])


class TestTheReport(_Folders):
    def test_grade_and_measures(self):
        subject = res.subject_for_input(self.cat12())
        report = subject.report()
        self.assertEqual(res.quality(report), ("A-", 2.08))
        sections = dict(res.describe(report))
        self.assertEqual(sections["Quality"][0], ("Image quality (IQR)", "A-  2.08"))
        volumes = dict(sections["Volumes"])
        self.assertEqual(volumes["Total intracranial"], "1433 mL")
        self.assertEqual(volumes["Grey matter"], "573 mL  ·  40.0 %")
        self.assertNotIn("WM hyperintensities", volumes)      # none measured
        self.assertEqual(dict(sections["Surfaces"])["Euler number, left | right"],
                         "88 | 176")
        self.assertEqual(dict(sections["Run"])["T1Prep"], "0.9.0")

    def test_a_report_that_is_missing_or_broken(self):
        subject = res.subject_for_input(self.touch(self.tmp / "s.nii"))
        self.assertIsNone(subject.report())
        self.assertEqual(res.describe(None), [])
        self.assertIsNone(res.quality({}))
        self.touch(subject.expected("Report_file"), "{ not json")
        self.assertIsNone(subject.report())


class TestTheViewerCalls(_Folders):
    def test_each_view_names_its_viewer_and_files(self):
        image = self.cat12()
        subject = res.subject_for_input(image)
        seg = res.segmentation_view(subject)
        self.assertEqual(seg[:3], [sys.executable, "-m", res.VOLUME_VIEWER])
        self.assertEqual(seg[3], str(image))
        self.assertEqual(Path(seg[seg.index("--contour") + 1]).name, "p0sub1.nii")
        surfaces = res.surfaces_view(subject)
        self.assertEqual([Path(item).name for item in surfaces[4:]],
                         ["lh.central.sub1.gii", "rh.central.sub1.gii"])
        thickness = res.thickness_view(subject)
        self.assertEqual(thickness[2], res.SURFACE_VIEWER)
        self.assertEqual(Path(thickness[3]).name, "lh.thickness.sub1")

    def test_both_borders_of_the_grey_matter_are_outlined(self):
        # Towards the CSF and towards the white matter (label map: 1, 2, 3):
        # one outline at the default level drew the outer border alone
        subject = res.subject_for_input(self.cat12())
        for command in (res.segmentation_view(subject),
                        res.preview(subject, "volume", self.tmp / "p.png")):
            at = command.index("--contour-level")
            self.assertEqual(command[at + 1:at + 3], ["1.5", "2.5"])
            self.assertEqual(command.count("--contour"), 1)

    def test_a_view_per_picture(self):
        subject = res.subject_for_input(self.cat12())
        self.assertEqual(res.PICTURES, ("volume", "surface", "thickness"))
        self.assertEqual(res.view(subject, "volume"), res.segmentation_view(subject))
        self.assertEqual(res.view(subject, "surface"), res.surfaces_view(subject))
        self.assertEqual(res.view(subject, "thickness"), res.thickness_view(subject))

    def test_no_view_without_its_files(self):
        subject = res.subject_for_input(self.cat12(surfaces=False))
        self.assertIsNotNone(res.segmentation_view(subject))
        self.assertIsNone(res.surfaces_view(subject))
        self.assertIsNone(res.thickness_view(subject))
        for picture in ("surface", "thickness"):
            self.assertIsNone(res.preview(subject, picture, self.tmp / "p.png"))
            self.assertEqual(res.preview_sources(subject, picture), [])

    def test_the_picture_is_rendered_without_a_window(self):
        subject = res.subject_for_input(self.cat12())
        for picture, shown in (("volume", "--contour"), ("surface", "lh.central.sub1.gii")):
            command = res.preview(subject, picture, self.tmp / "p.png")
            self.assertEqual(command[2], res.VOLUME_VIEWER)
            self.assertIn("--headless", command)
            self.assertIn("--no-crosshair", command)
            self.assertEqual(command[command.index("--screenshot") + 1],
                             str(self.tmp / "p.png"))
            self.assertTrue(any(shown in item for item in command))
            self.assertGreaterEqual(len(res.preview_sources(subject, picture)), 2)

    def test_the_thickness_is_drawn_on_four_views_of_the_surface(self):
        subject = res.subject_for_input(self.cat12())
        command = res.preview(subject, "thickness", self.tmp / "p.png", size=300)
        self.assertEqual(command[:3], [sys.executable, "-m", res.SURFACE_VIEWER])
        self.assertEqual(Path(command[3]).name, "lh.thickness.sub1")
        # --output is the surface viewer's batch mode: no window, no Dock icon
        self.assertEqual(command[command.index("--output") + 1], str(self.tmp / "p.png"))
        self.assertEqual(command[command.index("--views") + 1], "4")
        at = command.index("--size")
        self.assertEqual(command[at + 1:at + 3], ["600", "450"])
        # Stale when the values or the surface they are drawn on change
        self.assertEqual(
            sorted(path.name for path in res.preview_sources(subject, "thickness")),
            ["lh.central.sub1.gii", "lh.thickness.sub1",
             "rh.central.sub1.gii", "rh.thickness.sub1"])

    def test_one_hemisphere_is_enough_for_the_thickness(self):
        image = self.cat12()
        for name in ("lh.thickness.sub1", "lh.central.sub1.gii"):
            (self.tmp / "data" / "surf" / name).unlink()
        subject = res.subject_for_input(image)
        command = res.preview(subject, "thickness", self.tmp / "p.png")
        self.assertEqual(Path(command[3]).name, "rh.thickness.sub1")
        self.assertEqual(len(res.preview_sources(subject, "thickness")), 2)

    def test_the_viewer_imports_the_t1prep_of_its_caller(self):
        env = res.viewer_environment({"PYTHONPATH": "/elsewhere"})
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep),
                         [str(_SRC), "/elsewhere"])


class TestFindingImages(_Folders):
    def test_results_are_not_taken_for_inputs(self):
        self.cat12("a")
        self.touch(self.tmp / "data" / "sub" / "b_T1w.nii")
        self.touch(self.tmp / "data" / ".hidden" / "c.nii")
        self.touch(self.tmp / "data" / "notes.txt")
        here = [path.name for path in find_images(self.tmp / "data")]
        self.assertEqual(here, ["a.nii.gz"])
        below = [path.name for path in find_images(self.tmp / "data", recursive=True)]
        self.assertEqual(below, ["a.nii.gz", "b_T1w.nii"])
        matching = find_images(self.tmp / "data", "*_T1w.nii*", recursive=True)
        self.assertEqual([path.name for path in matching], ["b_T1w.nii"])


if __name__ == "__main__":
    unittest.main()
