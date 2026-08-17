import os
import platform
import plistlib
import sys
import tempfile
import unittest
from pathlib import Path

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep.gui import make_apps


class TestBundleContents(unittest.TestCase):
    """The bundles must launch the right binary and declare the file types."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.bin_dir = self.tmp / "bin"
        self.bin_dir.mkdir()
        for name in make_apps.VIEWERS:
            (self.bin_dir / name).write_text("#!/bin/sh\n")
            (self.bin_dir / name).chmod(0o755)
        self.out_dir = self.tmp / "Applications"

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self):
        return make_apps.build_apps(out_dir=self.out_dir, bin_dir=self.bin_dir,
                                    quiet=True)

    def _info(self, name):
        with open(self.out_dir / f"{name}.app" / "Contents" / "Info.plist", "rb") as fh:
            return plistlib.load(fh)

    def test_both_bundles_are_built(self):
        apps = self._build()
        self.assertEqual({a.name for a in apps},
                         {"CAT_SurfView.app", "CAT_VolView.app"})
        for app in apps:
            self.assertTrue((app / "Contents" / "Info.plist").exists())

    def test_launcher_starts_the_viewer_of_this_installation(self):
        self._build()
        launcher = self.out_dir / "CAT_VolView.app" / "Contents" / "MacOS" / "CAT_VolView"
        self.assertTrue(os.access(launcher, os.X_OK))
        text = launcher.read_text()
        self.assertIn("Resources/t1prep_launch.py", text)
        self.assertIn("T1PREP_APP=1", text)          # so a double-click asks for a file
        self.assertIn("Library/Logs/T1Prep", text)   # failures stay visible
        # exec, so the viewer keeps the process Launch Services started: the
        # documents to open arrive as Apple events addressed to exactly that one
        self.assertIn("exec ", text)

    def test_launch_script_reports_a_failed_start(self):
        """Finder shows no output, so the reason has to reach a dialog."""
        self._build()
        script = (self.out_dir / "CAT_VolView.app" / "Contents" / "Resources"
                  / "t1prep_launch.py")
        text = script.read_text()
        compile(text, str(script), "exec")            # it has to be valid Python
        self.assertIn("t1prep.gui.cat_vol_view", text)
        self.assertIn("osascript", text)
        self.assertIn("traceback", text)

    def test_the_architectures_are_named(self):
        """Without them a script bundle is started under Rosetta.

        Launch Services cannot read the architectures of a script and then picks
        x86_64 on Apple silicon, where the interpreter loads its translated
        slice and every arm64 extension module fails in dlopen — the viewers
        died before their first window while the same command worked in a shell.
        """
        self._build()
        for name in make_apps.VIEWERS:
            architectures = self._info(name)["LSArchitecturePriority"]
            self.assertTrue(architectures)
            if platform.machine() in ("arm64", "x86_64"):
                self.assertEqual(architectures[0], platform.machine())

    def test_volume_viewer_claims_nifti_and_offers_gzip(self):
        self._build()
        info = self._info("CAT_VolView")
        by_type = {d["CFBundleTypeName"]: d for d in info["CFBundleDocumentTypes"]}
        self.assertIn("gov.nih.nifti-1", by_type["NIfTI volume"]["LSItemContentTypes"])
        self.assertEqual(by_type["NIfTI volume"]["LSHandlerRank"], "Default")
        # .nii.gz is gzip to macOS: offered, but not claimed as the owner
        self.assertEqual(by_type["Compressed volume"]["LSHandlerRank"], "Alternate")
        self.assertIn("org.gnu.gnu-zip-archive",
                      by_type["Compressed volume"]["LSItemContentTypes"])

    def test_surface_viewer_declares_its_own_types(self):
        """Nothing on macOS declares .gii, so the app has to export it."""
        self._build()
        info = self._info("CAT_SurfView")
        exported = {d["UTTypeIdentifier"] for d in info["UTExportedTypeDeclarations"]}
        self.assertIn(f"{make_apps.BUNDLE_PREFIX}.gifti", exported)
        extensions = [d["UTTypeTagSpecification"]["public.filename-extension"]
                      for d in info["UTExportedTypeDeclarations"]]
        self.assertIn(["gii"], extensions)

    def test_missing_console_scripts_are_reported(self):
        with self.assertRaises(FileNotFoundError):
            make_apps.build_apps(out_dir=self.out_dir, bin_dir=self.tmp / "empty",
                                 quiet=True)


class TestFirstRun(unittest.TestCase):
    """The viewers create the bundles once, and never get in the way."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.bin_dir = self.tmp / "bin"
        self.bin_dir.mkdir()
        for name in make_apps.VIEWERS:
            (self.bin_dir / name).write_text("#!/bin/sh\n")
        self.out_dir = self.tmp / "Applications"

        self._saved = (make_apps.default_output_dir, make_apps.find_bin_dir,
                       sys.platform)
        make_apps.default_output_dir = lambda: self.out_dir
        make_apps.find_bin_dir = lambda: self.bin_dir
        self._env = {k: os.environ.get(k) for k in (make_apps.DISABLE_ENV, "T1PREP_APP")}
        for key in self._env:
            os.environ.pop(key, None)

    def tearDown(self):
        make_apps.default_output_dir, make_apps.find_bin_dir, _ = self._saved
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    @unittest.skipUnless(sys.platform == "darwin", "macOS only")
    def test_created_on_first_start(self):
        self.assertTrue(make_apps.ensure_apps_exist(quiet=True))
        self.assertTrue((self.out_dir / "CAT_VolView.app").exists())

    @unittest.skipUnless(sys.platform == "darwin", "macOS only")
    def test_not_built_again_when_present(self):
        make_apps.ensure_apps_exist(quiet=True)
        self.assertEqual(make_apps.ensure_apps_exist(quiet=True), [])

    @unittest.skipUnless(sys.platform == "darwin", "macOS only")
    def test_can_be_switched_off(self):
        os.environ[make_apps.DISABLE_ENV] = "1"
        self.assertEqual(make_apps.ensure_apps_exist(quiet=True), [])
        self.assertFalse(self.out_dir.exists())

    @unittest.skipUnless(sys.platform == "darwin", "macOS only")
    def test_skipped_when_started_from_a_bundle(self):
        os.environ["T1PREP_APP"] = "1"
        self.assertEqual(make_apps.ensure_apps_exist(quiet=True), [])

    def test_failure_is_never_fatal(self):
        """A viewer must start even if the bundles cannot be written."""
        make_apps.find_bin_dir = lambda: None
        make_apps.default_output_dir = lambda: Path("/does/not/exist/at/all")
        self.assertEqual(make_apps.ensure_apps_exist(quiet=True), [])


if __name__ == "__main__":
    unittest.main()
