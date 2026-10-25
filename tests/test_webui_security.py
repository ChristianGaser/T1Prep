"""The web UI only takes local, same-origin requests and passes no options.

Up to 0.8.0 any web page in the user's browser could POST /submit (no
token, any Host header), form values starting with ``-`` became T1Prep
options (``--atlas-surf=--defaults`` sourced an uploaded file), the
"Pial + white surfaces" box sent ``--pial-white``, which T1Prep read as
``--p``, ``/atlas-help`` read any ``.txt`` through ``../`` and the folder
pattern globbed outside the upload sandbox.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from t1prep.webui import app as webui
except ImportError:  # pragma: no cover - flask is a dependency
    webui = None


@unittest.skipIf(webui is None, "web UI not importable")
class TestOnlyLocalSameOriginRequests(unittest.TestCase):
    def setUp(self):
        self.client = webui.app.test_client()

    def test_a_foreign_host_header_is_refused(self):
        self.assertEqual(
            self.client.get("/", headers={"Host": "rebind.example:5050"}).status_code,
            403,
        )
        self.assertEqual(
            self.client.get("/", headers={"Host": "127.0.0.1:5050"}).status_code, 200
        )

    def test_the_form_carries_the_token(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn(webui.CSRF_TOKEN, page)

    def test_a_post_without_the_token_is_refused(self):
        self.assertEqual(self.client.post("/submit", data={}).status_code, 403)
        self.assertEqual(
            self.client.post("/submit", data={"csrf_token": "guess"}).status_code, 403
        )


@unittest.skipIf(webui is None, "web UI not importable")
class TestTheCommandLine(unittest.TestCase):
    def cmd(self, **form):
        return webui.build_command(dict(save_mwp="1", **form), [Path("x.nii")])

    def test_pial_white_is_explicit_both_ways(self):
        self.assertIn("--no-pial-white", self.cmd())
        on = self.cmd(**{"--pial-white": "1"})
        self.assertIn("--pial-white", on)
        self.assertNotIn("--no-pial-white", on)

    def test_no_vessel_is_forwarded(self):
        self.assertIn("--no-vessel", self.cmd(**{"--no-vessel": "1"}))

    def test_a_value_may_not_be_an_option(self):
        with self.assertRaises(ValueError):
            self.cmd(**{"--no-overwrite": "--defaults"})
        self.assertIn("-1", self.cmd(**{"--multi": "-1"}))


@unittest.skipIf(webui is None, "web UI not importable")
class TestNoReadsOutsideTheSandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_atlas_help_only_names_shipped_surface_atlases(self):
        (self.tmp / "secret.txt").write_text("secret\n")
        name = "../" * 12 + str(self.tmp / "secret").lstrip("/")
        response = webui.app.test_client().get(
            "/atlas-help", query_string={"kind": "surface", "name": name}
        )
        self.assertEqual(response.status_code, 404)

    def test_the_folder_pattern_stays_below_the_upload_root(self):
        uploads = self.tmp / "uploads"
        (uploads / "study").mkdir(parents=True)
        (uploads / "study" / "a.nii").write_bytes(b"")
        (self.tmp / "outside.nii").write_bytes(b"")
        with mock.patch.object(webui, "DEFAULT_UPLOAD_ROOT", uploads):
            found = webui.resolve_folder_pattern_inputs(
                {"folder_root": "study", "file_pattern": "*.nii"}
            )
            escaped = webui.resolve_folder_pattern_inputs(
                {"folder_root": "study", "file_pattern": "../../*.nii"}
            )
        self.assertEqual([p.name for p in found], ["a.nii"])
        self.assertEqual(escaped, [])


if __name__ == "__main__":
    unittest.main()
