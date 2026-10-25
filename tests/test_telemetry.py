"""Telemetry has one opt-out for every entry point and sends no personal data.

Up to 0.8.0 ``segment.py`` called ``sentry_sdk.init`` unconditionally:
``--no-track`` only silenced the bash summary event, the Python entry points
had no opt-out at all, and with the ``sentry_sdk`` defaults a crash uploaded
the input path (in the exception message and in every frame's local
variables), ``sys.argv`` and the host name.  Neuroimaging paths carry
subject identifiers.  These tests pin what replaced it (``_telemetry``).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep import _telemetry  # noqa: E402

try:
    import sentry_sdk
except ImportError:  # pragma: no cover - sentry-sdk is a dependency
    sentry_sdk = None

# A subject identifier as it shows up in real input paths
SUBJECT = "Mueller_Hans_1957-03-12"
SUBJECT_PATH = f"/data/clinic/{SUBJECT}/anat/sub-{SUBJECT}_T1w.nii.gz"


def _without_opt_out():
    """os.environ without T1PREP_NO_TRACK and DO_NOT_TRACK."""
    return {
        k: v for k, v in os.environ.items()
        if k not in (_telemetry.NO_TRACK_ENV, "DO_NOT_TRACK")
    }


class TestTheOptOut(unittest.TestCase):
    def test_telemetry_is_on_unless_a_variable_says_otherwise(self):
        self.assertTrue(_telemetry.telemetry_enabled({}))

    def test_either_variable_switches_it_off(self):
        for name in (_telemetry.NO_TRACK_ENV, "DO_NOT_TRACK"):
            for value in ("1", "true", "YES", "on"):
                with self.subTest(name=name, value=value):
                    self.assertFalse(_telemetry.telemetry_enabled({name: value}))

    def test_zero_false_no_off_or_empty_leave_it_on(self):
        for value in ("", "0", "false", "No", "off", " 0 "):
            with self.subTest(value=value):
                self.assertTrue(
                    _telemetry.telemetry_enabled({"DO_NOT_TRACK": value})
                )

    def test_switched_off_sentry_is_not_even_initialised(self):
        fake = types.ModuleType("sentry_sdk")
        fake.init = mock.Mock()
        with mock.patch.dict(sys.modules, {"sentry_sdk": fake}), \
                mock.patch.dict(os.environ, {_telemetry.NO_TRACK_ENV: "1"}):
            self.assertFalse(_telemetry.init_crash_reporting())
        fake.init.assert_not_called()


class TestTheScrubber(unittest.TestCase):
    """``_scrub_event`` keeps an allow-list; everything else is dropped."""

    EVENT = {
        "event_id": "0123456789abcdef0123456789abcdef",
        "timestamp": "2026-10-25T12:00:00Z",
        "level": "error",
        "platform": "python",
        "release": "0.8.0",
        "server_name": "lab-workstation-7",
        "extra": {"sys.argv": ["segment.py", "--input", SUBJECT_PATH]},
        "breadcrumbs": {"values": [{"message": f"loading {SUBJECT_PATH}"}]},
        "user": {"username": "gaser"},
        "contexts": {
            "runtime": {"name": "CPython", "version": "3.12.4",
                        "build": "3.12.4 (main) [Clang]"},
            "os": {"name": "Darwin", "kernel_version": "lab-workstation-7"},
        },
        "exception": {"values": [{
            "type": "FileNotFoundError",
            "value": f"[Errno 2] No such file or directory: '{SUBJECT_PATH}'",
            "module": None,
            "mechanism": {"type": "excepthook", "handled": False,
                          "meta": {"errno": {"number": 2}}},
            "stacktrace": {"frames": [{
                "filename": "nibabel/loadsave.py",
                "abs_path": "/Users/gaser/venv/nibabel/loadsave.py",
                "module": "nibabel.loadsave",
                "function": "load",
                "lineno": 101,
                "in_app": False,
                "pre_context": ["try:"],
                "context_line": "stat_result = os.stat(filename)",
                "post_context": ["except OSError:"],
                "vars": {"filename": SUBJECT_PATH},
            }]},
        }]},
    }

    def test_only_exception_types_and_their_locations_survive(self):
        scrubbed = _telemetry._scrub_event(json.loads(json.dumps(self.EVENT)))
        text = json.dumps(scrubbed)
        for secret in (SUBJECT, "lab-workstation-7", "gaser", "sys.argv",
                       "loadsave.py", "os.stat", "Clang"):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, text)

        exc = scrubbed["exception"]["values"][0]
        self.assertEqual(exc["type"], "FileNotFoundError")
        self.assertNotIn("value", exc)
        self.assertEqual(exc["mechanism"], {"type": "excepthook", "handled": False})
        self.assertEqual(exc["stacktrace"]["frames"], [{
            "module": "nibabel.loadsave", "function": "load",
            "lineno": 101, "in_app": False,
        }])
        self.assertEqual(scrubbed["release"], "0.8.0")
        self.assertEqual(scrubbed["contexts"],
                         {"runtime": {"name": "CPython", "version": "3.12.4"}})

    def test_an_event_without_an_exception_is_not_sent(self):
        self.assertIsNone(_telemetry._scrub_event(
            {"level": "error", "message": f"cannot read {SUBJECT_PATH}"}
        ))


# Runs in a subprocess: sentry_sdk installs process-wide hooks, and the
# unhandled exception goes through the real sys.excepthook as in segment.py.
# The network is blocked and the transport only records what would be sent.
_CRASH = textwrap.dedent("""
    import atexit, json, os, socket, sys

    def _blocked(*args, **kwargs):
        raise OSError("network blocked by the test")

    socket.socket.connect = _blocked
    socket.create_connection = _blocked

    import sentry_sdk
    from sentry_sdk.transport import Transport

    sent = []

    class Record(Transport):
        def capture_envelope(self, envelope):
            sent.append({
                "text": envelope.serialize().decode("utf-8", "replace"),
                "events": [item.payload.json for item in envelope.items
                           if item.type == "event"],
            })

        def flush(self, *args, **kwargs):
            pass

        def kill(self):
            pass

    real_init = sentry_sdk.init

    def init(*args, **kwargs):
        kwargs.update(dsn="https://public@127.0.0.1:9/1", transport=Record)
        return real_init(*args, **kwargs)

    sentry_sdk.init = init

    @atexit.register
    def dump():
        with open(sys.argv[1], "w") as f:
            json.dump(sent, f)

    from t1prep._telemetry import init_crash_reporting

    print("active" if init_crash_reporting() else "inactive", flush=True)

    def load(filename):
        raise FileNotFoundError(2, "No such file or directory", filename)

    load(os.environ["SUBJECT_PATH"])
""")


@unittest.skipIf(sentry_sdk is None, "sentry-sdk not installed")
class TestACrashReport(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # A stand-in package: only _telemetry is needed, not torch & co.
        package = self.tmp / "pkg" / "t1prep"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('__version__ = "9.9.9"\n')
        shutil.copy(_SRC / "t1prep" / "_telemetry.py", package)
        (self.tmp / "run").mkdir()
        self.script = self.tmp / "run" / "crash.py"
        self.script.write_text(_CRASH)

    def crash(self, **env):
        out = self.tmp / "sent.json"
        run_env = _without_opt_out()
        run_env.update(PYTHONPATH=str(self.tmp / "pkg"),
                       SUBJECT_PATH=SUBJECT_PATH, **env)
        done = subprocess.run(
            [sys.executable, str(self.script), str(out)],
            capture_output=True, text=True, timeout=120, env=run_env,
            cwd=self.tmp,
        )
        self.assertEqual(done.returncode, 1, done.stderr)
        return done, json.loads(out.read_text())

    def test_it_names_the_exception_and_where_but_not_the_subject(self):
        done, sent = self.crash()
        self.assertEqual(done.stdout.strip(), "active")
        self.assertEqual(len(sent), 1)
        text = sent[0]["text"]
        self.assertNotIn(SUBJECT, text)
        self.assertNotIn(str(self.tmp), text)

        (event,) = sent[0]["events"]
        for key in ("server_name", "extra", "breadcrumbs", "user", "modules"):
            self.assertNotIn(key, event)
        self.assertEqual(event["release"], "9.9.9")
        exc = event["exception"]["values"][-1]
        self.assertEqual(exc["type"], "FileNotFoundError")
        self.assertNotIn("value", exc)
        frames = exc["stacktrace"]["frames"]
        self.assertEqual(frames[-1]["function"], "load")
        for frame in frames:
            self.assertLessEqual(
                set(frame), {"module", "function", "lineno", "in_app"}
            )

    def test_the_opt_out_sends_nothing(self):
        done, sent = self.crash(DO_NOT_TRACK="1")
        self.assertEqual(done.stdout.strip(), "inactive")
        self.assertEqual(sent, [])


class TestTheEntryPointsPassTheChoiceOn(unittest.TestCase):
    def test_t1prep_run_has_no_track(self):
        from t1prep.t1prep import _build_parser

        parser = _build_parser()
        self.assertTrue(parser.parse_args(["x.nii"]).track)
        self.assertFalse(parser.parse_args(["x.nii", "--no-track"]).track)

    def test_the_segmentation_subprocess_gets_the_choice(self):
        from t1prep.t1prep import _segment_env

        with mock.patch.dict(os.environ, _without_opt_out(), clear=True):
            self.assertEqual(
                _segment_env(track=False)[_telemetry.NO_TRACK_ENV], "1"
            )
            self.assertNotIn(_telemetry.NO_TRACK_ENV, _segment_env(track=True))

    def test_the_web_ui_forwards_the_switch(self):
        try:
            from t1prep.webui.app import build_command
        except ImportError as exc:  # pragma: no cover - flask is a dependency
            self.skipTest(f"web UI not importable: {exc}")

        form = {"save_mwp": "1"}
        self.assertNotIn("--no-track", build_command(form, [Path("x.nii")]))
        form["--no-track"] = "1"
        self.assertIn("--no-track", build_command(form, [Path("x.nii")]))

    def test_the_bash_side_reads_the_same_variables(self):
        utils = _ROOT / "scripts" / "T1Prep_utils.sh"
        snippet = f'source "{utils}"; telemetry_allowed && echo on || echo off'
        cases = [({}, "on"), ({"T1PREP_NO_TRACK": "1"}, "off"),
                 ({"DO_NOT_TRACK": "true"}, "off"), ({"DO_NOT_TRACK": "0"}, "on")]
        for extra, expected in cases:
            with self.subTest(env=extra):
                env = _without_opt_out()
                env.update(extra)
                done = subprocess.run(
                    ["bash", "-c", snippet],
                    capture_output=True, text=True, timeout=120, env=env,
                )
                self.assertEqual(done.stdout.strip(), expected, done.stderr)

    def test_the_orchestrator_exports_the_choice(self):
        # The Python crash report only sees what T1Prep exports
        text = (_ROOT / "scripts" / "T1Prep").read_text()
        self.assertIn("! telemetry_allowed; then track=0; fi", text)
        self.assertIn("export T1PREP_NO_TRACK=1", text)


if __name__ == "__main__":
    unittest.main()
