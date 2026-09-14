"""The command-line behaviour every T1Prep tool shares.

CAT_VolView is the template: called without an argument a tool prints the
synopsis and exits non-zero, ``--help`` prints the full description, and only
the ``--`` spelling of an option is advertised.  The single-dash spellings the
tools used before still have to parse, or existing command lines break.
"""

import contextlib
import io
import re
import subprocess
import sys
import unittest
from pathlib import Path

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep.cli_help import ArgumentParser, hint, version  # noqa: E402


def run(callable_, argv):
    """Call *callable_* with *argv*, returning its output and exit status."""
    out, err = io.StringIO(), io.StringIO()
    status = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            callable_(argv)
        except SystemExit as exc:
            status = exc.code
    return out.getvalue() + err.getvalue(), status


def build_parser():
    """A parser with one option of every shape the tools use."""
    parser = ArgumentParser(prog="demo", description="A demo tool.")
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--subject", "-s", help="subject files")
    parser.add_argument("--quiet", "-q", action="store_true", help="be quiet")
    parser.add_argument("--overlay", "-overlay", "-ov", help="an overlay")
    return parser


class TestTheSharedParser(unittest.TestCase):
    def test_no_argument_prints_the_synopsis_and_fails(self):
        parser = build_parser()
        text, status = run(parser.exit_without_arguments, [])
        self.assertEqual(status, 1)
        self.assertIn("usage: demo", text)
        self.assertIn(hint("demo"), text)

    def test_an_argument_lets_the_tool_run(self):
        parser = build_parser()
        text, status = run(parser.exit_without_arguments, ["file.nii"])
        self.assertIsNone(status)
        self.assertEqual(text, "")

    def test_help_prints_the_full_description(self):
        text = build_parser().format_help()
        self.assertIn("A demo tool.", text)
        self.assertIn("--subject", text)

    def test_only_the_double_dash_spelling_is_advertised(self):
        text = build_parser().format_help()
        # Every option-looking token in the help has to carry two dashes; a
        # bare "-s" or "-overlay" would mean a legacy spelling leaked through
        single_dash = re.findall(r"(?<![\w-])-(?!-)[A-Za-z][\w-]*", text)
        self.assertEqual(single_dash, [], f"single-dash spellings advertised: {single_dash}")
        for shown in ("--subject", "--quiet", "--overlay", "--help"):
            self.assertIn(shown, text)

    def test_the_single_dash_spellings_still_parse(self):
        parser = build_parser()
        args = parser.parse_args(["-q", "-s", "subj", "-ov", "o.gii"])
        self.assertTrue(args.quiet)
        self.assertEqual(args.subject, "subj")
        self.assertEqual(args.overlay, "o.gii")

    def test_the_double_dash_spellings_mean_the_same(self):
        parser = build_parser()
        legacy = parser.parse_args(["-q", "-s", "subj", "-ov", "o.gii"])
        modern = parser.parse_args(
            ["--quiet", "--subject", "subj", "--overlay", "o.gii"]
        )
        self.assertEqual(legacy, modern)

    def test_a_parse_error_points_at_help(self):
        text, status = run(build_parser().parse_args, ["--nonesuch"])
        self.assertEqual(status, 2)
        self.assertIn("usage: demo", text)
        self.assertIn(hint("demo"), text)

    def test_version_answers_with_the_release(self):
        text, status = run(build_parser().parse_args, ["--version"])
        self.assertEqual(status, 0)
        self.assertIn(version(), text)

    def test_a_tool_can_opt_out_of_version(self):
        parser = ArgumentParser(prog="demo", add_version=False)
        self.assertNotIn("--version", parser.format_help())


class TestEveryToolFollowsIt(unittest.TestCase):
    """The tools that take a file all answer a bare call the same way."""

    #: (module, callable) pairs whose parser is reached without a display
    TOOLS = [
        ("t1prep.t1prep", "main"),
        ("t1prep.bbreg", "main"),
        ("t1prep.metrics", "_parse_dice_args"),
        ("t1prep.realign_longitudinal", "_parse_args"),
        ("t1prep.warp_longitudinal", "_parse_args"),
        ("t1prep.modulate_longitudinal", "_parse_args"),
    ]

    def test_a_bare_call_prints_the_synopsis(self):
        import importlib

        for module_name, entry in self.TOOLS:
            with self.subTest(tool=module_name):
                try:
                    module = importlib.import_module(module_name)
                except Exception as exc:  # pragma: no cover - optional deps
                    self.skipTest(f"{module_name} unavailable: {exc}")
                text, status = run(getattr(module, entry), [])
                self.assertEqual(status, 1)
                self.assertIn("usage:", text)
                self.assertIn("for the full description", text)


class TestTheBashToolsFollowIt(unittest.TestCase):
    """The same two answers from the bash side of the toolbox."""

    SCRIPTS = [
        "CAT_GrepJson",
        "CAT_VolSmooth_ui",
        "CAT_Surf2ROIMulti_ui",
        "CAT_SurfParameters_ui",
        "CAT_SurfResampleMulti_ui",
        "parallelize",
        "progress_bar_multi.sh",
    ]

    def run_script(self, name, *args):
        return subprocess.run(
            [str(_ROOT / "scripts" / name), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_a_bare_call_prints_the_synopsis(self):
        for name in self.SCRIPTS:
            with self.subTest(script=name):
                done = self.run_script(name)
                self.assertEqual(done.returncode, 1)
                output = done.stdout + done.stderr
                self.assertIn("USAGE:", output)
                self.assertIn(f"Run '{name} --help'", output)

    def test_version_answers_with_the_release(self):
        for name in self.SCRIPTS:
            with self.subTest(script=name):
                done = self.run_script(name, "--version")
                self.assertEqual(done.returncode, 0)
                self.assertEqual(done.stdout.strip(), f"{name} {version()}")


if __name__ == "__main__":
    unittest.main()
