"""The option table of the front ends agrees with scripts/T1Prep.

``t1prep.launch.options`` is a second description of T1Prep's options: which
flag switches what, and what each starts as.  The orchestrator is the first.
These tests hold the two against each other the direct way — the arguments
the table produces go through the ``parse_args`` of ``scripts/T1Prep``, and
the variables T1Prep ends up with have to be the values the form showed.  An
option that is renamed, removed or given another default in the orchestrator
fails here.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep.launch import options as opt  # noqa: E402

_T1PREP = _ROOT / "scripts" / "T1Prep"
_DEFAULTS = opt.load_defaults(_ROOT / "T1Prep_defaults.txt")

# The variable of scripts/T1Prep that holds an option after parse_args, where
# it is not the one the defaults file sets: parse_args turns save_hemi on for
# the surfaces and remembers the user's wish in keep_hemi
_AFTER_PARSING = {"hemi": "keep_hemi"}
_VARIABLES = sorted(
    {_AFTER_PARSING.get(item.key, item.variable) for item in opt.OPTIONS
     if item.variable}
    | {"skullstrip_only", "skip_skullstrip", "fmriprep"})

# Sourcing T1Prep defines its functions; main only runs when it is executed
_SHOW = (
    'source "$T1PREP" && parse_args "$@" && for v in '
    + " ".join(_VARIABLES)
    + '; do printf "%s=%s\\n" "$v" "${!v}"; done'
)


def parse(arguments):
    """What scripts/T1Prep makes of *arguments*: (process, {variable: value})."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "f.nii").write_bytes(b"")
        environment = dict(os.environ, T1PREP=str(_T1PREP))
        environment.pop("T1PREP_NN_BACKEND", None)
        done = subprocess.run(
            ["bash", "-c", _SHOW, "bash", *arguments, "f.nii"],
            capture_output=True, text=True, timeout=120, cwd=tmp,
            env=environment)
    values = dict(line.partition("=")[::2] for line in done.stdout.splitlines())
    return done, values


def as_t1prep_holds_it(item, value):
    """The text of *value* in the variable of *item* after parse_args."""
    if item.key == "gz":
        return "nii.gz" if value else "nii"
    if item.kind == "bool":
        return "1" if value else "0"
    if item.kind == "int":
        return str(item.auto if item.auto is not None and value == item.minimum
                   else value)
    if item.kind == "atlas":
        return opt.build_atlas_arg(list(value)) or ""
    return str(value)


def defaults():
    return opt.default_values(_DEFAULTS, environ={})


@unittest.skipUnless(shutil.which("bash"), "bash not available")
class TestTheTableAgainstTheOrchestrator(unittest.TestCase):
    def check(self, changes):
        """The form shows *changes*; T1Prep has to end up with the same."""
        shown = dict(defaults(), **changes)
        arguments = opt.build_arguments(shown, _DEFAULTS, environ={})
        done, held = parse(arguments)
        self.assertEqual(done.returncode, 0, done.stderr)
        expected = opt.effective(shown)
        unused = opt.irrelevant(shown)
        for item in opt.OPTIONS:
            if not item.variable or item.key in unused:
                continue
            variable = _AFTER_PARSING.get(item.key, item.variable)
            with self.subTest(option=item.key, arguments=arguments):
                value = expected[item.key]
                if item.kind == "atlas":
                    # The list T1Prep holds names the same atlases
                    self.assertEqual(
                        [name.lower() for name in opt.parse_atlas_string(held[variable])],
                        [name.lower() for name in value])
                else:
                    self.assertEqual(held[variable], as_t1prep_holds_it(item, value))
        return held, arguments

    def test_untouched_the_form_is_what_t1prep_does_anyway(self):
        self.check({})

    def test_every_switch_the_other_way(self):
        start = defaults()
        flipped = {
            item.key: not start[item.key] for item in opt.OPTIONS
            if item.kind == "bool" and item.variable
            and item.key not in ("seg", "surf", "fmriprep", "bids")
            and opt.can_change(item, start[item.key])
        }
        self.check(flipped)

    def test_every_number_and_choice_changed(self):
        self.check({"multi": 3, "min_memory": 17, "median_filter": 4,
                    "pre_fwhm": 5, "thickness_method": "2",
                    "atlas": ("SUIT",), "atlas_surf": ("aparc_HCP_MMP1",)})

    def test_one_job_at_a_time_and_auto(self):
        held, _ = self.check({"multi": 1})
        self.assertEqual(held["multi"], "1")
        held, arguments = self.check({"multi": 0})      # Auto
        self.assertEqual(held["multi"], "-1")
        self.assertNotIn("--multi", arguments)

    def test_no_surfaces(self):
        _, arguments = self.check({"surf": False, "pial_white": False})
        # What belongs to the surfaces is not spelled out without them
        self.assertIn("--no-surf", arguments)
        self.assertNotIn("--no-pial-white", arguments)
        self.assertNotIn("--atlas-surf", arguments)

    def test_surfaces_only(self):
        _, arguments = self.check({"seg": False, "p": True})
        self.assertIn("--no-seg", arguments)
        self.assertNotIn("--p", arguments)
        self.assertNotIn("--atlas", arguments)

    def test_bids_names_compress_the_volumes(self):
        held, arguments = self.check({"bids": True})
        self.assertEqual(held["nii_ext"], "nii.gz")
        self.assertNotIn("--gz", arguments)

    def test_fmriprep_sets_what_it_implies_by_itself(self):
        held, arguments = self.check({"fmriprep": True})
        self.assertEqual(held["fmriprep"], "1")
        for implied in ("--bids", "--gz", "--csf", "--p", "--no-mwp", "--no-atlas"):
            self.assertNotIn(implied, arguments)

    def test_skull_stripping_only_and_skipped(self):
        held, arguments = self.check({"skullstrip": "only"})
        self.assertEqual((held["skullstrip_only"], held["estimate_surf"]), ("1", "0"))
        self.assertNotIn("--no-surf", arguments)
        held, _ = self.check({"skullstrip": "skip"})
        self.assertEqual(held["skip_skullstrip"], "1")

    def test_no_atlas_at_all(self):
        held, arguments = self.check({"atlas": (), "atlas_surf": ()})
        self.assertIn("--no-atlas", arguments)
        self.assertIn("--no-atlas-surf", arguments)
        self.assertEqual((held["atlas_vol"], held["atlas_surf"]), ("", ""))

    def test_every_atlas_option_can_say_none(self):
        # An empty selection has to reach T1Prep as "none", not as "the default"
        for item in opt.OPTIONS:
            if item.kind == "atlas":
                with self.subTest(option=item.key):
                    self.assertTrue(item.off)


class TestTheTableAgainstTheDefaultsFile(unittest.TestCase):
    def test_every_default_comes_from_a_variable_of_the_file(self):
        for item in opt.OPTIONS:
            if item.variable:
                with self.subTest(option=item.key):
                    self.assertIn(item.variable, _DEFAULTS)

    def test_the_fallbacks_are_the_shipped_defaults(self):
        # A defaults file that names nothing: the form has to start the same
        self.assertEqual(opt.default_values({}, environ={}), defaults())

    def test_the_environment_overrides_the_backend_of_the_file(self):
        values = opt.default_values(_DEFAULTS, environ={opt.BACKEND_ENV: "tinygrad"})
        self.assertEqual(values["nn_backend"], "tinygrad")

    def test_a_defaults_file_of_one_s_own(self):
        values = opt.default_values(
            {"estimate_surf": "0", "nii_ext": "nii.gz", "multi": "6",
             "atlas_vol": "'suit'", "skullstrip_only": "1"}, environ={})
        self.assertEqual(
            (values["surf"], values["gz"], values["multi"], values["atlas"],
             values["skullstrip"]),
            (False, True, 6, ("SUIT",), "only"))


class TestTheFastPresetOfTheDefaultsFile(unittest.TestCase):
    """``fast=1`` in the defaults file is ``--fast`` for ``scripts/T1Prep``;
    the form shows what such a run does."""

    def test_the_form_starts_from_the_preset(self):
        values = opt.default_values(dict(_DEFAULTS, fast="1"), environ={})
        self.assertEqual(
            (values["pial_white"], values["sphere_reg"], values["mwp"],
             values["atlas"], values["atlas_surf"]),
            (False, False, False, (), ()))
        usual = defaults()
        self.assertTrue(usual["pial_white"] and usual["sphere_reg"] and usual["mwp"])
        self.assertTrue(usual["atlas"] and usual["atlas_surf"])

    def test_the_preset_is_the_one_of_the_script(self):
        # apply_fast_preset() in scripts/T1Prep, read from its source
        text = (_ROOT / "scripts" / "T1Prep").read_text()
        body = text[text.index("apply_fast_preset()\n{"):]
        body = body[body.index("{") + 1:body.index("\n}")]
        script = {}
        for line in body.splitlines():
            key, _, value = line.strip().partition("=")
            if key and key != "fast":
                script[key] = value.strip('"\'')
        self.assertEqual(script, opt.FAST_PRESET)


class TestWhereTheDefaultsFileIs(unittest.TestCase):
    """The form reads the defaults of the T1Prep it runs, as that T1Prep does.

    The web UI once read ``T1Prep_defaults.txt`` from the working directory
    and so started from fallbacks of its own; the file is looked up from the
    orchestrator instead.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def t1prep(self, folder, defaults=None, beside=False):
        """A T1Prep in tmp/*folder*, with *defaults* next to it or one up."""
        script = self.tmp / folder / "T1Prep"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/usr/bin/env bash\n")
        if defaults is not None:
            where = script.parent.parent if beside else script.parent
            (where / "T1Prep_defaults.txt").write_text(defaults)
        return str(script)

    def values(self, script):
        return opt.default_values(
            opt.load_defaults(opt.defaults_file(script)), environ={})

    def test_installed_next_to_t1prep_whatever_the_working_directory(self):
        script = self.t1prep("venv/bin", "save_pial_white=0\nmin_memory=7\n")
        decoy = self.tmp / "cwd"
        decoy.mkdir()
        (decoy / "T1Prep_defaults.txt").write_text("save_pial_white=1\nmin_memory=24\n")
        before = os.getcwd()
        os.chdir(decoy)
        self.addCleanup(os.chdir, before)
        values = self.values(script)
        self.assertEqual((values["pial_white"], values["min_memory"]), (False, 7))

    def test_a_source_checkout_keeps_it_at_the_root(self):
        script = self.t1prep("repo/scripts", "min_memory=12\n", beside=True)
        self.assertEqual(self.values(script)["min_memory"], 12)

    def test_without_a_file_the_form_starts_as_shipped(self):
        self.assertEqual(self.values(self.t1prep("empty/bin")), defaults())

    def test_quotes_and_comments_of_the_file(self):
        script = self.t1prep(
            "venv/bin", "nii_ext='nii.gz'   # compressed\n# multi=9\nmulti=\"4\"\n")
        values = self.values(script)
        self.assertEqual((values["gz"], values["multi"]), (True, 4))


class TestTheShippedAtlases(unittest.TestCase):
    def test_the_defaults_name_them_in_lower_case(self):
        # The file says neuromorphometrics, the atlas is Neuromorphometrics
        values = defaults()
        self.assertEqual(values["atlas"], ("CoBra", "Neuromorphometrics"))
        self.assertEqual(values["atlas_surf"],
                         ("aparc_DK40.freesurfer", "aparc_a2009s.freesurfer"))

    def test_each_has_a_description_for_its_tool_tip(self):
        for surface, names in ((False, opt.list_atlas_names()),
                               (True, opt.list_surface_atlas_names())):
            self.assertTrue(names)
            for name in names:
                with self.subTest(atlas=name):
                    self.assertTrue(opt.atlas_description(name, surface))

    def test_a_long_description_is_cut(self):
        text = opt.atlas_description("aparc_DK40.freesurfer", surface=True, max_lines=3)
        self.assertEqual(len(text.splitlines()), 4)
        self.assertTrue(text.endswith("…"))

    def test_only_shipped_atlases_are_read(self):
        # The name ends up in a file path
        for name in ("../../Names", "no_such_atlas", ""):
            self.assertEqual(opt.atlas_description(name), "")
            self.assertEqual(opt.atlas_description(name, surface=True), "")


class TestWhatTheFormMayDo(unittest.TestCase):
    def test_a_one_way_switch_cannot_leave_a_default_of_the_file(self):
        self.assertTrue(opt.can_change(opt.option("wp"), False))
        self.assertFalse(opt.can_change(opt.option("wp"), True))   # no --no-wp
        self.assertTrue(opt.can_change(opt.option("pial_white"), True))
        self.assertTrue(opt.can_change(opt.option("pial_white"), False))

    def test_one_of_the_two_steps_has_to_run(self):
        self.assertEqual(opt.problems(defaults()), [])
        self.assertTrue(opt.problems(dict(defaults(), seg=False, surf=False)))

    def test_what_belongs_to_a_step_is_irrelevant_without_it(self):
        unused = opt.irrelevant(dict(defaults(), surf=False))
        self.assertIn("pial_white", unused)
        self.assertIn("atlas_surf", unused)
        self.assertNotIn("mwp", unused)

    def test_every_option_sits_in_a_section_and_has_a_tool_tip(self):
        sections = {key for key, _title, _open in opt.GROUPS}
        for item in opt.OPTIONS:
            with self.subTest(option=item.key):
                self.assertIn(item.group, sections)
                self.assertTrue(item.help)
        keys = [item.key for item in opt.OPTIONS]
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
