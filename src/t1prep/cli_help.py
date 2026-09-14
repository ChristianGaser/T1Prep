"""Shared command-line behaviour for the T1Prep tools.

Every tool answers the same way, with ``CAT_VolView`` as the template:

* called without an argument it prints the short synopsis — the overview of
  the possible options — and exits non-zero,
* ``--help`` prints the full description,
* every option is advertised with two dashes.

Single-dash spellings the tools used before (``-overlay``, ``-s``, ``-h``)
remain valid so existing command lines keep working, but they are hidden from
the synopsis and the help, so only the ``--`` form is ever advertised.  That
hiding is what :class:`ArgumentParser` adds to :mod:`argparse`; the bash tools
get the same behaviour from ``print_usage`` in ``T1Prep_utils.sh``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import Iterator, List, Optional, Sequence

__all__ = [
    "ArgumentParser",
    "HelpFormatter",
    "RawTextHelpFormatter",
    "hint",
    "version",
    "add_hidden_aliases",
]


def hint(prog: str) -> str:
    """Return the line pointing from the synopsis to the full help."""
    return f"Run '{prog} --help' for the full description."


def version() -> str:
    """Return the T1Prep version, the answer to ``--version``.

    Read from the installed distribution's metadata, and from
    ``t1prep/__init__.py`` in a source checkout — never by importing the
    package, which would pull torch in just to print a version string.
    """
    import importlib.metadata

    try:
        return importlib.metadata.version("T1Prep")
    except importlib.metadata.PackageNotFoundError:
        pass
    init = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__init__.py")
    try:
        with open(init, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return "unknown"


class RawTextHelpFormatter(argparse.RawTextHelpFormatter):
    """Print descriptions and help strings exactly as they are written.

    For the tools whose help strings are hand-wrapped over several lines and
    already name their own defaults, so that appending them again would only
    repeat what the line says.
    """


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep the description as written and the default values visible.

    :class:`argparse.RawDescriptionHelpFormatter` and
    :class:`argparse.ArgumentDefaultsHelpFormatter` are both wanted — the
    former for the hand-laid-out descriptions and examples, the latter so a
    default never has to be repeated in the help string.
    """

    def _get_help_string(self, action: argparse.Action) -> Optional[str]:
        text = action.help
        if text is None or "%(default)" in text:
            return text
        default = action.default
        # Nothing to say for a flag that is off, an option without a default,
        # or an empty list waiting to be appended to
        if default in (None, False, argparse.SUPPRESS) or default == []:
            return text
        if action.option_strings or action.nargs in (
            argparse.OPTIONAL,
            argparse.ZERO_OR_MORE,
        ):
            if not isinstance(action.const, bool):
                return text + " (default: %(default)s)"
        return text


class ArgumentParser(argparse.ArgumentParser):
    """An :class:`argparse.ArgumentParser` that advertises only ``--`` options.

    Behaviour shared by every T1Prep tool:

    * :meth:`exit_without_arguments` prints the synopsis when the tool is
      called with nothing at all, instead of failing on a missing argument,
    * a parse error prints the synopsis and points at ``--help``,
    * the synopsis and the help list only the ``--`` spelling of an option;
      the single-dash aliases still parse.
    """

    def __init__(self, *args, add_version: bool = True, **kwargs):
        kwargs.setdefault("formatter_class", HelpFormatter)
        super().__init__(*args, **kwargs)
        if add_version:
            # Every tool answers --version with the T1Prep release it belongs
            # to; '-v'/'-V' are the spellings the bash tools took before
            self.add_argument(
                "--version", "-v", "-V",
                action="version",
                version=f"%(prog)s {version()}",
                help="show the T1Prep version and exit",
            )

    # -- only the "--" spelling is ever shown -------------------------- #

    @contextlib.contextmanager
    def _double_dash_only(self) -> Iterator[None]:
        """Hide the single-dash aliases of every option while rendering.

        argparse takes both the synopsis and the option list straight from
        ``action.option_strings``, and it has no per-spelling way of hiding
        one, so the legacy spellings are dropped for the length of the
        rendering and put back afterwards.
        """
        saved: List[tuple] = []
        for action in self._actions:
            shown = [s for s in action.option_strings if s.startswith("--")]
            if shown and shown != action.option_strings:
                saved.append((action, action.option_strings))
                action.option_strings = shown
        try:
            yield
        finally:
            for action, option_strings in saved:
                action.option_strings = option_strings

    def format_usage(self) -> str:
        with self._double_dash_only():
            return super().format_usage()

    def format_help(self) -> str:
        with self._double_dash_only():
            return super().format_help()

    # -- the two unified entry points ---------------------------------- #

    def print_synopsis(self, file=None) -> None:
        """Print the synopsis and the pointer to ``--help``."""
        stream = sys.stderr if file is None else file
        self.print_usage(stream)
        print(hint(self.prog), file=stream)

    def exit_without_arguments(self, argv: Sequence[str]) -> None:
        """Print the synopsis and exit 1 when *argv* is empty.

        The tools take a file to work on, so an empty command line is not an
        error the user made but a request to be told what the tool takes.
        """
        if not argv:
            self.print_synopsis()
            self.exit(1)

    def error(self, message: str):  # noqa: D102 - argparse's own contract
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n{hint(self.prog)}\n")


def add_hidden_aliases(parser: argparse.ArgumentParser, *names: str) -> None:
    """Accept *names* as spellings of an option without advertising them.

    Used for the handful of legacy spellings argparse cannot carry as an
    alias of an existing action, e.g. ``-help`` next to ``--help``: a
    single-dash ``-help`` would otherwise be read as ``-h elp``.
    """
    for name in names:
        parser.add_argument(name, action="help", help=argparse.SUPPRESS)
