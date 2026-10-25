"""Telemetry: one opt-out for every entry point, and crash reports without
personal data.

Unless the user opts out, T1Prep reports two things to its developers
(Sentry, EU region):

* ``scripts/T1Prep`` sends one event per call (with ``--multi``, one per file
  and processing step) with the T1Prep version, the operating system and CPU
  architecture, and the number of files that succeeded and failed
  (``send_sentry`` in ``scripts/T1Prep_utils.sh``).
* ``t1prep.segment`` reports an unhandled exception, reduced to its type and
  the module, function and line of each stack frame.

Exception messages, file names and paths, local variables, source lines, the
command line and the host name are dropped before anything leaves the
machine.  ``sentry_sdk`` sends all of them by default, and in neuroimaging
they carry subject identifiers: a ``FileNotFoundError`` message is nothing
but the path of the input.

``T1PREP_NO_TRACK=1`` or ``DO_NOT_TRACK=1`` in the environment switches both
off.  ``--no-track`` (``scripts/T1Prep``, ``t1prep-run``), ``track=0`` in the
defaults file and ``run_t1prep(track=False)`` set ``T1PREP_NO_TRACK`` for the
processes they start, so the segmentation follows the orchestrator's choice.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional

__all__ = ["NO_TRACK_ENV", "init_crash_reporting", "telemetry_enabled"]

NO_TRACK_ENV = "T1PREP_NO_TRACK"

SENTRY_DSN = (
    "https://ca6089ed5dc4326c6c69afc3684c5fc1@o4511309449068544"
    ".ingest.de.sentry.io/4511309454311504"
)

# Values of T1PREP_NO_TRACK and DO_NOT_TRACK that leave telemetry on
_NOT_SET = ("", "0", "false", "no", "off")

# What survives of an event, an exception, its mechanism and a stack frame;
# everything else is dropped (see _scrub_event)
_EVENT_KEYS = (
    "event_id", "timestamp", "level", "platform", "release", "environment",
    "sdk",
)
_EXCEPTION_KEYS = ("type", "module")
_MECHANISM_KEYS = (
    "type", "handled", "source", "exception_id", "parent_id",
    "is_exception_group",
)
_FRAME_KEYS = ("module", "function", "lineno", "in_app")


def telemetry_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return ``False`` when ``T1PREP_NO_TRACK`` or ``DO_NOT_TRACK`` opts out.

    Mirrors ``telemetry_allowed`` in ``scripts/T1Prep_utils.sh``.
    """
    env = os.environ if environ is None else environ
    return all(
        env.get(name, "").strip().lower() in _NOT_SET
        for name in (NO_TRACK_ENV, "DO_NOT_TRACK")
    )


def _pick(mapping: Mapping[str, Any], keys: Iterable[str]) -> dict:
    return {key: mapping[key] for key in keys if key in mapping}


def _scrub_event(event: dict, hint: Any = None) -> Optional[dict]:
    """Reduce a Sentry event to its exception types and where they were raised.

    An allow-list, not a deny-list: only the keys named above are copied, so
    anything a future ``sentry_sdk`` adds is dropped too.  Events without an
    exception (log records, messages) are not sent at all.
    """
    values = []
    for exc in (event.get("exception") or {}).get("values") or []:
        kept = _pick(exc, _EXCEPTION_KEYS)
        if exc.get("mechanism"):
            kept["mechanism"] = _pick(exc["mechanism"], _MECHANISM_KEYS)
        frames = (exc.get("stacktrace") or {}).get("frames") or []
        kept["stacktrace"] = {"frames": [_pick(f, _FRAME_KEYS) for f in frames]}
        values.append(kept)
    if not values:
        return None

    scrubbed = _pick(event, _EVENT_KEYS)
    runtime = (event.get("contexts") or {}).get("runtime")
    if runtime:
        scrubbed["contexts"] = {"runtime": _pick(runtime, ("name", "version"))}
    scrubbed["exception"] = {"values": values}
    return scrubbed


def init_crash_reporting() -> bool:
    """Report an unhandled exception of this process to the T1Prep developers.

    Does nothing when telemetry is switched off (:func:`telemetry_enabled`)
    or ``sentry_sdk`` is missing, and a failure here never stops the
    pipeline.  Returns whether reporting is active.
    """
    if not telemetry_enabled():
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.atexit import AtexitIntegration
        from sentry_sdk.integrations.excepthook import ExcepthookIntegration

        from . import __version__

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            # Without it sentry_sdk runs 'git rev-parse HEAD' in the current
            # directory and reports the commit of whatever repository that is
            release=__version__,
            # Only the hooks a crash report needs: the defaults add sys.argv,
            # log records and the command lines of subprocesses
            default_integrations=False,
            auto_enabling_integrations=False,
            integrations=[ExcepthookIntegration(), AtexitIntegration()],
            send_default_pii=False,
            include_local_variables=False,
            include_source_context=False,
            max_breadcrumbs=0,
            server_name="",
            before_send=_scrub_event,
        )
    except Exception:
        return False
    return True
