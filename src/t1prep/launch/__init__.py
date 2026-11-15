"""What the launcher of T1Prep knows that is not Qt.

``t1prep.gui.launcher`` starts the bash orchestrator ``T1Prep``, follows what
it prints and shows what it left on disk.  The parts of that which do not
depend on the toolkit live here, where they are tested without a window:

* :mod:`~t1prep.launch.options` — the options a front end offers and the
  arguments they become,
* :mod:`~t1prep.launch.progress` — the phases, bars and per-subject states of
  a running job,
* :mod:`~t1prep.launch.results` — the files a subject's run leaves behind,
  the measures in its report and the viewer calls that show them.

Nothing in this package imports Qt or VTK.
"""
