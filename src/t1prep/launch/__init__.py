"""What the front ends of T1Prep share.

``t1prep.gui.launcher`` (Qt) and ``t1prep.webui.app`` (Flask) both start the
bash orchestrator ``T1Prep``, follow what it prints and show what it left on
disk.  The parts of that which do not depend on the toolkit live here, so that
there is one copy of each:

* :mod:`~t1prep.launch.options` — the options a front end offers and the
  arguments they become,
* :mod:`~t1prep.launch.progress` — the phases, bars and per-subject states of
  a running job,
* :mod:`~t1prep.launch.results` — the files a subject's run leaves behind,
  the measures in its report and the viewer calls that show them.

Nothing in this package imports Qt, Flask or VTK.
"""
