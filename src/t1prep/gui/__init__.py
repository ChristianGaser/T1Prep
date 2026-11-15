"""t1prep.gui

The graphical tools of T1Prep.

* The viewers, Qt and VTK: ``cat_surf_view`` (``CAT_SurfView``) and
  ``cat_vol_view`` (``CAT_VolView``).
* The launcher, Qt only: ``launcher`` (``t1prep-gui``) runs the pipeline and
  shows its results.  It starts the viewers as processes of their own and
  imports no VTK itself.
"""

__all__ = [
    "cat_surf_view",
]
