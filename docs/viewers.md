# Viewers: CAT_SurfView and CAT_VolView

Two viewers ship with T1Prep and are installed as ordinary commands
(`pip install T1Prep` puts them in the environment's `bin/`):

| Command | Shows |
|---------|-------|
| `CAT_SurfView` | Cortical surfaces with overlays, atlases and a cluster table |
| `CAT_VolView` | Volumes as three orthogonal slices, with overlays and montages |

Both are built on PySide6 and VTK, both render without a display for batch use,
and both keep their display settings in a right-click menu. Neither needs
T1Prep output — they open any GIFTI surface or NIfTI volume.

```bash
CAT_SurfView lh.central.gii            # no arguments prints the help
CAT_VolView T1.nii.gz                  # three orthogonal slices
CAT_VolView T1.nii.gz p1T1.nii.gz      # up to 6 volumes, one window each, linked
```

- [Surface viewer](#surface-viewer)
- [Volume viewer](#volume-viewer)
- [Figures from a script](#figures-from-a-script)
- [macOS application bundles](#macos-application-bundles)
- [Keyboard reference](#keyboard-reference)

Back to the [README](../README.md).

---

## Surface viewer

`CAT_SurfView` shows both hemispheres as a six-view montage — lateral and
medial for each, dorsal in the middle — with an overlay on a shaded surface.
The second hemisphere is found from the file name (`lh.`↔`rh.`,
`left`↔`right`, `_hemi-L_`↔`_hemi-R_`), or split off a combined
`mesh.central.*` surface.

```bash
CAT_SurfView lh.thickness.sub-01                       # overlay; the mesh is found
CAT_SurfView lh.central.gii -overlay lh.logP.gii -clip -1.3 1.3 -colorbar
CAT_SurfView -output view.png lh.thickness.sub-01      # render a PNG and exit
```

### What a click tells you

Clicking a vertex marks it and reports it in the status bar: hemisphere, vertex
number, millimetre position, the overlay value and — for a −log10(p) map — the
p-value it stands for. With an atlas selected, the region name is part of the
same line:

```
lh vertex 1000    (-8.0, -31.7, 7.9) mm    value 6    p 0.000001    aparc_DK40: superiortemporal
```

`m` jumps to the strongest vertex of the overlay, which is where you usually
want to look first in a statistical map.

### Atlases and their borders

The right-click menu lists the atlases shipped with T1Prep (Desikan-Killiany
DK40, Destrieux a2009s, HCP-MMP1 and four Schaefer parcellations) and can draw
the **region borders** on the surface, as `cat_surf_results` does. An atlas only
fits a surface with the same number of vertices — the 32k templates T1Prep works
with — and a mismatch is reported rather than drawn in the wrong place.

### Surface and underlay

Two independent choices in the menu:

- **Surface** — `central`, `inflated` or `patch` (the flattened map), taken from
  the sibling files of the one you opened. Switching keeps the overlay.
- **Underlay** — mean curvature, sulcal depth, or nothing (an even grey). The
  shading always comes from the *folded* surface, so an inflated or flattened
  surface keeps its relief instead of turning blank.

A flat patch is shown once per hemisphere, the two mirroring each other.

### Cluster table

For a thresholded map, **Clusters…** lists every suprathreshold region with its
peak value, p-value, hemisphere, millimetre coordinate, vertex count, area in
mm² and — with an atlas selected — the region the peak falls in. The threshold
and a minimum cluster size can be changed in the dialog, selecting a row marks
that peak on the surface, and **Save CSV…** writes the table for a paper.

### Colours

Range, clip window, colormap, opacity, inversion and discrete levels live in the
control panel (`Ctrl`/`Cmd+D`, or `p`). The defaults follow `cat_surf_results`:

- two-sided data is scaled symmetrically, so both tails get the same colours;
- a −log10(p) map is rounded outwards to whole numbers;
- when nothing lies below the negative threshold the range **starts** at the
  threshold, so the whole colormap is spent on the values that are shown;
- changing the threshold moves the range with it.

`-colorbar` labels a −log10(p) overlay with the p-values it stands for
(0.05, 0.01, 0.001, …) rather than raw numbers.

---

## Volume viewer

`CAT_VolView` shows three orthogonal slices in an SPM-like layout, in
neurological orientation, with the anatomical direction of each pane edge marked
(L/R/A/P/S/I). Intensities are real intensities: `scl_slope` and `scl_inter`
from the NIfTI header are applied.

```bash
CAT_VolView T1.nii.gz --overlay spmT_0001.nii.gz   # overlay, resampled if needed
CAT_VolView T1.nii.gz --contour p1T1.nii.gz        # segmentation outlined on the T1
CAT_VolView T1.nii.gz --atlas aal3                 # name the region at the cursor
```

### Overlays and contours

An **overlay** or a **contour** only has to be *registered* to the image: a
different voxel grid is resampled through the millimetre space of the two
headers (nearest neighbour for overlays, so label maps and thresholded
statistics keep their values). That means an atlas, a template or a statistical
map can be shown on a native-space T1. Voxels the overlay has no value for stay
transparent.

Contours outline another volume on the slices — the fastest check of a
registration or a segmentation, as `CheckReg` does it. Several can be shown at
once, each in its own colour, with an editable level.

### Reading values and positions

The information panel in the free quadrant lists file name, dimensions, voxel
size, orientation code, data type and intensity range, plus the voxel index, mm
position and value under the cursor — and the region name when an atlas is
selected. The status bar repeats the position in **editable** mm and voxel
boxes, so a peak coordinate from a table can be typed in, with buttons for the
origin and the strongest voxel.

The display range can be dragged over the **intensity histogram** in the control
panel; the two handles are the ends of the window.

### Several volumes

Up to six volumes open one window each, tiled three per row and titled with the
directory they come from. Their cursors are linked — a click in one moves the
others to the same millimetre position — and what the context menu changes
applies to all of them.

### Zoom

The zoom belongs to the menu: dragging and scrolling do not change it, so the
wheel steps through slices instead. This is deliberate — a right-click opens the
menu and takes the mouse release with it, which otherwise leaves the view
zooming on every mouse move. Uncheck *Zoom → Lock zoom* (or start with
`--free-zoom`) to zoom with the mouse.

### Montage

**Montage…** opens a sheet of slices for a report figure: pick the orientation,
give start, step and stop in millimetres (as in `cat_vol_slice_overlay`) and the
number of columns and rows, or leave the layout on *auto*.

---

## Figures from a script

Both viewers render without opening a window, so the same figure can be produced
for a whole study.

```bash
# a montage of a statistical map, thresholded at p<0.05
CAT_VolView T1.nii.gz --montage --slices "25 30 40 80" --columns 4 \
    --overlay spmT_logP.nii.gz --threshold 0.05 --colormap FIRE --colorbar \
    --screenshot figure.png

# the surface equivalent
CAT_SurfView lh.central.gii -overlay lh.logP.gii -clip -1.3 1.3 \
    -colorbar -output figure_surface.png
```

`--slices` takes either a list of millimetre positions (`"25 30 40 80"`) or a
range (`"-40:10:60"` = start:step:stop); leave it out to cover the volume.
`--orientation` picks the plane, `--columns`/`--rows` the layout, `--no-labels`
drops the position labels and `--montage-size W H` sets the pixel size.

The overlay is coloured with `--range`, `--clip`, `--threshold P` (clips a
−log10(p) map at that p-value), `--colormap`, `--opacity`, `--inverse` and
`--discrete N`; the image underneath with `--range-bkg`. `--colorbar` labels a
−log10(p) overlay with p-values, exactly as `cat_surf_results` does.

`CAT_SurfView -output` and `CAT_VolView --screenshot` write a PNG and exit, so
neither needs a display. Run `CAT_VolView --help` or `CAT_SurfView -help` for
the full list.

---

## macOS application bundles

`t1prep-make-apps` wraps both viewers as `.app` bundles for the Dock and Finder:
double-click to pick a file, or drop files onto the icon. The first interactive
start of either viewer creates them in `~/Applications` by itself; set
`T1PREP_NO_APPS=1` to prevent that.

The bundles declare the file types they can open, so *Open With* offers them in
Finder. `.nii` and `.gii` can be owned outright; `.nii.gz` is a gzip archive to
macOS, so the viewer can only be offered as an alternative for it. See
[scripts/README.md](../scripts/README.md#make_macos_appssh) for the details of
registering them as the default.

---

## Keyboard reference

Both viewers list their keys with `h`, and both offer everything else through a
right-click menu.

**CAT_SurfView**

| Key | Does |
|-----|------|
| `u` `d` `l` `r` | rotate up, down, left, right (shift: 1°, ctrl: 180°) |
| `b` / `o` | flip the view / reset it |
| `m` | go to the strongest vertex of the overlay |
| `←` `→` | step through overlays (or meshes) |
| `+` `-` | zoom in and out (the mouse does not zoom) |
| `g` | save a screenshot |
| `w` / `s` | wireframe / shaded |
| `Ctrl`/`Cmd+D` | show or hide the control panel |

**CAT_VolView**

| Key | Does |
|-----|------|
| arrows, `PgUp`/`PgDn` | step through slices (one, or ten) |
| `+` `-` `0` | zoom in, out, whole volume |
| `o` / `m` | go to the origin / to the strongest voxel |
| `c` `a` `i` `n` `p` | crosshair, orientation letters, information, raw voxels, panel |
| `s` | save a screenshot |

Dropping files on a window opens them: in `CAT_VolView` a volume opens in a
linked window, **shift** makes it the overlay and **alt** outlines it; in
`CAT_SurfView` a mesh replaces the surface, an overlay is displayed and an
`.annot` becomes the atlas.
