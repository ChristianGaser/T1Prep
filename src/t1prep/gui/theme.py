"""The look of the T1Prep launcher: colours, palette and style sheet.

The viewers keep the native style of the platform — their windows are mostly
a render view.  The launcher is forms and a table, which Qt draws differently
on every desktop, so it brings one look of its own: the Fusion style with a
dark set of colours, whatever the desktop is set to.  The pictures of the
results are drawn on black, and a dark window is what they sit in best.

Only Qt Widgets are involved; nothing here needs OpenGL.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtGui, QtWidgets


@dataclass(frozen=True)
class Colors:
    """The colours of one scheme, as ``#rrggbb``."""

    window: str        # behind everything
    surface: str       # the table, fields, the panels' content
    raised: str        # hovered rows and buttons
    border: str
    text: str
    muted: str         # secondary text
    faint: str         # disabled text, queued marks
    accent: str
    on_accent: str     # text on the accent colour
    success: str
    warning: str
    danger: str


#: The launcher's colours
DARK = Colors(
    window="#1d1f23", surface="#25282d", raised="#30343b", border="#3a3f47",
    text="#e7e9ec", muted="#9aa2ad", faint="#5d6570", accent="#5b9bff",
    on_accent="#0d1117", success="#4cc38a", warning="#e0a73a", danger="#f0736b",
)


def blend(color: str, other: str, share: float) -> str:
    """*color* mixed with *share* (0..1) of *other*, as ``#rrggbb``."""
    a, b = QtGui.QColor(color), QtGui.QColor(other)
    mix = [round(x + (y - x) * share) for x, y in
           ((a.red(), b.red()), (a.green(), b.green()), (a.blue(), b.blue()))]
    return "#{:02x}{:02x}{:02x}".format(*mix)


def grade_color(grade: str, colors: Colors) -> str:
    """Colour of a quality grade: A and B are fine, C wants a look, D-F do not pass."""
    letter = (grade or "").strip()[:1].upper()
    if letter in ("A", "B"):
        return colors.success
    if letter == "C":
        return colors.warning
    return colors.danger if letter else colors.muted


def _palette(c: Colors) -> QtGui.QPalette:
    role, group = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
    palette = QtGui.QPalette()
    for which, value in (
        (role.Window, c.window), (role.WindowText, c.text),
        (role.Base, c.surface), (role.AlternateBase, c.window),
        (role.Text, c.text), (role.Button, c.surface), (role.ButtonText, c.text),
        (role.ToolTipBase, c.surface), (role.ToolTipText, c.text),
        (role.Highlight, c.accent), (role.HighlightedText, c.on_accent),
        (role.PlaceholderText, c.faint), (role.Link, c.accent),
        (role.Mid, c.border), (role.Dark, c.border), (role.Light, c.surface),
    ):
        palette.setColor(which, QtGui.QColor(value))
    for which in (role.WindowText, role.Text, role.ButtonText):
        palette.setColor(group.Disabled, which, QtGui.QColor(c.faint))
    return palette


def _style_sheet(c: Colors) -> str:
    selected = blend(c.surface, c.accent, 0.28)
    pressed = blend(c.accent, "#ffffff", 0.14)
    return f"""
    QWidget {{ color: {c.text}; }}
    QMainWindow, QDialog {{ background: {c.window}; }}
    QToolTip {{
        background: {c.surface}; color: {c.text};
        border: 1px solid {c.border}; padding: 5px 7px;
    }}

    /* ---- buttons ---- */
    QPushButton {{
        background: {c.surface}; border: 1px solid {c.border};
        border-radius: 6px; padding: 5px 12px; min-height: 18px;
    }}
    QPushButton:hover {{ background: {c.raised}; }}
    QPushButton:pressed {{ background: {c.border}; }}
    QPushButton:disabled {{ color: {c.faint}; background: {c.window}; }}
    QPushButton#primary {{
        background: {c.accent}; color: {c.on_accent};
        border: 1px solid {c.accent}; font-weight: 600; padding: 6px 18px;
    }}
    QPushButton#primary:hover, QPushButton#primary:pressed {{ background: {pressed}; }}
    QPushButton#primary:disabled {{
        background: {c.raised}; border-color: {c.border}; color: {c.faint};
    }}
    QPushButton#danger {{ color: {c.danger}; font-weight: 600; padding: 6px 18px; }}
    QPushButton#link, QToolButton#link {{
        border: none; background: transparent; color: {c.accent}; padding: 2px 0;
    }}
    QPushButton#link:hover, QToolButton#link:hover {{ text-decoration: underline; }}
    QPushButton#link:disabled, QToolButton#link:disabled {{ color: {c.faint}; }}
    QToolButton#section {{
        border: none; background: transparent; padding: 6px 0 4px 0;
        font-weight: 600; color: {c.text};
    }}
    QToolButton#picker {{
        background: {c.surface}; border: 1px solid {c.border};
        border-radius: 6px; padding: 3px 22px 3px 8px;
    }}
    QToolButton#picker:hover {{ background: {c.raised}; }}
    QToolButton#picker:disabled {{ color: {c.faint}; background: {c.window}; }}
    QToolButton#picker::menu-indicator {{
        subcontrol-position: right center; right: 7px;
    }}

    /* ---- the two-way switch above the side panel and the preview ---- */
    QPushButton#segment {{
        background: transparent; border: none; border-radius: 5px;
        padding: 4px 14px; color: {c.muted};
    }}
    QPushButton#segment:hover {{ color: {c.text}; }}
    QPushButton#segment:checked {{
        background: {c.surface}; color: {c.text}; font-weight: 600;
        border: 1px solid {c.border};
    }}
    QPushButton#segment:disabled {{ color: {c.faint}; }}
    QPushButton#segment[compact="true"] {{ padding: 4px 8px; }}
    QToolButton#icon {{
        border: none; background: transparent; padding: 4px; border-radius: 6px;
    }}
    QToolButton#icon:hover {{ background: {c.raised}; }}
    QFrame#segments {{ background: {c.raised}; border-radius: 7px; }}

    /* ---- fields ---- */
    QSpinBox, QComboBox, QLineEdit {{
        background: {c.surface}; border: 1px solid {c.border};
        border-radius: 6px; padding: 3px 6px; min-height: 18px;
        selection-background-color: {c.accent}; selection-color: {c.on_accent};
    }}
    QSpinBox:focus, QComboBox:focus, QLineEdit:focus {{ border-color: {c.accent}; }}
    QSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {{
        color: {c.faint}; background: {c.window};
    }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox QAbstractItemView, QMenu {{
        background: {c.surface}; border: 1px solid {c.border};
        selection-background-color: {selected}; selection-color: {c.text};
        outline: none;
    }}
    QMenu {{ padding: 4px; }}
    QMenu::item {{ padding: 5px 22px 5px 12px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {selected}; }}
    QMenu::item:disabled {{ color: {c.faint}; }}
    QMenu::separator {{ height: 1px; background: {c.border}; margin: 4px 6px; }}
    QCheckBox {{ spacing: 7px; padding: 2px 0; }}
    QCheckBox:disabled {{ color: {c.faint}; }}

    /* ---- the table of images ---- */
    QTableView {{
        background: {c.surface}; border: 1px solid {c.border};
        border-radius: 8px; gridline-color: transparent; outline: none;
        selection-background-color: {selected}; selection-color: {c.text};
    }}
    QTableView::item {{ padding: 0 8px; border: none; }}
    QTableView::item:selected {{ background: {selected}; color: {c.text}; }}
    QHeaderView {{ background: transparent; }}
    QHeaderView::section {{
        background: transparent; color: {c.muted}; border: none;
        border-bottom: 1px solid {c.border}; padding: 7px 8px; font-weight: 600;
    }}
    /* the numbers of the rows: quieter than the names beside them */
    QHeaderView::section:vertical {{
        color: {c.faint}; border-bottom: none; font-weight: normal;
        padding: 0 2px 0 10px;
    }}
    QTableCornerButton::section {{
        background: transparent; border: none;
        border-bottom: 1px solid {c.border};
    }}

    /* ---- panels ---- */
    QFrame#panel {{
        background: {c.surface}; border: 1px solid {c.border}; border-radius: 8px;
    }}
    QFrame#rule {{ background: {c.border}; max-height: 1px; border: none; }}
    QLabel#title {{ font-size: 15px; font-weight: 600; }}
    QLabel#muted, QLabel#path {{ color: {c.muted}; }}
    QLabel#heading {{ color: {c.muted}; font-weight: 600; padding-top: 8px; }}
    QScrollArea {{ background: transparent; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QPlainTextEdit {{
        background: {c.surface}; border: 1px solid {c.border};
        border-radius: 8px; padding: 6px;
        selection-background-color: {c.accent}; selection-color: {c.on_accent};
    }}

    /* ---- progress and scroll bars ---- */
    QProgressBar {{
        background: {c.raised}; border: none; border-radius: 3px;
        max-height: 6px; min-height: 6px;
    }}
    QProgressBar::chunk {{ background: {c.accent}; border-radius: 3px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{
        background: {c.border}; border-radius: 3px; min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {c.faint}; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{
        background: {c.border}; border-radius: 3px; min-width: 28px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QSplitter::handle {{ background: transparent; }}
    """


def apply(app: QtWidgets.QApplication) -> Colors:
    """Give *app* the launcher's look; return the colours it now uses."""
    colors = DARK
    app.setStyle("Fusion")
    app.setPalette(_palette(colors))
    app.setStyleSheet(_style_sheet(colors))
    return colors
