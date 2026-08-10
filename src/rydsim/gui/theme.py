"""GreyNOC visual identity for the RydSim GUI.

Dark, operator-console aesthetic: near-black slate surfaces, grey-scale
hierarchy, a single signal-green accent (the 'NOC green') plus amber/red for
warnings, monospace-forward typography. Matplotlib style dict included so
embedded plots match the chrome.
"""

from __future__ import annotations

# ---- palette ---------------------------------------------------------------
BG0 = "#0b0e11"        # app background (near-black)
BG1 = "#12161b"        # panel background
BG2 = "#1a2027"        # raised surface / input background
BG3 = "#232b34"        # hover / selection surface
BORDER = "#2d3742"     # hairline borders
FG0 = "#e6edf3"        # primary text
FG1 = "#9fb0c0"        # secondary text
FG2 = "#5c6b7a"        # muted / disabled text
ACCENT = "#3ddc84"     # GreyNOC signal green
ACCENT_DIM = "#1f7a4c"
AMBER = "#ffb454"      # caution / LITERATURE-RECALL provenance
RED = "#ff5c57"        # error / UNVERIFIED provenance
BLUE = "#58a6ff"       # links / secondary series
VIOLET = "#bc8cff"     # tertiary series
CYAN = "#39c5cf"       # quaternary series

SERIES = [ACCENT, BLUE, AMBER, VIOLET, CYAN, RED, FG1]

FONT_MONO = ("Consolas", 10)
FONT_MONO_SM = ("Consolas", 9)
FONT_UI = ("Segoe UI", 10)
FONT_UI_SM = ("Segoe UI", 9)
FONT_TITLE = ("Segoe UI Semibold", 12)
FONT_BANNER = ("Consolas", 9)

BANNER = r"""
  ____                 _   _  ___   ____      ____            _ ____  _
 / ___|_ __ ___ _   _ | \ | |/ _ \ / ___|    |  _ \ _   _  __| / ___|(_)_ __ ___
| |  _| '__/ _ \ | | ||  \| | | | | |     _  | |_) | | | |/ _` \___ \| | '_ ` _ \
| |_| | | |  __/ |_| || |\  | |_| | |___ (_) |  _ <| |_| | (_| |___) | | | | | | |
 \____|_|  \___|\__, ||_| \_|\___/ \____|    |_| \_\\__, |\__,_|____/|_|_| |_| |_|
                |___/                               |___/
"""

TAGLINE = "RYDBERG ELECTROMETRY SIMULATOR · reproducible or it didn't happen"


def apply_ttk_theme(root) -> None:
    """Apply the GreyNOC dark theme to a Tk root via ttk styles."""
    import tkinter as tk
    from tkinter import ttk

    root.configure(bg=BG0)
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=BG1, foreground=FG0, fieldbackground=BG2,
                    bordercolor=BORDER, darkcolor=BG1, lightcolor=BG1,
                    troughcolor=BG2, focuscolor=ACCENT, selectbackground=BG3,
                    selectforeground=FG0, insertcolor=ACCENT, font=FONT_UI)

    style.configure("TFrame", background=BG1)
    style.configure("App.TFrame", background=BG0)
    style.configure("Panel.TFrame", background=BG1, relief="flat")

    style.configure("TLabel", background=BG1, foreground=FG0)
    style.configure("Muted.TLabel", background=BG1, foreground=FG1, font=FONT_UI_SM)
    style.configure("Title.TLabel", background=BG1, foreground=FG0, font=FONT_TITLE)
    style.configure("Accent.TLabel", background=BG1, foreground=ACCENT, font=FONT_MONO)
    style.configure("Banner.TLabel", background=BG0, foreground=ACCENT, font=FONT_BANNER)
    style.configure("Tagline.TLabel", background=BG0, foreground=FG2, font=FONT_MONO_SM)
    style.configure("Warn.TLabel", background=BG1, foreground=AMBER, font=FONT_UI_SM)
    style.configure("Error.TLabel", background=BG1, foreground=RED, font=FONT_UI_SM)

    style.configure("TButton", background=BG2, foreground=FG0, borderwidth=1,
                    focusthickness=1, padding=(10, 5))
    style.map("TButton",
              background=[("active", BG3), ("pressed", BG3)],
              foreground=[("disabled", FG2)])
    style.configure("Accent.TButton", background=ACCENT_DIM, foreground=FG0)
    style.map("Accent.TButton",
              background=[("active", ACCENT), ("pressed", ACCENT)],
              foreground=[("active", BG0), ("pressed", BG0)])

    style.configure("TEntry", fieldbackground=BG2, foreground=FG0, padding=4)
    style.configure("TCombobox", fieldbackground=BG2, background=BG2,
                    foreground=FG0, arrowcolor=FG1, padding=4)
    root.option_add("*TCombobox*Listbox.background", BG2)
    root.option_add("*TCombobox*Listbox.foreground", FG0)
    root.option_add("*TCombobox*Listbox.selectBackground", BG3)
    root.option_add("*TCombobox*Listbox.selectForeground", ACCENT)

    style.configure("TNotebook", background=BG0, borderwidth=0, tabmargins=(8, 6, 8, 0))
    style.configure("TNotebook.Tab", background=BG1, foreground=FG1,
                    padding=(14, 6), font=FONT_UI)
    style.map("TNotebook.Tab",
              background=[("selected", BG2)],
              foreground=[("selected", ACCENT)])

    style.configure("Treeview", background=BG1, fieldbackground=BG1,
                    foreground=FG0, rowheight=22, font=FONT_MONO_SM)
    style.configure("Treeview.Heading", background=BG2, foreground=FG1,
                    font=FONT_UI_SM, relief="flat")
    style.map("Treeview", background=[("selected", BG3)],
              foreground=[("selected", ACCENT)])

    style.configure("TLabelframe", background=BG1, bordercolor=BORDER,
                    relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=BG1, foreground=FG1,
                    font=FONT_UI_SM)

    style.configure("Horizontal.TProgressbar", background=ACCENT,
                    troughcolor=BG2, bordercolor=BORDER,
                    lightcolor=ACCENT, darkcolor=ACCENT)

    style.configure("TCheckbutton", background=BG1, foreground=FG0)
    style.map("TCheckbutton", background=[("active", BG1)])
    style.configure("TRadiobutton", background=BG1, foreground=FG0)
    style.map("TRadiobutton", background=[("active", BG1)])
    style.configure("Vertical.TScrollbar", background=BG2, troughcolor=BG0,
                    bordercolor=BG0, arrowcolor=FG2)
    style.configure("TSeparator", background=BORDER)


MPL_RC = {
    "figure.facecolor": BG1,
    "axes.facecolor": BG0,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": FG1,
    "axes.titlecolor": FG0,
    "axes.grid": True,
    "grid.color": BG2,
    "grid.linewidth": 0.6,
    "xtick.color": FG2,
    "ytick.color": FG2,
    "xtick.labelcolor": FG1,
    "ytick.labelcolor": FG1,
    "text.color": FG0,
    "legend.facecolor": BG1,
    "legend.edgecolor": BORDER,
    "legend.labelcolor": FG0,
    "lines.linewidth": 1.6,
    "font.family": "Consolas",
    "font.size": 9,
    "axes.prop_cycle": __import__("cycler").cycler(color=SERIES),
    "savefig.facecolor": BG1,
    "savefig.dpi": 150,
}
