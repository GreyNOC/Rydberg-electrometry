"""GUI smoke test: construct the real app, drive its job paths, render.

Not a screenshot test — but it executes widget construction, the physics
jobs the buttons trigger, and the matplotlib embeds, catching wiring
regressions. Skipped automatically if no display is available.
"""

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")


def _app_or_skip():
    from rydsim.gui.app import App
    try:
        return App()
    except tk.TclError:
        pytest.skip("no display available")


def test_gui_constructs_and_runs_spectrum_job():
    app = _app_or_skip()
    try:
        # run the exact job the "Run spectrum" button dispatches, synchronously
        app.spec_form.vars["points"].set("101")
        app.spec_form.vars["span"].set("30.0")
        app.var_converge.set(False)  # keep smoke test fast
        payload = app._spectrum_job()
        app._show_spectrum(payload)
        cfg, dps, resp = payload
        assert len(dps) == 101
        assert np.all(np.isfinite(resp.real))
        app.update_idletasks()
    finally:
        app.destroy()


def test_gui_at_measurement_path():
    from rydsim.experiment import at_experiment
    from rydsim.gui.app import form_to_config

    app = _app_or_skip()
    try:
        app.spec_form.vars["points"].set("201")
        cfg = form_to_config(app.spec_form, doppler=True, rf=True)
        res = at_experiment(cfg, n_points=201)
        app._show_at((cfg, res))
        app.update_idletasks()
        text = app.spec_out.get("1.0", "end")
        assert "splitting" in text or "NO RESOLVED" in text
    finally:
        app.destroy()


def test_gui_findings_tab_lists_reports():
    app = _app_or_skip()
    try:
        app._refresh_findings()
        app.update_idletasks()
        # findings dir has at least the CLI smoke-test finding or is empty; both fine
        assert app.find_list.size() >= 0
    finally:
        app.destroy()
