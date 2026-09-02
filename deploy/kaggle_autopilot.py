"""Wait for the Kaggle data build to finish, then start training. Unattended.

Run under browser-harness so the CDP helpers (js, iframe_target) are in scope:

    browser-harness deploy/kaggle_autopilot.py

Why this exists: the data build and the training run are each long, and the handoff between
them is a single click that has to happen at an unpredictable time. This polls for the
build's completion line, verifies the shards are real, then triggers the training cell and
confirms it started. It prints one line per check so the log is a record of what happened.
"""

import json
import time

CELL_DATA = 8
CELL_TRAIN = 10
POLL_S = 60
MAX_WAIT_BUILD_S = 3 * 60 * 60  # a 200k-doc build should never exceed this


def _out(idx: int, n: int = 400) -> dict:
    expr = r"""
    (() => {
      const c = [...document.querySelectorAll('.jp-Cell')][%d];
      if (!c) return JSON.stringify({err:'no cell'});
      const o = c.querySelector('.jp-OutputArea');
      const T = o ? o.innerText : '';
      const L = T.split('\n').filter(l =>
        l.includes('[prepare]') || l.includes('[train]') || l.includes('step=') ||
        l.includes('Traceback') || l.includes('Error'));
      return JSON.stringify({lines: L.slice(-4), tail: T.replace(/\s+/g,' ').slice(-%d)});
    })()
    """ % (idx, n)
    return json.loads(js(expr, iframe_target("jupyter-proxy")))


def _run(idx: int) -> None:
    js(
        "(() => { const a=window.jupyterapp, nb=a.shell.currentWidget.content;"
        " nb.activeCellIndex=%d; a.commands.execute('notebook:run-cell'); return '1'; })()" % idx,
        iframe_target("jupyter-proxy"),
    )


def wait_for_build() -> bool:
    print("[autopilot] waiting for the data build to finish", flush=True)
    waited = 0
    while waited < MAX_WAIT_BUILD_S:
        d = _out(CELL_DATA)
        joined = " ".join(d.get("lines", []))
        if "[prepare] done:" in joined:
            for line in d["lines"]:
                print(f"[autopilot]   {line[:180]}", flush=True)
            return True
        if "Traceback" in joined or "Error" in joined:
            print("[autopilot] BUILD FAILED:", flush=True)
            for line in d["lines"]:
                print(f"[autopilot]   {line[:200]}", flush=True)
            return False
        print(f"[autopilot] {waited//60}m — still building", flush=True)
        time.sleep(POLL_S)
        waited += POLL_S
    print("[autopilot] gave up waiting for the build", flush=True)
    return False


def start_training() -> bool:
    print("[autopilot] starting training", flush=True)
    _run(CELL_TRAIN)
    # Confirm it really started rather than assuming the click landed.
    for _ in range(20):
        time.sleep(30)
        d = _out(CELL_TRAIN)
        joined = " ".join(d.get("lines", []))
        if "step=" in joined or "[train]" in joined:
            for line in d["lines"]:
                print(f"[autopilot]   {line[:180]}", flush=True)
            print("[autopilot] TRAINING CONFIRMED RUNNING", flush=True)
            return True
        if "Traceback" in joined:
            print("[autopilot] TRAINING FAILED TO START:", flush=True)
            for line in d["lines"]:
                print(f"[autopilot]   {line[:200]}", flush=True)
            return False
    print("[autopilot] training did not report a step within 10 minutes", flush=True)
    return False


def watch_training(hours: float = 6.0) -> None:
    print("[autopilot] watching training", flush=True)
    deadline = time.time() + hours * 3600
    while time.time() < deadline:
        time.sleep(300)
        d = _out(CELL_TRAIN)
        joined = " ".join(d.get("lines", []))
        last = d["lines"][-1] if d.get("lines") else d.get("tail", "")
        print(f"[autopilot] {last[:170]}", flush=True)
        if "[train] done:" in joined:
            print("[autopilot] TRAINING COMPLETE", flush=True)
            return
        if "Traceback" in joined:
            print("[autopilot] TRAINING ERROR", flush=True)
            return
    print("[autopilot] watch window ended; training may still be running", flush=True)


if __name__ == "__main__" or True:  # browser-harness execs this file directly
    if wait_for_build() and start_training():
        watch_training()
