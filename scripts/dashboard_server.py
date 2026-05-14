"""Web dashboard server for live training monitoring.

Tails a JSONL training log, polls nvidia-smi, and streams events over
Server-Sent Events to a browser-based dashboard. Single Python file, single
HTML page, no npm — point browser at http://localhost:8765 and watch the run.

Usage:
    python scripts/dashboard_server.py
        --log runs/finewebedu_50m.jsonl
        --params 51490000
        --peak-tflops 35
        --total-steps 10000
        --total-tokens 1710000000
        --port 8765

The dashboard reads:
  /          → static HTML/JS dashboard
  /stream    → SSE event stream (train, gpu, config events)
  /config    → JSON with run config (one-shot)
  /history   → JSON with the full historical JSONL (for chart bootstrap)

Cloud-ready: run this on the same machine as the trainer (reads the JSONL
directly) or in a sidecar (mount the runs/ directory). The dashboard is a
single static HTML page so it can be hosted by any web server too.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory


# ---------------------------------------------------------------------------
# State (one instance, shared across SSE clients)
# ---------------------------------------------------------------------------

class State:
    def __init__(self, args):
        self.args = args
        self.config = {
            "log_path": str(args.log),
            "params": args.params,
            "peak_tflops": args.peak_tflops,
            "total_steps": args.total_steps,
            "total_tokens": args.total_tokens,
            "n_layers": args.n_layers,
            "d_model": args.d_model,
            "seq_len": args.seq_len,
        }
        # All JSONL lines we've ever seen, oldest first.
        self.history: list[dict] = []
        # Subscriber queues (one per connected browser).
        self.subscribers: list[queue.Queue] = []
        self.subscribers_lock = threading.Lock()
        # Latest GPU sample (None until first nvidia-smi success).
        self.last_gpu: dict | None = None

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self.subscribers_lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.subscribers_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, kind: str, payload: dict) -> None:
        event = {"type": kind, **payload}
        with self.subscribers_lock:
            dead = []
            for q in self.subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self.subscribers.remove(q)


# ---------------------------------------------------------------------------
# JSONL tailer — reads existing history once, then watches for new lines.
# ---------------------------------------------------------------------------

def jsonl_tailer(state: State) -> None:
    path = state.args.log
    pos = 0
    last_size = 0
    bootstrap_done = False

    while True:
        try:
            if not os.path.exists(path):
                time.sleep(1.0)
                continue

            size = os.path.getsize(path)
            if size < last_size:
                # File was truncated (e.g. fresh run); restart.
                pos = 0
                state.history.clear()
                bootstrap_done = False

            if size > pos:
                with open(path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    for raw in f:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        state.history.append(obj)
                        if bootstrap_done:
                            state.broadcast("train", obj)
                    pos = f.tell()
                last_size = size
                if not bootstrap_done:
                    bootstrap_done = True

            time.sleep(0.2)

        except Exception as e:
            print(f"[tailer] {e}", file=sys.stderr)
            time.sleep(1.0)


# ---------------------------------------------------------------------------
# nvidia-smi poller
# ---------------------------------------------------------------------------

def nvidia_smi_poller(state: State) -> None:
    fields = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    cmd = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    while True:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
            if proc.returncode != 0:
                state.broadcast("gpu_error", {"message": proc.stderr.strip()[:200]})
            else:
                line = proc.stdout.strip().splitlines()[0]
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 5:
                    sample = {
                        "util_pct": int(float(parts[0])),
                        "vram_used_mb": int(float(parts[1])),
                        "vram_total_mb": int(float(parts[2])),
                        "temp_c": int(float(parts[3])),
                        "power_w": float(parts[4]),
                        "t": time.time(),
                    }
                    state.last_gpu = sample
                    state.broadcast("gpu", sample)
        except Exception as e:
            state.broadcast("gpu_error", {"message": str(e)[:200]})
        time.sleep(2.0)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def make_app(state: State) -> Flask:
    here = Path(__file__).resolve().parent.parent
    dashboard_dir = here / "dashboard"
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(dashboard_dir, "index.html")

    @app.route("/<path:fname>")
    def static_file(fname):
        return send_from_directory(dashboard_dir, fname)

    @app.route("/config")
    def config():
        return jsonify(state.config)

    @app.route("/history")
    def history():
        return jsonify({"history": state.history})

    @app.route("/stream")
    def stream():
        q = state.subscribe()

        def generate():
            # First event always includes config + history so client can
            # bootstrap before the first new event arrives.
            yield f"data: {json.dumps({'type': 'config', **state.config})}\n\n"
            for h in state.history:
                yield f"data: {json.dumps({'type': 'train', **h})}\n\n"
            if state.last_gpu is not None:
                yield f"data: {json.dumps({'type': 'gpu', **state.last_gpu})}\n\n"

            try:
                while True:
                    try:
                        event = q.get(timeout=15.0)
                        yield f"data: {json.dumps(event)}\n\n"
                    except queue.Empty:
                        # Keep-alive comment so proxies don't close the
                        # connection during idle stretches.
                        yield ": keepalive\n\n"
            finally:
                state.unsubscribe(q)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )

    return app


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path,
                    default=Path("runs/finewebedu_50m.jsonl"),
                    help="path to the JSONL log to tail")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--params", type=int, default=51_490_000,
                    help="model parameter count for MFU computation")
    p.add_argument("--peak-tflops", type=float, default=35.0,
                    help="achievable peak TFLOPS for MFU baseline")
    p.add_argument("--total-steps", type=int, default=10_000,
                    help="run total steps for progress/ETA")
    p.add_argument("--total-tokens", type=int, default=1_710_000_000,
                    help="run total training tokens for token-progress bar")
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--seq-len", type=int, default=512)
    args = p.parse_args()

    state = State(args)

    t1 = threading.Thread(target=jsonl_tailer, args=(state,), daemon=True)
    t2 = threading.Thread(target=nvidia_smi_poller, args=(state,), daemon=True)
    t1.start()
    t2.start()

    app = make_app(state)
    url = f"http://{args.host}:{args.port}"
    print(f"\nDashboard live at {url}")
    print(f"  log:        {args.log}")
    print(f"  params:     {args.params:,}")
    print(f"  peak FLOPS: {args.peak_tflops} TFLOPS")
    print(f"  total:      {args.total_steps:,} steps · {args.total_tokens:,} tokens\n")

    app.run(host=args.host, port=args.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
