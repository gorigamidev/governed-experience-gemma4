"""OpenAI-compatible chat server that plays a scripted trajectory through the real harness (stdlib only).

The plan file (JSON) holds, per instance id, the coder steps and the real analyzer answer. Requests from the
coder (its tool list has `submit_patch`) get the next scripted step, indexed by how many assistant messages the
request already carries; requests from `code_analyzer` get the recorded analyzer answer as a final text. Every
request and its response are appended to the log, so each coder turn is a training example with the exact
prefix the harness built.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_lock = threading.Lock()


def _role_and_task(body: dict, plan: dict) -> tuple[str, str | None]:
    tools = {t.get("function", {}).get("name") for t in body.get("tools") or []}
    role = "coder" if "submit_patch" in tools else "analyzer"
    text = "\n".join(m["content"] for m in body.get("messages", []) if m.get("role") == "user"
                     and isinstance(m.get("content"), str))
    for iid, entry in plan.items():
        if entry["problem_head"] in text:
            return role, iid
    return role, None


def respond(body: dict, plan: dict) -> tuple[dict, dict]:
    """Return (message, meta) for one chat request."""
    role, iid = _role_and_task(body, plan)
    if iid is None:
        return {"role": "assistant", "content": "ERROR: unknown task"}, {"role": role, "task": None, "error": "unknown task"}
    entry = plan[iid]
    if role == "analyzer":
        return {"role": "assistant", "content": entry["analyzer_text"]}, {"role": role, "task": iid}
    turn = sum(1 for m in body["messages"] if m.get("role") == "assistant")
    steps = entry["steps"]
    if turn >= len(steps):
        return {"role": "assistant", "content": "DONE"}, {"role": role, "task": iid, "turn": turn, "overrun": True,
                                                         "loss": False}
    s = steps[turn]
    msg: dict = {"role": "assistant", "content": s["content"]}
    if s["thought"]:
        msg["reasoning_content"] = s["thought"]
    if s["tool"]:
        msg["tool_calls"] = [{"id": f"call_{iid}_{turn}", "type": "function",
                              "function": {"name": s["tool"], "arguments": json.dumps(s["args"])}}]
    return msg, {"role": role, "task": iid, "turn": turn, "loss": bool(s.get("loss", True))}


def make_handler(plan_path: Path, log_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self._send({"object": "list", "data": [{"id": "scripted", "object": "model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            plan = json.loads(plan_path.read_text())
            msg, meta = respond(body, plan)
            finish = "tool_calls" if msg.get("tool_calls") else "stop"
            out = {"id": f"chatcmpl-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
                   "model": body.get("model", "scripted"),
                   "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                   "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            with _lock, log_path.open("a") as f:
                f.write(json.dumps({**meta, "variant": plan.get(meta.get("task") or "", {}).get("variant"),
                                    "request": body, "response": msg}) + "\n")
            self._send(out)

        def _send(self, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(plan_path: Path, log_path: Path, port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(plan_path, log_path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18765)
    a = ap.parse_args()
    serve(a.plan, a.log, a.port)
    threading.Event().wait()
