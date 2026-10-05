from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, is_dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


class RpcError(RuntimeError):
    pass


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


class JsonRpcClient:
    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def call(self, method: str, payload: Any = None) -> Any:
        body = json.dumps({"method": method, "payload": json_value(payload)}).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RpcError(f"RPC {method!r} failed: {message}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RpcError(f"RPC {method!r} could not reach {self.endpoint}") from exc
        if not result.get("ok"):
            raise RpcError(f"RPC {method!r} failed: {result.get('error', 'unknown error')}")
        return result.get("result")

    def wait_ready(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.call("health")
                return
            except RpcError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"RPC service did not become ready: {self.endpoint}")
                time.sleep(0.1)


class JsonRpcServer:
    def __init__(
        self,
        host: str,
        port: int,
        dispatch: Callable[[str, Any], Any],
    ) -> None:
        self._dispatch = dispatch
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    request = json.loads(self.rfile.read(length) or b"{}")
                    method = str(request.get("method", ""))
                    if method == "health":
                        result = {"status": "ready"}
                    else:
                        result = outer._dispatch(method, request.get("payload"))
                    self._reply(200, {"ok": True, "result": json_value(result)})
                except BaseException as exc:
                    self._reply(
                        500,
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _reply(self, status: int, payload: Any) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def serve_until(self, stop_file: Path, poll_interval: float = 0.2) -> None:
        watcher = threading.Thread(
            target=self._watch_stop,
            args=(stop_file, poll_interval),
            daemon=True,
        )
        watcher.start()
        self._server.serve_forever(poll_interval=poll_interval)
        self._server.server_close()

    def serve_until_event(
        self, stop_event: threading.Event, poll_interval: float = 0.2
    ) -> None:
        watcher = threading.Thread(
            target=self._watch_event,
            args=(stop_event, poll_interval),
            daemon=True,
        )
        watcher.start()
        self._server.serve_forever(poll_interval=poll_interval)
        self._server.server_close()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _watch_stop(self, stop_file: Path, poll_interval: float) -> None:
        while not stop_file.exists():
            time.sleep(poll_interval)
        self._server.shutdown()

    def _watch_event(
        self, stop_event: threading.Event, poll_interval: float
    ) -> None:
        while not stop_event.wait(poll_interval):
            pass
        self._server.shutdown()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(json_value(payload), sort_keys=True) + "\n")
    temporary.replace(path)
