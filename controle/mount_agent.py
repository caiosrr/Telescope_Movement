import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from controle.mov_mount_remoto import (
    DEFAULT_BASE_URL,
    TOLERANCIA_GRAUS,
    TelescopeClient,
    calc_error_az,
    move_relative_remote,
)


class MountAgentState:
    def __init__(self, base_url: str, label: str):
        self.label = label
        self.telescope = TelescopeClient(base_url)
        self.lock = threading.Lock()
        self.last_move = None
        self.last_error = None


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(state: MountAgentState):
    class MountAgentHandler(BaseHTTPRequestHandler):
        server_version = "MountAgent/0.1"

        def log_message(self, fmt, *args):
            print(f"[{self.log_date_time_string()}] {self.address_string()} - {fmt % args}")

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/health":
                    _json_response(self, 200, {"ok": True, "label": state.label})
                    return
                if path == "/position":
                    az, alt = state.telescope.read_altaz()
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "label": state.label,
                            "az_deg": az,
                            "alt_deg": alt,
                            "timestamp_epoch": time.time(),
                        },
                    )
                    return
                if path == "/status":
                    az, alt = state.telescope.read_altaz()
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "label": state.label,
                            "az_deg": az,
                            "alt_deg": alt,
                            "last_move": state.last_move,
                            "last_error": state.last_error,
                        },
                    )
                    return
                _json_response(self, 404, {"ok": False, "error": f"unknown endpoint {path}"})
            except Exception as exc:
                state.last_error = str(exc)
                _json_response(self, 500, {"ok": False, "error": str(exc)})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_len).decode("utf-8") if content_len else "{}"
                payload = json.loads(raw or "{}")

                if path == "/stop":
                    state.telescope.stop()
                    _json_response(self, 200, {"ok": True, "label": state.label, "stopped": True})
                    return

                if path == "/move_relative":
                    delta_az = float(payload.get("delta_az_deg", 0.0))
                    delta_alt = float(payload.get("delta_alt_deg", 0.0))
                    tolerance = float(payload.get("tolerance_deg", TOLERANCIA_GRAUS))

                    with state.lock:
                        az0, alt0 = state.telescope.read_altaz()
                        ok = move_relative_remote(
                            state.telescope,
                            delta_az,
                            delta_alt,
                            tolerance=tolerance,
                        )
                        azf, altf = state.telescope.read_altaz()
                        target_az = (az0 + delta_az) % 360.0
                        target_alt = alt0 + delta_alt
                        result = {
                            "ok": bool(ok),
                            "label": state.label,
                            "delta_az_deg": delta_az,
                            "delta_alt_deg": delta_alt,
                            "az_before_deg": az0,
                            "alt_before_deg": alt0,
                            "az_after_deg": azf,
                            "alt_after_deg": altf,
                            "final_err_az_deg": calc_error_az(target_az, azf),
                            "final_err_alt_deg": target_alt - altf,
                            "timestamp_epoch": time.time(),
                        }
                        state.last_move = result
                    _json_response(self, 200, result)
                    return

                _json_response(self, 404, {"ok": False, "error": f"unknown endpoint {path}"})
            except Exception as exc:
                state.last_error = str(exc)
                try:
                    state.telescope.stop()
                except Exception:
                    pass
                _json_response(self, 500, {"ok": False, "error": str(exc)})

    return MountAgentHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP agent for a locally connected ASCOM/Alpaca mount.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind. Use 0.0.0.0 for LAN access.")
    parser.add_argument("--port", type=int, default=18080, help="HTTP port for this agent.")
    parser.add_argument("--label", default="mount-agent", help="Human-readable mount label.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Local ASCOM/Alpaca telescope URL, e.g. http://127.0.0.1:11111/api/v1/telescope/0",
    )
    args = parser.parse_args()

    state = MountAgentState(args.base_url, args.label)
    state.telescope.ensure_ready()
    az, alt = state.telescope.read_altaz()
    print(f"{args.label}: conectado em {args.base_url}")
    print(f"Pos inicial: Az={az:.6f} deg | Alt={alt:.6f} deg")
    print(f"Servidor agent em http://{args.host}:{args.port}")
    print("Endpoints: GET /health, GET /position, POST /move_relative, POST /stop")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando agente...")
    finally:
        state.telescope.stop()
        server.server_close()


if __name__ == "__main__":
    main()
