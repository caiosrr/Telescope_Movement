import argparse
import json
from urllib import request


DEFAULT_AGENT_URL = "http://10.6.0.34:18080"


def call_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Small client for controle/mount_agent.py.")
    parser.add_argument("--agent-url", default=DEFAULT_AGENT_URL)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health")
    sub.add_parser("position")
    sub.add_parser("stop")

    move = sub.add_parser("move")
    move.add_argument("--az", type=float, default=0.0, help="Relative azimuth move in degrees.")
    move.add_argument("--alt", type=float, default=0.0, help="Relative altitude move in degrees.")
    move.add_argument("--tol", type=float, default=0.005, help="Move tolerance in degrees.")

    args = parser.parse_args()
    base = args.agent_url.rstrip("/")

    if args.command == "health":
        result = call_json("GET", f"{base}/health")
    elif args.command == "position":
        result = call_json("GET", f"{base}/position")
    elif args.command == "stop":
        result = call_json("POST", f"{base}/stop", {})
    elif args.command == "move":
        result = call_json(
            "POST",
            f"{base}/move_relative",
            {
                "delta_az_deg": args.az,
                "delta_alt_deg": args.alt,
                "tolerance_deg": args.tol,
            },
        )
    else:
        raise RuntimeError(f"Comando desconhecido: {args.command}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
