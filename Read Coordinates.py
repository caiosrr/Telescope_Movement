import itertools
import requests
import time

BASE_URL = "http://127.0.0.1:11111/api/v1/telescope/0"
CLIENT_ID = 1
_transaction_ids = itertools.count(1)


def call(method: str, command: str, **request_kwargs):
    params = {"ClientID": CLIENT_ID, "ClientTransactionID": next(_transaction_ids)}
    params.update(request_kwargs.pop("params", {}))
    response = requests.request(
        method, f"{BASE_URL}/{command}", params=params, timeout=10, **request_kwargs
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("ErrorNumber", 0):
        raise RuntimeError(f"{command}: {payload.get('ErrorMessage', 'erro desconhecido')}")
    return payload.get("Value")

print(call("GET", "axisrates", params={"Axis": 0}))  # Azimuth
