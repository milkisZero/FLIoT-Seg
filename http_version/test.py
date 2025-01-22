import json

host = "127.0.0.1"
port = "8080"

data = {
    "event":"client_wakeup",
    "client_id":"1",
    "url":f"http://{host}:{port}/aeWatcher"
}

print(data)