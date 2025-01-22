import json

message = {
    "header":"OPERATE",
    "message":{
        "event":"client_wakeup",
        "client_id":"1",
        "url":"http://127.0.0.1:8080/aeWatcher"
    }
}

json_message = json.dumps(message)
message_data = json.loads(json_message)
header = message_data.get('header')
message = message_data.get('message')

print(header)
print(message)