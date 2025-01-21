
import paho.mqtt.client as mqtt

# for tas
# let tas = {
#     client: {
#         connected: false,
#     },

#     connection: {
#         host: '192.168.0.60',
#         port: 1883,
#         endpoint: '',
#         clean: true,
#         connectTimeout: 4000,
#         reconnectPeriod: 4000,
#         clientId: 'thyme_' + nanoid(15),
#         username: 'keti_thyme',
#         password: 'keti_thyme',
#     },
# };

host = '192.168.0.60'
port = 1883
endpoint =  ''
# connectUrl = f"mqtt://{host}:${port}${endpoint}"

topics = ["/thyme/weights1",
          "/thyme/weights2",
          "/thyme/metrics1",
          "/thyme/metrics2",
          "/thyme/results1",
          "/thyme/results2",
          ]
 
# MQTT 클라이언트 생성 및 콜백 함수 정의
def on_connect(client, userdata, flags, connection):
    if connection == 0:
        print(f"Connected to MQTT Broker: {host}:{port}")
        # 연결 성공 후 토픽 구독
        
        for topic in topics:
            client.subscribe(topic)
            print(f"Subscribed to topic: {topic}")
    else:
        print(f"Failed to connect, return code {connection}")

def on_message(client, userdata, msg):
    try:
        # JSON 처리 시도
        payload = json.loads(msg.payload.decode('utf-8'))
        print("Detected JSON format:", payload)
       # print(len(payload))
    except Exception:
        try:
            # Pickle 처리 시도
            payload = pickle.loads(msg.payload)
          #  print("Detected Pickle format:", payload)
            print(len(payload))
        except Exception as e:
            print(f"Failed to process message: {e}")
    publish_message()
            
def publish_message():
    client.publish('/fl/set', "Hello MQTT from Python!")
            
def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT Broker")

if __name__ == '__main__':
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "PythonMQTTClient" )
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    try:
        client.connect(host, port)
        # MQTT 이벤트 루프 시작
        client.loop_forever()
    except Exception as e:
        print(f"Failed to connect to MQTT Broker: {e}")

    