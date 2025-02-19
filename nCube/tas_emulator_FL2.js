let mqtt = require('mqtt');
let { nanoid } = require('nanoid');
let net = require('net');
const fs = require('fs');
const path = require('path');
if (process.env.CLIENT) serverAddress = '192.168.0.10';
else serverAddress = 'gateway';
let serverPort = 3105;
const HEADER_SIZE = 7; // 'WEIGHTS' 또는 'METRICS' 헤더의 크기

if (process.env.CLIENT) host = '192.168.0.60';
else host = 'server';

let tas = {
    client: {
        connected: false,
    },
    connection: {
        host: host,
        port: 1883,
        endpoint: '',
        clean: true,
        connectTimeout: 4000,
        reconnectPeriod: 4000,
        clientId: 'tas_' + nanoid(15),
        username: 'keti_thyme',
        password: 'keti_thyme',
    },
};
let sendDataTopic = {
    weights1: '/thyme/weights1',
    weights2: '/thyme/weights2',
    metrics1: '/thyme/metrics1',
    metrics2: '/thyme/metrics2',
    results1: '/thyme/results1',
    results2: '/thyme/results2',
    client1FromC: '/thyme/client1',
    client2FromC: '/thyme/client2',
};
let recvDataTopic = {
    led: '/led/set',
    client1FromS: '/client1FromS/set',
    client2FromS: '/client2FromS/set',
};
let socket = net.createServer();
let clientCount = 0;
const clients = new Map(); // for communication

socket.on('connection', (client) => {
    clientCount++;
    const clientId = `client${clientCount}`;
    console.log(`Connected to Sender Raspberry Pi (${clientId})`);
    let dataBuffer = Buffer.alloc(0);
    let dataSize = null;

    clients.set(clientId, client);

    client.on('data', (data) => {
        try {
            dataBuffer = Buffer.concat([dataBuffer, data]);
            while (dataBuffer.length >= HEADER_SIZE + 4) {
                const header = dataBuffer.slice(0, HEADER_SIZE).toString();
                const dataSize = dataBuffer.readUInt32BE(HEADER_SIZE);
                if (dataBuffer.length >= HEADER_SIZE + 4 + dataSize) {
                    const payload = dataBuffer.slice(HEADER_SIZE + 4, HEADER_SIZE + 4 + dataSize);
                    if (header === 'WEIGHTS') {
                        processWeightsData(payload, clientId);
                    } else if (header === 'METRICS') {
                        processMetricsData(payload, clientId);
                    } else if (header === 'RESULTS') {
                        processResultsData(payload, clientId);
                    } else if (header === 'OPERATE') {
                        processOperateData(payload, clientId);
                    } else {
                        console.error(`Unknown header: ${header}`);
                    }
                    dataBuffer = dataBuffer.slice(HEADER_SIZE + 4 + dataSize);
                } else {
                    break;
                }
            }
        } catch (error) {
            console.error('Error processing data:', error);
        }
    });
    client.on('end', () => {
        clients.delete(clientId);
        console.log(`Disconnected from Sender Raspberry Pi (${clientId})`);
    });
});
socket.listen(serverPort, serverAddress, () => {
    console.log(`Server listening on ${serverAddress}:${serverPort}`);
});

function processWeightsData(data, clientId) {
    console.log(`Processing weights data from ${clientId}, size: ${data.length} bytes`);
    const weightstopic = sendDataTopic[`weights${clientId.slice(-1)}`];
    if (weightstopic) {
        doPublish(weightstopic, data);
    } else {
        console.error(`No weights topic found for client ${clientId}`);
    }
}

function processMetricsData(data, clientId) {
    console.log(`Processing metrics data from ${clientId}, size: ${data.length} bytes`);
    const metrics = JSON.parse(data);
    const metricstopic = sendDataTopic[`metrics${clientId.slice(-1)}`];
    if (metricstopic) {
        doPublish(metricstopic, JSON.stringify(metrics));
    } else {
        console.error(`No metrics topic found for client ${clientId}`);
    }
}

function processResultsData(data, clientId) {
    console.log(`Processing results data from ${clientId}, size: ${data.length} bytes`);
    const results = JSON.parse(data);
    const resultstopic = sendDataTopic[`results${clientId.slice(-1)}`];
    if (resultstopic) {
        doPublish(resultstopic, JSON.stringify(results));
    } else {
        console.error(`No results topic found for client ${clientId}`);
    }
}

function processOperateData(data, clientId) {
    console.log(`Processing operate data from ${clientId}, size: ${data.length} bytes`);
    const operate = JSON.parse(data);
    const operatetopic = sendDataTopic[`client${clientId.slice(-1)}FromC`];
    if (operatetopic) {
        doPublish(operatetopic, JSON.stringify(operate));
    } else {
        console.error(`No operate topic found for client ${clientId}`);
    }
}

let createConnection = () => {
    if (tas.client.connected) {
        console.log('Already connected --> destroyConnection');
        destroyConnection();
    }
    if (!tas.client.connected) {
        tas.client.loading = true;
        const { host, port, endpoint, ...options } = tas.connection;
        const connectUrl = `mqtt://${host}:${port}${endpoint}`;
        try {
            tas.client = mqtt.connect(connectUrl, options);
            tas.client.on('connect', () => {
                console.log(host, 'Connection succeeded!');
                tas.client.connected = true;
                tas.client.loading = false;
                for (let topicName in recvDataTopic) {
                    if (recvDataTopic.hasOwnProperty(topicName)) {
                        doSubscribe(recvDataTopic[topicName]);
                    }
                }
            });
            tas.client.on('error', (error) => {
                console.log('Connection failed', error);
                destroyConnection();
            });
            tas.client.on('close', () => {
                console.log('Connection closed');
                destroyConnection();
            });
            tas.client.on('message', (topic, message) => {
                if (topic === recvDataTopic.led) {
                    // LED 제어 로직
                } else {
                    const regex = /^\/(.*?)FromS\//;
                    const match = topic.match(regex);

                    if (match) {
                        const bufferData = Buffer.from(message);
                        const decodedMessage = JSON.parse(bufferData.toString());
                        // decodedMessage = JSON.parse(decodedMessage);
                        console.log(decodedMessage);
                        message = decodedMessage;

                        const clientId = match[1];
                        const clientSocket = clients.get(clientId);
                        if (clientSocket) {
                            try {
                                const header = 'OPERATE';
                                const jsonMessage = JSON.stringify({
                                    header: header,
                                    message: message,
                                });
                                const messageLength = Buffer.alloc(4);
                                const messageBuffer = Buffer.from(jsonMessage, 'utf-8');
                                messageLength.writeUInt32BE(Buffer.byteLength(jsonMessage), 0);
                                const fullMessage = Buffer.concat([messageLength, messageBuffer]);

                                clientSocket.write(fullMessage);
                                console.log('Sent JSON message to client:', jsonMessage);
                            } catch (error) {
                                console.error('Error broadcasting message:', error);
                            }
                        } else {
                            console.log(`Client socket not found for ID: ${clientId}`);
                        }
                    } else {
                        console.log('No match found');
                    }
                }
            });
        } catch (error) {
            console.log('mqtt.connect error', error);
            tas.client.connected = false;
        }
    }
};

let doSubscribe = (topic) => {
    if (tas.client.connected) {
        const qos = 0;
        tas.client.subscribe(topic, { qos }, (error) => {
            if (error) {
                console.log('Subscribe to topics error', error);
                return;
            }
            console.log('Subscribe to topics (', topic, ')');
        });
    }
};

let doPublish = (topic, payload) => {
    if (tas.client.connected) {
        tas.client.publish(topic, payload, 0, (error) => {
            if (error) {
                console.log('Publish error', error);
            } else {
                console.log(`Published data to ${topic}, size: ${payload.length} bytes`);
            }
        });
    }
};

let destroyConnection = () => {
    if (tas.client.connected) {
        try {
            if (Object.hasOwnProperty.call(tas.client, '__ob__')) {
                tas.client.end();
            }
            tas.client = {
                connected: false,
                loading: false,
            };
            console.log('Successfully disconnected!');
        } catch (error) {
            console.log('Disconnect failed', error.toString());
        }
    }
};
createConnection();