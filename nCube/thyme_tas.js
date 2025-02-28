global.socket_arr = {};
let tas_buffer = {};
exports.buffer = tas_buffer;

let mqtt = require('mqtt');
let moment = require('moment');

let getDataTopic = {
    weights1: '/thyme/weights1',
    weights2: '/thyme/weights2',
    metrics1: '/thyme/metrics1',
    metrics2: '/thyme/metrics2',
    results1: '/thyme/results1',
    results2: '/thyme/results2',
    client1FromC: '/thyme/client1',
    client2FromC: '/thyme/client2',
};

let setDataTopic = {
    led: '/led/set',
    client1FromS: '/client1FromS/set',
    client2FromS: '/client2FromS/set',
};

let createConnection = () => {
    if (conf.tas.client.connected) {
        console.log('Already connected --> destroyConnection');
        destroyConnection();
    }

    if (!conf.tas.client.connected) {
        conf.tas.client.loading = true;
        const { host, port, endpoint, ...options } = conf.tas.connection;
        const connectUrl = `mqtt://${host}:${port}${endpoint}`;
        try {
            conf.tas.client = mqtt.connect(connectUrl, options);

            conf.tas.client.on('connect', () => {
                console.log(host, 'Connection succeeded!');
                conf.tas.client.connected = true;
                conf.tas.client.loading = false;
                for (let topicName in getDataTopic) {
                    if (getDataTopic.hasOwnProperty(topicName)) {
                        doSubscribe(getDataTopic[topicName]);
                    }
                }
            });

            conf.tas.client.on('error', (error) => {
                console.log('Connection failed', error);
                destroyConnection();
            });

            conf.tas.client.on('close', () => {
                console.log('Connection closed');
                destroyConnection();
            });

            conf.tas.client.on('message', (topic, message) => {
                let content = null;
                let parent = null;
                if (topic === getDataTopic.weights1 || topic === getDataTopic.weights2) {
                    try {
                        parent =
                            topic === getDataTopic.weights1
                                ? conf.cnt[1].parent + '/' + conf.cnt[0].name
                                : conf.cnt[1].parent + '/' + conf.cnt[1].name;
                        content = message;
                        console.log(`Received weights data: ${topic}, size: ${message.length} bytes`);
                        if (content) {
                            onem2m_client.create_cin(parent, 1, content, this, (status, res_body, to, socket) => {
                                console.log('x-m2m-rsc : ' + status + ' <----');
                            });
                        }
                    } catch (error) {
                        console.error('Error processing weights data:', error);
                    }
                } else if (topic === getDataTopic.metrics1 || topic === getDataTopic.metrics2) {
                    try {
                        parent =
                            topic === getDataTopic.metrics1
                                ? conf.cnt[1].parent + '/' + conf.cnt[2].name
                                : conf.cnt[1].parent + '/' + conf.cnt[3].name;
                        content = JSON.parse(message.toString());
                        console.log(`Received metrics data: ${topic}`);
                        if (content) {
                            onem2m_client.create_cin(
                                parent,
                                1,
                                JSON.stringify(content),
                                this,
                                (status, res_body, to, socket) => {
                                    console.log('x-m2m-rsc : ' + status + ' <----');
                                }
                            );
                        }
                    } catch (error) {
                        console.error('Error processing metrics data:', error);
                    }
                } else if (topic === getDataTopic.results1 || topic === getDataTopic.results2) {
                    try {
                        parent =
                            topic === getDataTopic.results1
                                ? conf.cnt[1].parent + '/' + conf.cnt[4].name
                                : conf.cnt[1].parent + '/' + conf.cnt[5].name;
                        content = JSON.parse(message.toString());
                        console.log(`Received results data: ${topic}`);
                        if (content) {
                            onem2m_client.create_cin(
                                parent,
                                1,
                                JSON.stringify(content),
                                this,
                                (status, res_body, to, socket) => {
                                    console.log('x-m2m-rsc : ' + status + ' <----');
                                }
                            );
                        }
                    } catch (error) {
                        console.error('Error processing results data:', error);
                    }
                } else if (topic === getDataTopic.client1FromC || topic === getDataTopic.client2FromC) {
                    try {
                        parent =
                            topic === getDataTopic.client1FromC
                                ? conf.cnt[1].parent + '/' + conf.cnt[6].name
                                : conf.cnt[1].parent + '/' + conf.cnt[7].name;
                        content = JSON.parse(message.toString());
                        console.log(`Received client data: ${topic}`);
                        if (content) {
                            onem2m_client.create_cin(
                                parent,
                                1,
                                JSON.stringify(content),
                                this,
                                (status, res_body, to, socket) => {
                                    console.log('x-m2m-rsc : ' + status + ' <----');
                                }
                            );
                        }
                    } catch (error) {
                        console.error('Error processing results data:', error);
                    }
                }
            });
        } catch (error) {
            console.error('ERROR!!!', error);
        }
    }
};
let doSubscribe = (topic) => {
    if (conf.tas.client.connected) {
        const qos = 0;
        conf.tas.client.subscribe(topic, { qos }, (error) => {
            if (error) {
                console.log('Subscribe to topics error', error);
                return;
            }
            console.log('Subscribe to topics (', topic, ')');
        });
    }
};

let doUnSubscribe = (topic) => {
    if (conf.tas.client.connected) {
        conf.tas.client.unsubscribe(topic, (error) => {
            if (error) {
                console.log('Unsubscribe error', error);
            }
            console.log('Unsubscribe to topics (', topic, ')');
        });
    }
};

let doPublish = (topic, payload) => {
    if (tas.client.connected) {
        tas.client.publish(topic, payload, 0, (error) => {
            if (error) {
                console.log('Publish error', error);
            }
        });
    }
};

let destroyConnection = () => {
    if (conf.tas.client.connected) {
        try {
            if (Object.hasOwnProperty.call(conf.tas.client, '__ob__')) {
                conf.tas.client.end();
            }
            conf.tas.client = {
                connected: false,
                loading: false,
            };
            console.log('Successfully disconnected!');
        } catch (error) {
            console.log('Disconnect failed', error.toString());
        }
    }
};

exports.ready_for_tas = function ready_for_tas() {
    createConnection();

    if (conf.sim === 'enable') {
        let t_count = 0;
        setInterval(
            () => {
                let val = (Math.random() * 50).toFixed(1);
                doPublish('/thyme/co2', val);
            },
            5000,
            t_count
        );
    }
};

exports.send_to_tas = function send_to_tas(topicName, message) {
    console.log(message);
    console.log(message.toString());
    if (setDataTopic.hasOwnProperty(topicName)) {
        conf.tas.client.publish(setDataTopic[topicName], JSON.stringify(message));
    }
};
