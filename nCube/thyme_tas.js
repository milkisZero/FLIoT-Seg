global.socket_arr = {};
let tas_buffer = {};
exports.buffer = tas_buffer;

let mqtt = require('mqtt');
let moment = require('moment');

const { makeConnection, getDataTopic, setDataTopic } = require('./conf.js');

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
                // console.log(topic);
                const key = Object.keys(getDataTopic).find((key) => getDataTopic[key] === topic);

                if (topic === getDataTopic.fromTas) {
                    clientCount = JSON.parse(message.toString()) + 1;
                    makeConnection(clientCount);
                    const clientId = `client${clientCount}`;
                    push_cnt_arr(clientId + 'FromC', '/thyme/' + clientId, 1);
                    push_cnt_arr('metrics' + clientCount, '/thyme/metrics' + clientCount, 1);
                    push_cnt_arr('results' + clientCount, '/thyme/results' + clientCount, 1);
                    push_cnt_arr('attacks' + clientCount, '/thyme/attacks' + clientCount, 1);
                    push_cnt_arr(clientId + 'FromS', '/' + clientId + 'FromS/set', 0);
                } else if (key) {
                    try {
                        parent = conf.cnt[1].parent + '/' + key;
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
                        console.error('Error processing data:', error);
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
    // console.log(message);
    // console.log(message.toString());
    if (setDataTopic.hasOwnProperty(topicName)) {
        conf.tas.client.publish(setDataTopic[topicName], JSON.stringify(message));
    }
};

let push_cnt_arr = (key, value, tag) => {
    if (conf.cnt.some((item) => item.name === key)) return;

    conf.cnt.push({
        parent: '/' + conf.cse.name + '/' + conf.ae.name,
        name: key,
    });

    let count = conf.cnt.length - 1;
    var parent = conf.cnt[count].parent;
    var rn = conf.cnt[count].name;
    onem2m_client.create_cnt(parent, rn, count, (rsc, res_body, count) => {
        console.log('created container: ', rn);
        if (tag) doSubscribe(value);
        else sub_for_mobius(key);
    });

};

let sub_for_mobius = (value) => {
    if (conf.sub.some((item) => item.name === value)) return;

    conf.sub.push({
        parent: '/' + conf.cse.name + '/' + conf.ae.name + '/' + value,
        name: value,
        nu: 'mqtt://' + conf.cse.host + ':' + conf.cse.mqttport + '/' + conf.ae.id + '?ct=json',
    });

    let count = conf.sub.length - 1;
    var parent = conf.sub[count].parent;
    var rn = conf.sub[count].name;
    var nu = conf.sub[count].nu;
    onem2m_client.create_sub(parent, rn, nu, count, (rsc, res_body, count) => {
        console.log('created subscribe container: ', rn);
    });
};