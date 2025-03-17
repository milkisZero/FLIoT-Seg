/**
 * Created by Il Yeup, Ahn in KETI on 2017-02-23.
 */

/**
 * Copyright (c) 2018, OCEAN
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products derived from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

const ip = require('ip');
const { nanoid } = require('nanoid');

let conf = {};
let cse = {};
let ae = {};
let cnt_arr = [];
let sub_arr = [];
let acp = {};

conf.useprotocol = 'mqtt'; // select one for 'http' or 'mqtt' or 'coap' or 'ws'

conf.sim = 'disable'; // enable or disable

// build cse
cse = {
    host: 'mobius',
    host: 'mobius',
    port: '7579',
    name: 'Mobius',
    id: '/Mobius2',
    mqttport: '1883',
    wsport: '7577',
};

// build ae
let ae_name = 'FLIoT';
let ae_name = 'FLIoT';

ae = {
    name: ae_name,
    id: 'S' + ae_name,
    parent: '/' + cse.name,
    appid: 'Gateway',
    port: '9727',
    bodytype: 'json',
    tasport: '3105',
};

// build cnt
var count = 0;
cnt_arr = [
    {
        parent: '/' + cse.name + '/' + ae.name,
        name: 'metrics1',
    },
    // {
    //     parent: '/' + cse.name + '/' + ae.name,
    //     name: 'metrics2',
    // },
    // {
    //     parent: '/' + cse.name + '/' + ae.name,
    //     name: 'metrics2',
    // },
    {
        parent: '/' + cse.name + '/' + ae.name,
        name: 'results1',
    },
    // {
    //     parent: '/' + cse.name + '/' + ae.name,
    //     name: 'results2',
    // },
    // {
    //     parent: '/' + cse.name + '/' + ae.name,
    //     name: 'results2',
    // },
    {
        parent: '/' + cse.name + '/' + ae.name,
        name: 'client1FromC',
    },
    // {
    //     parent: '/' + cse.name + '/' + ae.name,
    //     name: 'client2FromC',
    // },
    // {
    //     parent: '/' + cse.name + '/' + ae.name,
    //     name: 'client2FromC',
    // },
    {
        parent: '/' + cse.name + '/' + ae.name,
        name: 'client1FromS',
    },
    // {
    //     parent: '/' + cse.name + '/' + ae.name,
    //     name: 'client2FromS',
    // },
    {
        parent: '/' + cse.name + '/' + ae.name,
        name: 'attacks1',
    },
];

// build sub
sub_arr = [
    {
        parent: '/' + cse.name + '/' + ae.name + '/' + 'client1FromS',
        name: 'client1FromS',
        parent: '/' + cse.name + '/' + ae.name + '/' + 'client1FromS',
        name: 'client1FromS',
        nu: 'mqtt://' + cse.host + ':' + cse.mqttport + '/' + ae.id + '?ct=json', // 'http:/' + ip.address() + ':' + ae.port + '/noti?ct=json',
    },
];

// for tas
let tas = {
    client: {
        connected: false,
    },

    connection: {
        host: 'mobius',
        host: 'mobius',
        port: 1883,
        endpoint: '',
        clean: true,
        connectTimeout: 4000,
        reconnectPeriod: 4000,
        clientId: 'thyme_' + nanoid(15),
        username: 'keti_thyme',
        password: 'keti_thyme',
    },
};

// build acp: not complete
acp.parent = '/' + cse.name + '/' + ae.name;
acp.name = 'acp-' + ae.name;
acp.id = ae.id;

conf.usesecure = 'disable';

if (conf.usesecure === 'enable') {
    cse.mqttport = '8883';
}

conf.cse = cse;
conf.ae = ae;
conf.cnt = cnt_arr;
conf.sub = sub_arr;
conf.acp = acp;
conf.tas = tas;

let getDataTopic = {
    fromTas: '/thyme/fromTas',
    client1FromC: '/thyme/client1',
    metrics1: '/thyme/metrics1',
    results1: '/thyme/results1',
    attacks1: '/thyme/attacks1',
};

let setDataTopic = {
    led: '/led/set',
    client1FromS: '/client1FromS/set',
};

let makeConnection = (clientCount) => {
    const clientId = `client${clientCount}`;
    getDataTopic[clientId + 'FromC'] = '/thyme/' + clientId;
    getDataTopic['metrics' + clientCount] = '/thyme/metrics' + clientCount;
    getDataTopic['results' + clientCount] = '/thyme/results' + clientCount;
    getDataTopic['attacks' + clientCount] = '/thyme/attacks' + clientCount;
    setDataTopic[clientId + 'FromS'] = '/' + clientId + 'FromS/set';

    // conf.cnt.push(
    //     {
    //         parent: '/' + cse.name + '/' + ae.name,
    //         name: 'metrics' + clientCount,
    //     },
    //     {
    //         parent: '/' + cse.name + '/' + ae.name,
    //         name: 'results' + clientCount,
    //     },
    //     {
    //         parent: '/' + cse.name + '/' + ae.name,
    //         name: clientId + 'FromS',
    //     },
    //     {
    //         parent: '/' + cse.name + '/' + ae.name,
    //         name: clientId + 'FromC',
    //     }
    // );
};

module.exports = { conf, getDataTopic, setDataTopic, makeConnection };