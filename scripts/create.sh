#!/bin/bash

if [ $# != 3 ]; then
    echo "Usage: create <host:port> <json file> <name>"
    exit 1
fi

curl --verbose -H "Content-Type: application/json" --data @$2 http://$1/create/$3
