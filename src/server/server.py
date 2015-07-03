import json
from ecc import ECCurve, ECPoint
import os
import random
import sys
import ecdl

from MySQLPointDatabase import MySQLPointDatabase
from pprint import pprint
from flask import Flask, jsonify, request
from flask_jsonschema import JsonSchema, ValidationError

app = Flask(__name__)
app.config['JSONSCHEMA_DIR'] = os.path.join(app.root_path, '/')
app.config['PROPAGATE_EXCEPTIONS'] = True

jsonschema = JsonSchema(app)

NUM_R_POINTS = 32
WORKDIR = './work'

# Global dictionary of contexts
_ctx = {}


'''
Look up a context based on id
'''
def getContext(id):
    if id in _ctx:
        return _ctx[id]

    return None


def loadAllContexts():
    files = os.listdir('work')

    for f in files:
        print("Reading Context" + f)
        ctx = ecdl.loadContext('work/' + f)
        _ctx[ctx.name] = ctx


'''
Converts a string to integer by guessing the base
'''
def parseInt(n):
    return int(n, 0)

'''
Encodes ECDLP parameters as json
'''
def encodeContextParams(ctx):

    content = {}
    content['params'] = ctx.params.encode()

    # Convert R points to string values
    content['points'] = []
    for e in ctx.rPoints:
        p = {}
        p['a'] = str(e['a'])
        p['b'] = str(e['b'])
        p['x'] = str(e['x'])
        p['y'] = str(e['y'])
        content['points'].append(p)

    return content

'''
Returns the status to the client
'''
@app.route("/status/<id>", methods=['GET'])
def status(id):
    ctx = getContext(id)
    if ctx == None:
        return "", 404

    response = {}
    response['status'] = ctx.status;

    return jsonify(response)

'''
Create a new ECDL problem. Sets up database etc so points
can be submitted
'''
@app.route("/create/<id>", methods=['POST'])
def create(id):

    # Make sure it doesn't already exist
    if getContext(id) != None:
        return "", 500

    print("Creating new context: " + id)
    print(request.json)

    content = request.json

    params = ecdl.ECDLPParams(content['params'])

    print("Creating context")
    ctx = ecdl.createContext(params, id)

    _ctx[ctx.name] = ctx

    return ""

'''
Returns the problem parameters to the client
'''
@app.route("/params/<id>", methods=['GET'])
def get_params(id):
    print("ID: " + id)
    ctx = getContext(id)

    if ctx == None:
        return "", 404

    content = encodeContextParams(ctx)

    return json.dumps(content)


@app.route("/submit/<id>", methods=['POST'])
#@jsonschema.validate('schemas', 'submit')
def submit_points(id):
    ctx = getContext(id)

    if ctx == None:
        print("Count not find context " +id)
        return "", 404

    if ctx.status == "stopped":
        return "", 500

    content = request.json

    # Verify all points
    for i in range(len(content)):
        
        a = parseInt(content[i]['a'])
        b = parseInt(content[i]['b'])
        x = parseInt(content[i]['x'])
        y = parseInt(content[i]['y'])

        # Check that the x value has 0 bits on the end
        if x % pow(2, ctx.params.dBits) != 0:
            print("Not distinguished point! Rejecting!")
            return "", 400

        endPoint = ECPoint(x, y)
        if verifyPoint(ctx.curve, ctx.pointG, ctx.pointQ, a, b, endPoint) == False:
            print("Invalid point!")
            print(content[i])
            print("")
            return "", 400

    foundCollision = False

    # Write points to database
    for i in range(len(content)):
        a = parseInt(content[i]['a'])
        b = parseInt(content[i]['b'])
        x = parseInt(content[i]['x'])
        y = parseInt(content[i]['y'])

        #Check for collision
        dp = ctx.database.get(x, y)
        if dp != None:
            if dp['a'] != a and dp['b'] != b:
                print("==== FOUND COLLISION ====")
                print("a1: " + hex(a))
                print("b1: " + hex(b))
                print("")
                print("a2: " + hex(dp['a']))
                print("b2: " + hex(dp['b']))
                print("")
                print("x: " + hex(x))
                print("y: " + hex(y))
                foundCollision = True
            else:
                print("Point already exists in database. Rejecting.")
        else:
            ctx.database.insert(a, b, x, y)

    return ""

'''
Verify that a point is on the curve
'''
def verifyPoint(curve, g, q, endA, endB, endPoint):

    # Check that end point exists
    if not curve.verifyPoint(endPoint):
        return False

    return True

'''
Program entry point
'''
def main():

    print("ECDLP Server")

    try:
        ecdl.loadConfig("config/config.json")
    except:
        print("Error opening config: " + sys.exc_info[0])
        sys.exit(1)

    loadAllContexts()

    print("Starting server")
    app.run(host = "0.0.0.0", port = ecdl.Config.port)

if __name__ == "__main__":
    main()
