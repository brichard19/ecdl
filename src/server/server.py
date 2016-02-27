import json
import ecc
from ecc import ECCurve, ECPoint
import os
import random
import sys
import ecdl
import util
from util import ECDLPParams

from pprint import pprint
from flask import Flask, jsonify, request
from flask_jsonschema import JsonSchema, ValidationError

app = Flask(__name__)
app.config['JSONSCHEMA_DIR'] = os.path.join(app.root_path, '/')
app.config['PROPAGATE_EXCEPTIONS'] = True

jsonschema = JsonSchema(app)

NUM_R_POINTS = 32

# Global dictionary of contexts
_ctx = {}


'''
Look up a context based on id
'''
def getContext(id):
    if id in _ctx:
        return _ctx[id]

    return None

'''
Loads all the contexts into the _ctx dictionary
'''
def loadAllContexts():
    names = ecdl.Database.getNames()

    if names == None:
        return

    for n in names:
        print("Loading context '" + n + "'")
        ctx = ecdl.loadContext(n)
        _ctx[ctx.name] = ctx


'''
Converts a string to integer by guessing the base
'''
def parseInt(n):
    return int(n, 0)

'''
Converts integer to string
'''
def intToString(n):
    return str(n).rstrip("L")

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
Route for /status/<id>

Gets the status of a job
'''
@app.route("/status/<id>", methods=['GET'])
def status(id):

    # Get the context
    ctx = getContext(id)
    if ctx == None:
        return "", 404

    # Get the status
    response = {}
    response['status'] = ctx.status;

    # Return the status
    return jsonify(response)

'''
Route for /create/<id>

Create a news job with the specified id
'''
@app.route("/create/<id>", methods=['POST'])
def create(id):

    # Make sure it doesn't already exist
    if getContext(id) != None:
        return "", 500

    content = request.json

    # Decode parameters
    params = ECDLPParams()
    params.decode(content['params'])

    if content.has_key('email'):
        email = content['email']
    else:
        email = ''

    #Verify parameters are correct
    if not ecc.verifyCurveParameters(params.a, params.b, params.p, params.n, params.gx, params.gy):
        return "Invalid ECC parameters", 400

    # Create the context
    ctx = ecdl.createContext(params, id, email)

    # Add context to list
    _ctx[ctx.name] = ctx

    return ""

'''
Route for /params/<id>

Gets the parameters for a job
'''
@app.route("/params/<id>", methods=['GET'])
def get_params(id):

    # Get the context
    ctx = getContext(id)

    if ctx == None:
        return "", 404

    # Get the parameters
    content = encodeContextParams(ctx)

    # Return parameters
    return json.dumps(content)

'''
Route for /submit/<id>

This route is for submitting distinguished points to the server
'''
@app.route("/submit/<id>", methods=['POST'])
def submit_points(id):

    # Get the context
    ctx = getContext(id)

    if ctx == None:
        print("Count not find context " + id)
        return "", 404

    # Check if the job is still running
    if ctx.status == "stopped":
        return "", 500

    content = request.json

    modulus = pow(2, ctx.params.dBits)

    points = []

    # Verify all points
    for i in range(len(content)):
        
        a = parseInt(content[i]['a'])
        b = parseInt(content[i]['b'])
        x = parseInt(content[i]['x'])
        y = parseInt(content[i]['y'])
        length = content[i]['length']

        # Verify the exponents are within range
        if a <= 0 or a >= ctx.curve.n or b <= 0 or b >= ctx.curve.n:
            print("Invalid exponents:")
            print(str(a))
            print(str(b))
            return "", 400

        # Verify point is valid
        if x < 0 or x >= ctx.curve.p or y < 0 or y >= ctx.curve.p:
            print("Invalid point:")
            print("X: " + str(x))
            print("y: " + str(y))
            return "", 400

        # Check that the x value has the correct number of 0 bits
        if x % modulus != 0:
            print("[" + hex(x) + "," + hex(y) +"]")
            print("Not distinguished point! Rejecting!")
            return "", 400

        # Verify aG = bQ = (x,y)
        endPoint = ECPoint(x, y)
        if verifyPoint(ctx.curve, ctx.pointG, ctx.pointQ, a, b, endPoint) == False:
            print("Invalid point!")
            print(content[i])
            print("")
            return "", 400

        # Append to list
        dp = {}
        dp['a'] = a
        dp['b'] = b
        dp['x'] = x
        dp['y'] = y
        dp['length'] = length
        points.append(dp)

    # Connect to database
    ctx.database.open()

    # Write list to database
    collisions = ctx.database.insertMultiple(points)

    # If there are any collisions, add them to the collisions table
    if collisions != None:
        for c in collisions:
            dp = ctx.database.get(c['x'], c['y'])
            print("==== FOUND COLLISION ====")
            print("a1:     " + hex(c['a']))
            print("b1:     " + hex(c['b']))
            print("length: " + intToString(c['length']))
            print("")
            print("a2:     " + hex(dp['a']))
            print("b2:     " + hex(dp['b']))
            print("length: " + intToString(dp['length']))
            print("")
            print("x:      " + hex(c['x']))
            print("y:      " + hex(c['y']))

            ctx.database.insertCollision(c['a'], c['b'], c['length'], dp['a'], dp['b'], dp['length'], c['x'], c['y'])

    ctx.database.close()

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
    except Exception as e:
        print("Error opening config file " + str(e))
        sys.exit(1)

    loadAllContexts()

    print("Starting server")
    app.run(host = "0.0.0.0", port = ecdl.Config.port)

if __name__ == "__main__":
    main()
