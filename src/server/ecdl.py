import json
from MySQLPointDatabase import MySQLPointDatabase
from ecc import ECCurve, ECPoint
import random

NUM_R_POINTS = 32
WORKDIR = './work'

'''
Converts a string to integer by guessing the base
'''
def parseInt(n):
    return int(n, 0)


'''
Config object. Need to call loadConfig() to load it
'''
Config = None


class ServerConfig:

    dbUser = ""
    dbPassword = ""
    dbHost = ""
    port = 9999

    def __init__(self, path):

        with open(path) as data_file:
            data = json.load(data_file)

        self.dbUser = data['dbUser']
        self.dbPassword = data['dbPassword']
        self.dbHost = data['dbHost']
        self.port = data['port']

'''
Loads the config
'''
def loadConfig(path):
    global Config
    Config = ServerConfig(path)


'''
Class to store ECDLP context.
'''
class ECDLPContext:

    database = None
    rA = []
    rB = []
    rPoints = []
    curve = None
    params = None
    name = ""
    status = "running"

    def __init__(self):
        self.database = None
        self.rA = []
        self.rB = []
        self.rPoints = []
        self.curve = None
        self.params = None
        self.name = ""
        self.status = "running"


'''
Class to hold ECDLP parameters
'''
class ECDLPParams:

    field = None
    p = 0
    n = 0
    a = 0
    b = 0
    gx = 0
    gy = 0
    qx = 0
    qy = 0
    dBits = 0

    '''
    ECDLPParams constructor
    Takes a dictionary containing the parameters
    '''
    def __init__(self, params):
        self.field = params['field']
        self.p = parseInt(params['p'])
        self.n = parseInt(params['n'])
        self.a = parseInt(params['a'])
        self.b = parseInt(params['b'])
        self.gx = parseInt(params['gx'])
        self.gy = parseInt(params['gy'])
        self.qx = parseInt(params['qx'])
        self.qy = parseInt(params['qy'])
        self.dBits = params['bits']

    '''
    Encode into json format
    '''
    def encode(self):
        encoded = {}
        encoded['field'] = self.field
        encoded['p'] = str(self.p)
        encoded['n'] = str(self.n)
        encoded['a'] = str(self.a)
        encoded['b'] = str(self.b)
        encoded['gx'] = str(self.gx)
        encoded['gy'] = str(self.gy)
        encoded['qx'] = str(self.qx)
        encoded['qy'] = str(self.qy)
        encoded['bits'] = self.dBits

        return encoded


'''
Loads problem parameters from file
'''
def loadParameters(path):
    with open(path) as data_file:
        params = json.load(data_file)

    paramsObj = ECDLPParams(params)
    a = parseInt(params['a'])
    b = parseInt(params['b'])
    p = parseInt(params['p'])
    n = parseInt(params['n'])
    gx = parseInt(params['gx'])
    gy = parseInt(params['gy'])
    qx = parseInt(params['qx'])
    qy = parseInt(params['qy'])

    curve = ECCurve(a, b, p, n, gx, gy)
    pointG = ECPoint(parseInt(params['gx']), parseInt(params['gy']))
    pointQ = ECPoint(parseInt(params['qx']), parseInt(params['qy']))

    return paramsObj, curve, pointG, pointQ

'''
Generates the R points for the random walk from the ECDLP parameters
'''
def generateRPoints(params):

    curve = ECCurve(params.a, params.b, params.p, params.n, params.gx, params.gy)
    pointG = ECPoint(params.gx, params.gy)
    pointQ = ECPoint(params.qx, params.qy)

    rPoints = []
     
    for i in range(NUM_R_POINTS):
        a = random.randint(2, curve.n)
        b = random.randint(2, curve.n)

        aG = curve.multiply(a, pointG)
        bQ = curve.multiply(b, pointQ)

        r = curve.add(aG, bQ)

        e = {}
        e['a'] = a
        e['b'] = b
        e['x'] = r.x
        e['y'] = r.y
        rPoints.append(e)

    return rPoints

'''
 Creates a new context
'''
def createContext(params, name):
    global Config
    ctx = ECDLPContext()

    # Create params object
    ctx.params = params

    # Generate random walk points
    ctx.rPoints = generateRPoints(ctx.params)

    # Set up curve
    ctx.curve = ECCurve(ctx.params.a, ctx.params.b, ctx.params.p, ctx.params.n, ctx.params.gx, ctx.params.gy)
    ctx.pointG = ECPoint(ctx.params.gx, ctx.params.gy)
    ctx.pointQ = ECPoint(ctx.params.qx, ctx.params.qy)

    # Create database
    ctx.name = name
    ctx.database = MySQLPointDatabase(Config.dbHost, Config.dbUser, Config.dbPassword, ctx.name)

    ctx.status = "running"

    # Create json object to write to file
    data = {}
    data['params'] = ctx.params.encode()
    data['name'] = name
    data['status'] = ctx.status

    # Encode R points
    data['rPoints'] = []
    for i in range(NUM_R_POINTS):
        record = {}
        record['a'] = str(ctx.rPoints[i]['a'])
        record['b'] = str(ctx.rPoints[i]['b'])
        record['x'] = str(ctx.rPoints[i]['x'])
        record['y'] = str(ctx.rPoints[i]['y'])
        data['rPoints'].append(record)

    # Write to file
    with open(WORKDIR + '/' + name + '.json', 'w') as outfile:
        json.dump(json.dumps(data), outfile)

    return ctx


'''
Load context from file
'''
def loadContext(path):

    ctx = ECDLPContext()

    with open(path) as data_file:
        data = json.load(data_file)

    # Load JSON data
    data = json.loads(data)


    ctx.name = data['name']
    ctx.status = data['status']

    ctx.database = MySQLPointDatabase(Config.dbHost, Config.dbUser, Config.dbPassword, ctx.name)

    ctx.params = ECDLPParams(data['params'])

    ctx.curve = ECCurve(ctx.params.a, ctx.params.b, ctx.params.p, ctx.params.n, ctx.params.gx, ctx.params.gy)
    ctx.pointG = ECPoint(ctx.params.gx, ctx.params.gy)
    ctx.pointQ = ECPoint(ctx.params.qx, ctx.params.qy)

    for i in range(len(data['rPoints'])):
        #ctx.rA.append(parseInt(data['rPoints'][i]['a']))
        #ctx.rB.append(parseInt(data['rPoints'][i]['b']))
        a = parseInt(data['rPoints'][i]['a'])
        b = parseInt(data['rPoints'][i]['b'])
        x = parseInt(data['rPoints'][i]['x'])
        y = parseInt(data['rPoints'][i]['y'])
       
        e = {}
        e['a'] = a
        e['b'] = b
        e['x'] = x
        e['y'] = y
        ctx.rPoints.append(e)

    return ctx
