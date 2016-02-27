import json
from MySQLPointDatabase import MySQLPointDatabase

from ecc import ECCurve, ECPoint
import random

NUM_R_POINTS = 32

'''
Converts a string to integer by guessing the base
'''
def parseInt(n):
    return int(n, 0)

'''
Config object. Need to call loadConfig() to load it
'''
Config = None
Database = None

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
    global Database

    Config = ServerConfig(path)
    Database = MySQLPointDatabase(Config.dbHost, Config.dbUser, Config.dbPassword)

'''
Class to store ECDLP context.
'''
class ECDLPContext:

    def __init__(self, ctxName):
        self.name = ctxName
        self.database = None
        self.rPoints = []
        self.curve = None
        self.params = None
        self.status = "running"
        self.email = None
        self.collisions = 0

    def getConnection(self):
        return database.getConnection(self.name)

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
def createContext(params, name, email):
  
    ctx = ECDLPContext(name)

    ctx.params = params
    ctx.rPoints = generateRPoints(ctx.params)

    Database.createContext(ctx.name, ctx.email, ctx.params, ctx.rPoints)

    ctx.curve = ECCurve(ctx.params.a, ctx.params.b, ctx.params.p, ctx.params.n, ctx.params.gx, ctx.params.gy)
    ctx.pointG = ECPoint(ctx.params.gx, ctx.params.gy)
    ctx.pointQ = ECPoint(ctx.params.qx, ctx.params.qy)

    ctx.database = Database.getConnection(ctx.name)

    return ctx

'''
 Loads an existing context
'''
def loadContext(name):
    ctx = ECDLPContext(name)

    ctx.database = Database.getConnection(ctx.name)

    ctx.database.open()
    ctx.rPoints = ctx.database.getRPoints()
    ctx.params = ctx.database.getParams()
    ctx.status = ctx.database.getStatus()
    ctx.solution = ctx.database.getSolution()
    ctx.collisions = ctx.database.getNumCollisions()
    ctx.database.close()

    ctx.curve = ECCurve(ctx.params.a, ctx.params.b, ctx.params.p, ctx.params.n, ctx.params.gx, ctx.params.gy)
    ctx.pointG = ECPoint(ctx.params.gx, ctx.params.gy)
    ctx.pointQ = ECPoint(ctx.params.qx, ctx.params.qy)

    return ctx
