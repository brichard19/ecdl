from ecc import ECCurve, ECPoint
import random


'''
Converts a string to integer by guessing the base
'''
def parseInt(n):
    return int(n, 0)

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
Class to hold ECDLP parameters
'''
class ECDLPParams:

    '''
    ECDLPParams constructor
    Takes a dictionary containing the parameters
    '''
    def __init__(self):
        self.field = None
        self.p = 0
        self.n = 0
        self.a = 0
        self.b = 0
        self.gx = 0
        self.gy = 0
        self.qx = 0
        self.qy = 0
        self.dBits = 0

    def decode(self, params):
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