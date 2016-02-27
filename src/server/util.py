from ecc import ECCurve, ECPoint
import random


'''
Converts a string to integer by guessing the base
'''
def parseInt(n):
    return int(n, 0)

def toHex(n):
    return hex(n).rstrip("L").lstrip("0x") or "0"

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

'''
Compresses an elliptic curve point
'''
def compress_point(x, y):
    even = True

    if y & 0x01 == 0x01:
        even = False

    if even:
        return "02" + toHex(x)
    else:
        return "03" + toHex(x)


'''
Decompresses a compressed point
'''
def decompress_point(compressed, a, b, p):

    print "Decompressing " + compressed

    # Get the even/odd of the y coordinate
    sign = compressed[:2]

    # Extract x coordinate
    x = int(compressed[2:], 16)

    # compute the solutions of y to y^2 = x^3 + ax + b
    z = ((x*x*x) + a * x + b) % p
    print "Computing square root of " + hex(z)
    y = squareRootModP(z, p)

    #y1 is even, y2 is odd
    y1 = y[0]
    y2 = y[1]

    if y1 & 0x01 == 0x01:
        tmp = y1
        y1 = y2
        y2 = tmp

    # Return the x,y pair
    if sign == "02":
        return x, y1
    else:
        return x, y2


def legendre_symbol(a,p):
    ls = pow(a, (p-1)//2, p)

    if(ls == p - 1):
        return -1
    return ls

def squareRootModP(n,p):

    if(n == 0):
        return [0]

    if(p == 2):
        return [n]

    if(legendre_symbol(n,p) != 1):
        return []

    # Check for easy case
    if(n % 4 == 3):
        r = pow(n, (p+1)/4, p)
        return [r, p - r]

    # Factor out powers of 2 from p - 1
    q = p - 1
    s = 0

    while(q % 2 == 0):
        s = s + 1
        q = q // 2

    # Select z which is a quadratic non-residue mod p
    z = 2
    while(legendre_symbol(z,p) != -1):
        z = z + 1

    c = pow(z, q, p)

    r = pow(n, (q+1)//2, p)
    t = pow(n, q, p)
    m = s

    while(t != 1):

        # Find lowest i where t^(2^i) = 1
        i = 1
        e = 0
        for e in range(1,m):
            if(pow(t, pow(2,i), p) == 1):
                break
            i = i * 2
        
        b = pow(c, 2**(m - i - 1), p)
        r = (r * b) % p
        t = (t * b * b) % p
        c = (b * b) % p
        m = i

    return [r, p - r]