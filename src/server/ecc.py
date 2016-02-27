import random
import sys
import binascii
import math

'''
Compute inverse mod p
'''
def invModP(n, p):
    a = n
    b = p
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q,r = b//a,b%a; m,n = x-u*q,y-v*q # use x//y for floor "floor division"
        b,a, x,y, u,v = a,r, u,v, m,n
    return x % p

class ECPoint:
    x = 0
    y = 0
    
    def __init__(self,x=0,y=0):
        self.x = x
        self.y = y

    def isPointAtInfinity(self):
        if self.x == 0 and self.y == 0:
            return True
        else:
            return False

# Determines if a number is prime using the
# Miller-Rabin primality test
def isPrime(p):
    k = 128
    s = 0
    t = p - 1
    q = 1
   
    # Eliminate even numbers
    if( p & 1  == 0 ):
        return False
    
    # Fermat's little theorem check
    if(pow(2, p - 1, p) != 1):
        return False
    
    # Factor powers of 2 from p - 1    
    while(t & 1 == 0):
        t = t >> 1
        s = s + 1
        q = q * 2

    d = int((p - 1) / q)

    while(k >= 0):
        k = k - 1
        a = random.randint(2, p - 2)

        x = pow(a, d, p)

        if( x == 1 or x == p - 1 ):
            continue

        r = 1
        while( r <= s - 1 ):
            x = (x*x)%p
            if( x == 1 ):
                return False
            if( x == p - 1 ):
                break
            r = r + 1
            
        if( r == s ):
            return False
        else:
            continue
        
    return True

# Checks that elliptic curve parameters are correct
def verifyCurveParameters(a, b, p, n, x, y):

    # Verify P is prime
    if not isPrime(p):
        return False

    # Verify n is order of the group
    curve = ECCurve(a, b, p, n, x, y)
    bp = curve.bp

    p1 = curve.multiply(n, bp)
    if not p1.isPointAtInfinity():
        return False

    return True

class ECCurve:
    a = 0
    b = 0
    p = 0
    n = 0
    bp = None

    def __init__(self, a, b, p, n, x, y):
        self.a = a
        self.b = b
        self.p = p
        self.n = n
        self.bp = ECPoint(x, y)

    def add(self, p, q):
        
        if( p.isPointAtInfinity() ):
            return q

        if( q.isPointAtInfinity() ):
            return p

        # P + Q = 2P where P == Q
        if( p.x == q.x and p.y == q.y ):
            return ECCurve.double(self, p)
       
        # P + Q = 0 where Q == -P 
        if( p.x == q.x ):
            return ECPoint()
       
        rise = (p.y - q.y) % self.p
        run = (p.x - q.x) % self.p
        s = ( rise * invModP( run, self.p ) ) % self.p
        r = ECPoint()
        r.x = ( s * s - p.x - q.x ) % self.p
        r.y = ( -p.y + s * (p.x - r.x ) ) % self.p
        
        return r

    # Performs EC point doubling
    def double(self, point):
        r = ECPoint()

        if( point.y == 0 ):
            return r
      
        # 3 * p.x 
        t = (point.x * point.x) % self.p
        px3 = ((t << 1) + t) % self.p

        # calculate the slope
        rise = (px3 + self.a) % self.p
        run = (point.y << 1) % self.p
        s = ( rise * invModP( run, self.p ) ) % self.p
     
        r.x = ( s * s - ( point.x << 1 ) ) % self.p
        r.y = ( -point.y + s * ( point.x - r.x ) ) % self.p
       
        return r

    # Multiply an EC point by a scalar
    def multiply(self, k, point):
        y = k
        r = ECPoint()
      
        while( y > 0 ):
            if( y & 1 == 1 ):
                r = ECCurve.add(self, r, point)
            y = y >> 1
            point = ECCurve.double(self, point)

        return r

    def verifyPoint(self, point ):
        y1 = (point.y * point.y) % self.p
        y2 = (pow(point.x,3,self.p) + self.a * point.x + self.b) % self.p

        if( y1 == y2 ):
            return True
        else:
            return False
