import random
import sys
import binascii
import math

def inv( x, m ):
    return pow(x, m-2, m)

class ECPoint:
    x = 0
    y = 0
    
    def __init__(self,x=0,y=0):
        self.x = x
        self.y = y

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
         
        # P + Q = 2P where P == Q
        if( p.x == q.x and p.y == q.y ):
            return ECCurve.double(self, p)
       
        # P + Q = 0 where Q == -P 
        if( p.x == q.x ):
            return ECPoint()
       
        # P + Q = Q where P == 0 
        if( p.x == 0 and p.y == 0 ):
            return q

        # P + Q == P where Q == 0
        if( q.x == 0 and q.y == 0 ):
            return p

        rise = (p.y - q.y) % self.p
        run = (p.x - q.x) % self.p
        s = ( rise * inv( run, self.p ) ) % self.p
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
        px3 = (t << 1) + t

        rise = px3 + self.a
        run = point.y << 1
       
        s = ( rise * inv( run, self.p ) ) % self.p
     
        r.x = ( s * s - ( point.x << 1 ) ) % self.p
        r.y = ( -point.y + s * ( point.x - r.x ) ) % self.p
       
        return r

    # Multiply an EC point by a scalar
    def multiply(self, k, point):
        y = k
        r = ECPoint()
      
        while( y > 0 ):
            if( y % 2 == 1 ):
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
