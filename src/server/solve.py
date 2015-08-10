import json
from ecc import ECCurve, ECPoint
import os
import ecdl
import sys

NUM_R_POINTS = 32
WORKDIR = './work'

def getContext(id):
    if id in _ctx:
        return _ctx[id]

    return None

'''
Converts a string to integer by guessing the base
'''
def parseInt(n):
    return int(n, 0)

'''
Verify that a point is on the curve
'''
def verifyPoint(curve, g, q, endA, endB, endPoint):

    # Check that end point exists
    if not curve.verifyPoint(endPoint):
        return False

    return True

'''
Verifies that one random walk does not walk into the
start of the other one.

Robin Hood is a reference to character in English
folklore hero who could shoot a second arrow on the
exact trajectory as the first.
'''
def checkForRobinHood(curve, start, end, start2, rPoints):

    point = start
    count = 0
    while True:
        idx = point.x & 0x1f

        if point.x == start2.x and point.y == start2.y:
            return True

        if point.x == end.x and point.y == end.y:
            return False

        point = curve.add(point, ECPoint(rPoints[idx]['x'], rPoints[idx]['y']))

'''
Gets the total length of the random walk
'''
def getLength(curve, startA, startB, startPoint, endPoint, rPoints, dBits):

    point = startPoint
    a = startA
    b = startB
    points = []
    i = 0
    length = 1

    # We want to terminate the walk if it is statistically too long
    limit = (2**dBits) * 4
    while True:
        idx = point.x & 0x1f
      
        length = length + 1

        if point.x == endPoint.x and point.y == endPoint.y:
            print("Found endpoint")
            return length
            break

        # Increment the point and coefficients
        r = ECPoint(rPoints[idx]['x'], rPoints[idx]['y'])
        point = curve.add(point, r)
        a = (a + rPoints[idx]['a']) % curve.n
        b = (b + rPoints[idx]['b']) % curve.n

        if i > limit:
            print("Walk is too long. Terminating")
            return -1

    return length

def invm(x, m):
    y = x % m
    return pow(y, m-2, m)

def swap(a, b):
    return b, a

'''
Gets the next point in the random walk
'''
def nextPoint(curve, a, b, point, rPoints):

    idx = point.x & 0x1f

    newPoint = curve.add(point, ECPoint(rPoints[idx]['x'], rPoints[idx]['y']))
    newA = (a + rPoints[idx]['a']) % curve.n
    newB = (b + rPoints[idx]['b']) % curve.n

    return newA, newB, newPoint


'''
Given two walks with the same ending point, it finds where the walks collide.
'''
def findCollision(curve, g, q, a1Start, b1Start, p1Start, a2Start, b2Start, p2Start, endPoint, rPoints, dBits):

    a1 = a1Start
    b1 = b1Start

    a2 = a2Start
    b2 = b2Start

    print("Counting walk 1 length")
    p1Len = getLength(curve, a1, b1, p1Start, endPoint, rPoints, dBits)

    if p1Len < 0:
        return None, None, None, None, None, None
    print(str(p1Len))

    print("Counting walk 2 length")
    p2Len = getLength(curve, a2, b2, p2Start, endPoint, rPoints, dBits)
   
    if p2Len < 0:
        return None, None, None, None, None, None
    print(str(p2Len))

    # For simplicity, we want P1 to always be the longer one
    if p1Len < p2Len:
        a1, a2 = swap(a1, a2)
        b1, b2 = swap(b1, b2)
        p1Start, b2Start = swap(p1Start, p2Start)
        p1Len, p2Len = swap(p1Len, p2Len)

    print("Checking for Robin Hood")

    if checkForRobinHood(curve, p1Start, endPoint, p2Start, rPoints):
        print("It's a Robin Hood :(")
        return None, None, None, None, None, None
    else:
        print("Not a Robin Hood :)") 

    point1 = p1Start
    point2 = p2Start

    diff = p1Len - p2Len
    print("Stepping " + str(diff) + " times")
    for i in xrange(diff):
        a1, b1, point1 = nextPoint(curve, a1, b1, point1, rPoints)

    print("Searching for collision")
    while True:

        if (point1.x == endPoint.x and point1.y == endPoint.y) or (point2.x == endPoint.x and point2.y == endPoint.y):
            print("Reached the end :(")
            return None

        a1Old = a1
        b1Old = b1
        point1Old = point1

        a2Old = a2
        b2Old = b2
        point2Old = point2

        a1, b1, point1 = nextPoint(curve, a1Old, b1Old, point1Old, rPoints)
        a2, b2, point2 = nextPoint(curve, a2Old, b2Old, point2Old, rPoints)

        if point1.x == point2.x and point1.y == point2.y:
            print("Found collision!")
            print(hex(a1) + " " + hex(b1) + " " + hex(point1.x) + " " + hex(point1.y))
            print(hex(a2) + " " + hex(b2) + " " + hex(point2.x) + " " + hex(point2.y))
            return a1, b1, point1, a2, b2, point2

'''
Program entry point
'''
def main():

    if len(sys.argv) != 8:
        print("Usage: a1 b1 a2 b2 x y job")
        exit()

    a1 = parseInt(sys.argv[1])
    b1 = parseInt(sys.argv[2])
    a2 = parseInt(sys.argv[3])
    b2 = parseInt(sys.argv[4])
    x = parseInt(sys.argv[5])
    y = parseInt(sys.argv[6])

    name = sys.argv[7]
   
    try:
        ecdl.loadConfig("config/config.json")
    except:
        print("Error opening config: " + sys.exc_info[0])
        sys.exit(1)

    ctx = ecdl.loadContext(name)
    curve = ctx.curve
    rPoints = ctx.rPoints
    dBits = ctx.params.dBits

    g = ECPoint(ctx.params.gx, ctx.params.gy)
    q = ECPoint(ctx.params.qx, ctx.params.qy)

    endPoint = ECPoint(x, y)

    p1Start = curve.add(curve.multiply(a1, g), curve.multiply(b1, q))
    p2Start = curve.add(curve.multiply(a2, g), curve.multiply(b2, q))

    a1, b1, point1, a2, b2, point2 = findCollision(curve, g, q, a1, b1, p1Start, a2, b2, p2Start, endPoint, rPoints, dBits)

    if a1 == None:
        return

    k = (((a1 - a2)%curve.n) * invm(b2 - b1, curve.n)) % curve.n
    r = curve.multiply(k, g)
    print("Q=   " + hex(q.x) + " " + hex(q.y))
    print("kG = " + hex(r.x) + " " + hex(r.y))
    print("") 
    print("k=" + hex(k))

if __name__ == "__main__":
    main()
