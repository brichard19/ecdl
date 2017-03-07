import json
from ecc import ECCurve, ECPoint
import os
import ecdl
import sys

import util
import time
import smtplib
from email.mime.text import MIMEText

'''
Inversion mod p using Fermat method
'''
def invm(x, m):
    y = x % m
    return pow(y, m-2, m)

def swap(a, b):
    return b, a

class RhoSolver:
    point1 = None
    p1Len = 0
    point2 = None
    p2Len = 0
    endPoint = None
    params = None
    curve = None
    a1 = None
    b1 = None
    a2 = None
    b2 = None
    rPoints = None
    solved = False
    k = None

    '''
    Constructs a new RhoSolver object
    '''
    def __init__(self, params, rPoints, a1, b1, a2, b2, endPoint):

        # Get params
        self.params = params
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2
        self.endPoint = endPoint
        self.rPoints = rPoints

        # Set up curve
        self.curve = ECCurve(params.a, params.b, params.p, params.n, params.gx, params.gy)

        # Compute starting points
        g = ECPoint(self.params.gx, self.params.gy)
        q = ECPoint(self.params.qx, self.params.qy)

        self.point1 = self.curve.add(self.curve.multiply(a1, g), self.curve.multiply(b1, q))
        self.point2 = self.curve.add(self.curve.multiply(a2, g), self.curve.multiply(b2, q))

        # Set up array of random walk points
        #self.rPoints = []
        #for rPoint in rPoints:
        #    self.rPoints.append(ECPoint(rPoint['x'], rPoint['y']))


    '''
    Checks if the end point in the first random walk is the
    start point of the second walk

    Robin Hood is a reference to character in English
    folklore hero who could shoot a second arrow on the
    exact trajectory as the first.
    '''
    def _isRobinHood(self):

        currentPoint = self.point1
        count = 0
        while True:
            idx = currentPoint.x & 0x1f

            if currentPoint.x == self.point2.x and currentPoint.y == self.point2.y:
                return True

            if currentPoint.x == self.endPoint.x and currentPoint.y == self.endPoint.y:
                return False

            # Iterate to next point
            currentPoint = self.curve.add(currentPoint, ECPoint(self.rPoints[idx]['x'], self.rPoints[idx]['y']))

    '''
    Gets the total length of the random walk
    '''
    def _getWalkLength(self, startA, startB, startPoint):

        point = startPoint
        a = startA
        b = startB
        i = 0
        length = 1

        # We want to terminate the walk if it is statistically too long
        limit = (2**self.params.dBits) * 4

        while True:
            idx = point.x & 0x1f
          
            length = length + 1

            if point.x == self.endPoint.x and point.y == self.endPoint.y:
                print("Found endpoint")
                return length
                break

            # Increment the point and coefficients
            r = ECPoint(self.rPoints[idx]['x'], self.rPoints[idx]['y'])
            point = self.curve.add(point, r)

            a = (a + self.rPoints[idx]['a']) % self.curve.n
            b = (b + self.rPoints[idx]['b']) % self.curve.n

            if i > limit:
                print("Walk is too long. Terminating")
                return -1

        return length



    '''
    Gets the next point in the random walk
    '''
    def _nextPoint(self, a, b, point):

        idx = point.x & 0x1f

        newPoint = self.curve.add(point, ECPoint(self.rPoints[idx]['x'], self.rPoints[idx]['y']))
        newA = (a + self.rPoints[idx]['a']) % self.curve.n
        newB = (b + self.rPoints[idx]['b']) % self.curve.n

        return newA, newB, newPoint


    def _countWalkLengths(self):

        print("Counting walk #1 length")
        self.p1Len = self._getWalkLength(self.a1, self.b1, self.point1)

        #if self.p1Len < 0:
        #    return None, None, None, None, None, None
        print(str(self.p1Len))

        print("Counting walk #2 length")
        self.p2Len = self._getWalkLength(self.a2, self.b2, self.point2)
       
        #if p2Len < 0:
        #    return None, None, None, None, None, None
        print(str(self.p2Len))

        # For simplicity, we want P1 to always be the longer one
        if self.p1Len < self.p2Len:
            self.a1, self.a2 = swap(self.a1, self.a2)
            self.b1, self.b2 = swap(self.b1, self.b2)
            self.point1, self.point2 = swap(self.point1, self.point2)
            self.p1Len, self.p2Len = swap(self.p1Len, self.p2Len)


    '''
    Given two walks with the same ending point, it finds where the walks collide.
    '''
    #curve, g, q, a1Start, b1Start, p1Start, a2Start, b2Start, p2Start, endPoint, rPoints, dBits):
    def _findCollision(self):

        self._countWalkLengths()

        print("Checking for Robin Hood")

        if self._isRobinHood():
            print("It's a Robin Hood :(")
            return None, None, None, None, None, None
        else:
            print("Not a Robin Hood :)") 

        point1 = self.point1
        point2 = self.point2

        diff = self.p1Len - self.p2Len
        print("Stepping " + str(diff) + " times")

        a1 = self.a1
        b1 = self.b1

        a2 = self.a2
        b2 = self.b2

        for i in xrange(diff):
            a1, b1, point1 = self._nextPoint(a1, b1, point1)

        print("Searching for collision")
        while True:

            if (point1.x == self.endPoint.x and point1.y == self.endPoint.y) or (point2.x == self.endPoint.x and point2.y == self.endPoint.y):
                print("Reached the end :(")
                return None

            a1Old = a1
            b1Old = b1
            point1Old = point1

            a2Old = a2
            b2Old = b2
            point2Old = point2

            a1, b1, point1 = self._nextPoint(a1, b1, point1)
            a2, b2, point2 = self._nextPoint(a2, b2, point2)

            if point1.x == point2.x and point1.y == point2.y:
                print("Found collision!")
                print(hex(a1) + " " + hex(b1) + " " + hex(point1.x) + " " + hex(point1.y))
                print(hex(a2) + " " + hex(b2) + " " + hex(point2.x) + " " + hex(point2.y))
                return a1, b1, point1, a2, b2, point2

    def solve(self):
        a1, b1, point1, a2, b2, point2 = self._findCollision()

        if a1 == None:
            self.solved = False
            return

        k = (((a1 - a2)%self.curve.n) * invm(b2 - b1, self.curve.n)) % self.curve.n
       
        # Verify 
        r = self.curve.multiply(k, self.curve.bp)

        if r.x != self.params.qx or r.y != self.params.qy:
            print("Verification failed")
            self.solved = False
        else:
            print("Verification successful")
            self.solved = True

        self.k = k

def sendNotificationEmail(ctx):
    print("Sending email to " + ctx.email)


def solveNextCollision(ctx):

    ctx.database.open()
    coll = ctx.database.getNextCollision()
    ctx.database.close()

    if coll == None:
        return

    solver = RhoSolver(ctx.params, ctx.rPoints, coll['a1'], coll['b1'], coll['a2'], coll['b2'], ECPoint(coll['x'], coll['y']))

    solver.solve()

    ctx.database.open()

    if solver.solved == True:
        print("The solution is " + util.toHex(solver.k))
        ctx.database.setSolution(solver.k)
        ctx.database.updateCollisionStatus(coll['id'], 'T')
    else:
        ctx.database.updateCollisionStatus(coll['id'], 'F')

    ctx.database.close()

def mainLoop():

    while True:

        contextNames = ecdl.Database.getNames()

        if contextNames != None and len(contextNames) != 0:

            # Get a list of unsolved contexts
            ctxList = []
            for name in contextNames:
                ctx = ecdl.loadContext(name)

                if ctx.status.lower() == 'unsolved' and ctx.collisions > 0:
                    ctxList.append(ctx)
                    solveNextCollision(ctx)

            # If anything is in the list, solve them
            if len(ctxList) > 0:
                for ctx in ctxList:
                    solveNextCollision(ctx)
            else:
                time.sleep(30)
        else:
            time.sleep(30)

'''
Program entry point
'''
def main():

    try:
        ecdl.loadConfig("config/config.json")
    except:
        print("Error opening config: " + sys.exc_info[0])
        sys.exit(1)

    mainLoop()


if __name__ == "__main__":
    main()
 