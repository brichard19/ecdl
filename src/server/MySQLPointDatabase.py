from PointDatabase import PointDatabase, PointDatabaseConnection
from util import ECDLPParams
import MySQLdb
import hashlib

ECDL_DB_NAME = 'ecdl'

def toHex(n):
    return hex(n).rstrip("L").lstrip("0x") or "0"

def hashPoint(x, y):
    s = toHex(x) + '-' + toHex(y)
    return hashlib.sha256(s).hexdigest()

def connectToSQLServer(creds):
    return MySQLdb.connect(creds.hostName, creds.userName, creds.password)

def connectToSQLDatabase(creds, dbName):
    return MySQLdb.connect(creds.hostName, creds.userName, creds.password, dbName)

'''
Stores database credentials
'''
class DBCredentials:
    hostName = ""
    userName = ""
    password = ""

    def __init__(self, hostName, userName, password):
        self.hostName = hostName
        self.userName = userName
        self.password = password

class MySQLPointDatabase:
    db = None
    creds = None

    def __init__(self, hostName, userName, password):
        self.creds = DBCredentials(hostName, userName, password)

        self.initDatabase()

    def insertRPoint(self, cursor, name, idx, a, b, x, y):
        xHex = toHex(x)
        yHex = toHex(y)
        aHex = toHex(a)
        bHex = toHex(b)

        s = "INSERT INTO RPOINTS(NAME, IDX, A, B, X, Y) VALUES('%s', %d, '%s', '%s', '%s', '%s');" % (name, idx, aHex, bHex, xHex, yHex)
        cursor.execute(s)

    def insertParams(self, cursor, name, p, a, b, n, gx, gy, qx, qy, dBits):
        pHex = toHex(p)
        aHex = toHex(a)
        bHex = toHex(b)
        nHex = toHex(n)
        gxHex = toHex(gx)
        gyHex = toHex(gy)
        qxHex = toHex(qx)
        qyHex = toHex(qy)

        s = "INSERT INTO PARAMS(NAME, P, A, B, N, GX, GY, QX, QY, DBITS) VALUES('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', %d);" % (name, pHex, aHex, bHex, nHex, gxHex, gyHex, qxHex, qyHex, dBits)
        cursor.execute(s)

    def createPointsTable(self, cursor, name):
        s = "CREATE TABLE IF NOT EXISTS %s(HASH CHAR(64) NOT NULL, START_A VARCHAR(256) NOT NULL, START_B VARCHAR(256) NOT NULL, END_X VARCHAR(256) NOT NULL, END_Y VARCHAR(256) NOT NULL, PRIMARY KEY(HASH));" % (name)

        results = cursor.execute(s)

        return results

    def getConnection(self, name):
        return MySQLPointDatabaseConnection(self.creds, name)

    '''
    Creates the database if it does not already exist
    '''
    def initDatabase(self):
        db = connectToSQLServer(self.creds)

        dbName = ECDL_DB_NAME
        cursor = db.cursor()

        # If the database exists, use it, otherwise create it
        cursor.execute("CREATE DATABASE IF NOT EXISTS " + dbName + ";")
        cursor.execute("USE " + dbName + ";")

        # Create the R-points table
        s = "CREATE TABLE IF NOT EXISTS RPOINTS(NAME VARCHAR(32) NOT NULL, IDX INTEGER NOT NULL, A VARCHAR(256) NOT NULL, B VARCHAR(256) NOT NULL, X VARCHAR(256) NOT NULL, Y VARCHAR(256) NOT NULL);"
        cursor.execute(s)

        # Create params table
        s = "CREATE TABLE IF NOT EXISTS PARAMS(NAME VARCHAR(32) NOT NULL, P VARCHAR(256) NOT NULL, A VARCHAR(256) NOT NULL, B VARCHAR(256) NOT NULL, N VARCHAR(256) NOT NULL, GX VARCHAR(256) NOT NULL, GY VARCHAR(256) NOT NULL, QX VARCHAR(256) NOT NULL, QY VARCHAR(256) NOT NULL, DBITS INTEGER NOT NULL);"
        cursor.execute(s) 
        db.close()


    def getNames(self):
        db = connectToSQLDatabase(self.creds, ECDL_DB_NAME)

        cursor = db.cursor()

        s = "SELECT NAME FROM PARAMS;";
        cursor.execute(s)

        names = []
        for n in cursor:
            names.append(n[0])

        db.close()

        return names

    '''
    Copies the parameters and R-points to the database. Creates a table to store distinguished points
    '''
    def createContext(self, name, params, rPoints):
        db = connectToSQLDatabase(self.creds, ECDL_DB_NAME)
        cursor = db.cursor()

        for i in xrange(len(rPoints)):
            a = rPoints[i]['a']
            b = rPoints[i]['b']
            x = rPoints[i]['x']
            y = rPoints[i]['y']
            self.insertRPoint(cursor, name, i, a, b, x, y)

        self.insertParams(cursor, name, params.p, params.a, params.b, params.n, params.gx, params.gy, params.qx, params.qy, params.dBits)

        self.createPointsTable(cursor, name)

        db.close()


class MySQLPointDatabaseConnection(PointDatabaseConnection):

    def __init__(self, creds, name):
        self.creds = creds
        self.name = name
        self.db = None

    def open(self):
        self.db = connectToSQLDatabase(self.creds, ECDL_DB_NAME)

    def close(self):
        self.db.commit()
        self.db.close()

    def getSize(self):
        cursor.execute("SELECT TABLE_ROWS from information_schema.tables where TABLE_SCHEMA = '%s' and TABLE_NAME = '%s'" % (ECDL_DB_NAME, self.name))
        results = cursor.fetchOne()

        count = int(results[0])

        return count

    def createInsertString(self, a, b, x, y):
        aHex = toHex(a)
        bHex = toHex(b)
        xHex = toHex(x)
        yHex = toHex(y)
        h = hashPoint(x, y)

        return "INSERT INTO %s(HASH, START_A, START_B, END_X, END_Y) VALUES('%s', '%s', '%s', '%s', '%s');" % (self.name, h, aHex, bHex, xHex, yHex)

    def createReadString(self, x, y):
        h = hashPoint(x, y)

        return "SELECT * FROM %s WHERE HASH='%s' LIMIT 1;" % (self.name,h)

    def getParams(self):
        cursor = self.db.cursor()

        s = "SELECT P, A, B, N, GX, GY, QX, QY, DBITS FROM PARAMS WHERE NAME='%s';" % (self.name)

        cursor.execute(s)

        (p, a, b, n, gx, gy, qx, qy, dBits) = cursor.fetchone()

        params = ECDLPParams()
        params.p = int(p, 16)
        params.a = int(a, 16)
        params.b = int(b, 16)
        params.n = int(n, 16)
        params.gx = int(gx, 16)
        params.gy = int(gy, 16)
        params.qx = int(qx, 16)
        params.qy = int(qy, 16)
        params.dBits = dBits
        params.field = "prime"

        return params

    def getRPoints(self):
        cursor = self.db.cursor()

        s = "SELECT A, B, X, Y FROM RPOINTS WHERE NAME='%s';" % (self.name)

        cursor.execute(s)
        rPoints = []
        for (a, b, x, y) in cursor:
            r = {}
            r['a'] = int(a,16)
            r['b'] = int(b,16)
            r['x'] = int(x,16)
            r['y'] = int(y,16)
            rPoints.append(r)

        return rPoints

    '''
    Insert distinguished point into database
    '''
    def insert(self, a, b, x, y):
        cursor = self.db.cursor()
        cursor.execute(self.createInsertString(a, b, x, y))

    '''
    Read a distinguished point from the database.
    Returns a dictinary containing:
    'a', 'b', 'x', 'y'
    '''
    def get(self, x, y):
        cursor = self.db.cursor()
        s = self.createReadString(x, y)
        cursor.execute(s)

        results = cursor.fetchone()

        if results == None:
            return None

        dp = {}
        dp['a'] = int(results[1], 16)
        dp['b'] = int(results[2], 16)
        dp['x'] = int(results[3], 16)
        dp['y'] = int(results[4], 16)

        return dp

    '''
    Check if the databse contains a particular distinguished point
    '''
    def contains(self, x, y):

        cursor = self.db.cursor()
        exists = True
        try:
            cursor.execute(createReadString(x))
        except:
            exists = False

        return exists
