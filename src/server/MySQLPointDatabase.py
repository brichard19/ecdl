from PointDatabase import PointDatabase
import MySQLdb
import hashlib

def toHex(n):
    return hex(n).rstrip("L").lstrip("0x") or "0"

class MySQLPointDatabase(PointDatabase):
    CREATE_POINTS_TABLE_STRING = "CREATE TABLE IF NOT EXISTS POINTS(HASH CHAR(64) NOT NULL, START_A VARCHAR(256) NOT NULL, START_B VARCHAR(256) NOT NULL, END_X VARCHAR(256) NOT NULL, END_Y VARCHAR(256) NOT NULL, PRIMARY KEY(HASH))"
    CREATE_COLLISIONS_TABLE_STRING = "CREATE TABLE IF NOT EXISTS COLLISIONS(START_A VARCHAR(256) NOT NULL, START_B VARCHAR(256) NOT NULL, END_X VARCHAR(256) NOT NULL, END_Y VARCHAR(256) NOT NULL)"
    hostName = ""
    userName = ""
    password = ""
    dbName = ""

    def getConnection(self):
        conn = self.connectToDB()

        return conn

    def closeConnection(self, conn):
        conn.commit()
        conn.close()

    def sha256(self, s):
        return hashlib.sha256(s).hexdigest()

    def toHex(self, n):
        return hex(n).rstrip("L").lstrip("0x") or "0"

    def createInsertString(self, a, b, x, y):
        aHex = toHex(a)
        bHex = toHex(b)
        xHex = toHex(x)
        yHex = toHex(y)
        h = self.sha256(xHex + yHex)

        return "INSERT INTO POINTS(HASH, START_A, START_B, END_X, END_Y) VALUES('%s', '%s', '%s', '%s', '%s');" % (h, aHex, bHex, xHex, yHex)

    def createReadString(self, x, y):
        xHex = toHex(x)
        yHex = toHex(y)
        h = self.sha256(xHex + yHex)

        return "SELECT * FROM POINTS WHERE HASH='%s';" % (h)

    def __init__(self, hostName, userName, password, dbName):
        self.hostName = hostName
        self.userName = userName
        self.password = password
        self.dbName = dbName

        # Create database and tables
        db = self.connectToServer()
        cursor = db.cursor()

        # If the database exists, use it, otherwise create it
        cursor.execute("CREATE DATABASE IF NOT EXISTS " + dbName + ";")
        cursor.execute("USE " + dbName + ";")

        cursor.execute(self.CREATE_POINTS_TABLE_STRING)

    '''
    Insert distinguished point into database
    '''
    def insert(self, db, a, b, x, y):
        cursor = db.cursor()
        cursor.execute(self.createInsertString(a, b, x, y))

    '''
    Read a distinguished point from the database.
    Returns a dictinary containing:
    'a', 'b', 'x', 'y',
    '''
    def get(self, db, x, y):
        cursor = db.cursor()
        cursor.execute(self.createReadString(x,y))

        results = cursor.fetchone()

        if results == None:
            return None

        dp = {}
        dp['a'] = int(results[1], 16)
        dp['b'] = int(results[2], 16)
        dp['x'] = int(results[3], 16)
        dp['y'] = int(results[4], 16)
        db.close()

        return dp

    '''
    Check if the databse contains a particular distinguished point
    '''
    def contains(self, db, x, y):

        cursor = db.cursor()
        exists = True
        try:
            cursor.execute(createReadString(x))
        except:
            exists = False

        return exists

    '''
    Open connection to the server
    '''
    def connectToServer(self):
        return MySQLdb.connect(self.hostName, self.userName, self.password)

    '''
    Open connection to database
    '''
    def connectToDB(self):
        db = MySQLdb.connect(self.hostName, self.userName, self.password, self.dbName)
        return db
