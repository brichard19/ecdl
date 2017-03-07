from PointDatabase import PointDatabase, PointDatabaseConnection
from util import ECDLPParams
import util
import MySQLdb
import warnings

# Don't display 
warnings.filterwarnings('ignore', category = MySQLdb.Warning)

ECDL_DB_NAME = 'ecdl'

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

'''
Implementation of PointDatabase using MySQL database
'''
class MySQLPointDatabase:
    db = None
    creds = None

    def __init__(self, hostName, userName, password):
        self.creds = DBCredentials(hostName, userName, password)

        self.initDatabase()

    '''
    Creates a new record in the Info table
    '''
    def _initializeInfo(self, cursor, name, email):
        s = ("INSERT INTO JobInfo(Name, NotificationEmail, Status, Solution) VALUES('%s', '%s', 'unsolved', '')") % (Name, Email)

        cursor.execute(s)

    '''
    Inserts a distinguished point into the table
    '''
    def insertRPoint(self, cursor, name, idx, a, b, x, y):
        xHex = util.toHex(x)
        yHex = util.toHex(y)
        aHex = util.toHex(a)
        bHex = util.toHex(b)

        s = "INSERT INTO RPoints(Name, Idx, A, B, X, Y) VALUES('%s', %d, '%s', '%s', '%s', '%s');" % (name, idx, aHex, bHex, xHex, yHex)

        cursor.execute(s)

    '''
    Inserts parameters into the PARAMS table
    '''
    def insertParams(self, cursor, name, p, a, b, n, gx, gy, qx, qy, dBits):
        pHex = util.toHex(p)
        aHex = util.toHex(a)
        bHex = util.toHex(b)
        nHex = util.toHex(n)
        gxHex = util.toHex(gx)
        gyHex = util.toHex(gy)
        qxHex = util.toHex(qx)
        qyHex = util.toHex(qy)

        s = ("INSERT INTO JobParams(Name, P, A, B, N, Gx, Gy, Qx, Qy, DBits) "
            "VALUES('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', %d);") % (name, pHex, aHex, bHex, nHex, gxHex, gyHex, qxHex, qyHex, dBits)
        cursor.execute(s)
    
    '''
    Creates table to store distinguished points
    '''
    def createPointsTable(self, cursor, name):
        s = ("CREATE TABLE IF NOT EXISTS %s("
        "StartA VARCHAR(256) NOT NULL,"
        "StartB VARCHAR(256) NOT NULL,"
        "EndPoint VARCHAR(256) NOT NULL,"
        "WalkLength INT(10) UNSIGNED NOT NULL,"
        "PRIMARY KEY(EndPoint));") % (name)

        results = cursor.execute(s)

        return results



    '''
    Returns a data connection for the given context
    '''
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
        s = ("CREATE TABLE IF NOT EXISTS RPoints("
            "Name VARCHAR(32) NOT NULL,"
            "Idx INTEGER NOT NULL,"
            "A VARCHAR(256) NOT NULL,"
            "B VARCHAR(256) NOT NULL,"
            "X VARCHAR(256) NOT NULL,"
            "Y VARCHAR(256) NOT NULL);")
        cursor.execute(s)

        # Create params table
        s = ("CREATE TABLE IF NOT EXISTS JobParams("
            "Name VARCHAR(32) NOT NULL,"
            "P VARCHAR(256) NOT NULL,"
            "A VARCHAR(256) NOT NULL,"
            "B VARCHAR(256) NOT NULL,"
            "N VARCHAR(256) NOT NULL,"
            "Gx VARCHAR(256) NOT NULL,"
            "Gy VARCHAR(256) NOT NULL,"
            "Qx VARCHAR(256) NOT NULL,"
            "Qy VARCHAR(256) NOT NULL,"
            "DBITS INTEGER NOT NULL);")

        cursor.execute(s) 

        # Create table to store collisions
        s = ("CREATE TABLE IF NOT EXISTS Collisions("
            "Id INT NOT NULL AUTO_INCREMENT,"
            "Name VARCHAR(32) NOT NULL,"
            "A1 VARCHAR(256) NOT NULL,"
            "B1 VARCHAR(256) NOT NULL,"
            "WalkLength1 INT(10) UNSIGNED NOT NULL,"
            "A2 VARCHAR(256) NOT NULL,"
            "B2 VARCHAR(256) NOT NULL,"
            "WalkLength2 INT(10) UNSIGNED NOT NULL,"
            "X VARCHAR(256) NOT NULL,"
            "Y VARCHAR(256) NOT NULL,"
            "Checked VARCHAR(10) DEFAULT 'F',"
            "PRIMARY KEY(ID));")

        cursor.execute(s)
        s = ("CREATE TABLE IF NOT EXISTS JobInfo("
            "Name varchar(256) NOT NULL,"
            "NotificationEmail varchar(256) NULL,"
            "Status varchar(16) DEFAULT 'unsolved',"
            "Solution varchar(256) NULL);")

        cursor.execute(s)

        db.close()


    def insertInfo(self, cursor, name, email):

        s = "INSERT INTO JobInfo(Name, NotificationEmail) VALUES('%s', '%s');" % (name, email)

        cursor.execute(s)

    '''
    Gets a list of context names
    '''
    def getNames(self):
        db = connectToSQLDatabase(self.creds, ECDL_DB_NAME)

        cursor = db.cursor()

        s = "SELECT Name FROM JobParams;";
        cursor.execute(s)

        names = []
        for n in cursor:
            names.append(n[0])

        db.close()

        return names

    '''
    Copies the parameters and R-points to the database. Creates a table to store distinguished points
    '''
    def createContext(self, name, email, params, rPoints):
        db = connectToSQLDatabase(self.creds, ECDL_DB_NAME)
        cursor = db.cursor()

        for i in xrange(len(rPoints)):
            a = rPoints[i]['a']
            b = rPoints[i]['b']
            x = rPoints[i]['x']
            y = rPoints[i]['y']
            self.insertRPoint(cursor, name, i, a, b, x, y)

        self.insertParams(cursor, name, params.p, params.a, params.b, params.n, params.gx, params.gy, params.qx, params.qy, params.dBits)
        self.insertInfo(cursor, name, email)

        self.createPointsTable(cursor, name)


        db.close()


'''
Holds an open connection to the database
'''
class MySQLPointDatabaseConnection(PointDatabaseConnection):

    #Keep a copy of the params since they are needed for point compressin/decompression
    _params = None

    def __init__(self, creds, name):
        self.creds = creds
        self.name = name
        self.db = None

    def _createInsertString(self, a, b, x, y, length):
        aHex = util.toHex(a)
        bHex = util.toHex(b)
        endHex = util.compress_point(x, y)

        return "INSERT INTO %s(StartA, StartB, EndPoint, WalkLength) VALUES('%s', '%s', '%s', %d);" % (self.name, aHex, bHex, endHex, length)

    def _createMultipleInsertString(self, points):
        statement = "INSERT INTO %s(StartA, StartB, EndPoint, WalkLength) VALUES " % (self.name)

        for i in xrange(len(points)):
            aHex = util.toHex(points[i]['a'])
            bHex = util.toHex(points[i]['b'])
            endHex = util.compress_point(points[i]['x'], points[i]['y'])

            statement += "('%s', '%s', '%s', %d)" % (aHex, bHex, endHex, points[i]['length'])
            if i < len(points) - 1:
                statement += ","

        statement += ";"

        return statement;

    def _createReadString(self, x, y):
        point = util.compress_point(x, y)

        return "SELECT * FROM %s WHERE EndPoint='%s' LIMIT 1;" % (self.name, point)

    def _insert(self, cursor, a, b, x, y, length):
        cursor.execute(self._createInsertString(a, b, x, y, length))


    def _insertMultiple(self, cursor, points):

        try:
            cursor.execute(self._createMultipleInsertString(points))
        except MySQLdb.Error, e:
            coll = []

            for i in xrange(len(points)):
                try:
                    self._insert(cursor, points[i]['a'], points[i]['b'], points[i]['x'], points[i]['y'], points[i]['length'])
                except MySQLdb.Error, e2:
                    coll.append(points[i])

            return coll

        return None

    '''
    Open the connection to the database
    '''
    def open(self):
        self.db = connectToSQLDatabase(self.creds, ECDL_DB_NAME)
        self._params = self.getParams()

    '''
    Close the database connection
    '''
    def close(self):
        self.db.commit()
        self.db.close()

    '''
    Gets the number of distinguished points
    '''
    def getCount(self):
        cursor.execute("SELECT TABLE_ROWS from information_schema.tables where TABLE_SCHEMA = '%s' and TABLE_NAME = '%s'" % (ECDL_DB_NAME, self.name))
        results = cursor.fetchOne()

        count = int(results[0])

        return count

    '''
    Get the curve params for this context
    '''
    def getParams(self):
        cursor = self.db.cursor()

        s = "SELECT P, A, B, N, Gx, Gy, Qx, Qy, DBits FROM JobParams WHERE Name='%s';" % (self.name)

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

    '''
    Get the R points for this context
    '''
    def getRPoints(self):
        cursor = self.db.cursor()

        s = "SELECT A, B, X, Y FROM RPoints WHERE Name='%s';" % (self.name)

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

    def insertCollision(self, a1, b1, length1, a2, b2, length2, x, y):

        cursor = self.db.cursor()

        a1Hex = util.toHex(a1)
        b1Hex = util.toHex(b1) 
        a2Hex = util.toHex(a2)
        b2Hex = util.toHex(b2)
        xHex = util.toHex(x)
        yHex = util.toHex(y)

        s = "INSERT INTO Collisions(Name, A1, B1, WalkLength1, A2, B2, WalkLength2, X, Y) VALUES('%s', '%s', '%s', %d, '%s', '%s', %d, '%s', '%s');" % (self.name, a1Hex, b1Hex,length1, a2Hex, b2Hex, length2, xHex, yHex)
        cursor.execute(s)


    def setSolution(self, value):
        cursor = self.db.cursor()
        solutionHex = util.toHex(value)

        s = "UPDATE JobInfo SET Solution = '%s', Status = 'solved' WHERE Name = '%s';" % (solutionHex, self.name)
        cursor.execute(s)

    def updateCollisionStatus(self, collId, status):
       
        cursor = self.db.cursor()

        s = "UPDATE Collisions SET Checked='%s' WHERE Id=%d;" % (status, collId)
        cursor.execute(s)

    # Returns the next available collision
    def getNextCollision(self):
        cursor = self.db.cursor()

        s = "SELECT * FROM Collisions WHERE Name='%s' AND Checked != 'T' LIMIT 1;" % (self.name);

        cursor.execute(s)

        result = cursor.fetchone()

        if result == None:
            return None

        coll = {}
        coll['id'] = int(result[0])
        coll['a1'] = int(result[2],16)
        coll['b1'] = int(result[3],16)
        coll['length1'] = int(result[4])
        coll['a2'] = int(result[5],16)
        coll['b2'] = int(result[6],16)
        coll['length1'] = int(result[7])
        coll['x'] = int(result[8],16)
        coll['y'] = int(result[9],16)

        return coll

    '''
    Insert distinguished point into database
    '''
    def insert(self, a, b, x, y, length):
        cursor = self.db.cursor()
        self._insert(cursor, a, b, x, y, length)


    def insertMultiple(self, points):
        cursor = self.db.cursor()
        return self._insertMultiple(cursor, points)

    '''
    Read a distinguished point from the database.
    Returns a dictinary containing:
    'a', 'b', 'x', 'y'
    '''
    def get(self, x, y):
        print("get: " + str(x) + " " + str(y))
        cursor = self.db.cursor()
        s = self._createReadString(x, y)
        cursor.execute(s)

        results = cursor.fetchone()

        if results == None:
            return None

        dp = {}
        dp['a'] = int(results[0], 16)
        dp['b'] = int(results[1], 16)

        x1, y1 = util.decompress_point(results[2], self._params.a, self._params.b, self._params.p)

        dp['x'] = x1
        dp['y'] = y1
        dp['length'] = results[3];

        return dp

    def getStatus(self):
        cursor = self.db.cursor()
        s = "SELECT Status FROM JobInfo WHERE Name = '%s';" % (self.name)

        cursor.execute(s)
        result = cursor.fetchone()

        if result == None:
            return None

        return result[0]

    def getSolution(self):
        cursor = self.db.cursor()
        s = "SELECT Solution FROM JobInfo WHERE Name = '%s';" % (self.name)

        cursor.execute(s)
        result = cursor.fetchone()

        if result == None:
            return None

        if result[0] == None:
            return None

        return int(result[0], 16)

    def getNumCollisions(self):
        cursor = self.db.cursor()

        s = "SELECT count(*) as count FROM Collisions WHERE Name = '%s' AND Checked = 'F';" % (self.name);

        cursor.execute(s)

        result = cursor.fetchone()

        if result == None:
            return 0

        return result[0]

    '''
    Check if the databse contains a particular distinguished point
    '''
    def contains(self, x, y):

        cursor = self.db.cursor()
        exists = True
        try:
            cursor.execute(_createReadString(x,y))
        except:
            exists = False

        return exists
