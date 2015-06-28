from PointDatabase import PointDatabase
import MySQLdb

def toHex(n):
    return hex(n).rstrip("L").lstrip("0x") or "0"

class MySQLPointDatabase(PointDatabase):
    hostName = ""
    userName = ""
    password = ""
    dbName = ""

    def toHex(self, n):
        return hex(n).rstrip("L").lstrip("0x") or "0"

    def createTableString(self):
        return "CREATE TABLE IF NOT EXISTS POINTS(START_A VARCHAR(256) NOT NULL, START_B VARCHAR(256) NOT NULL, END_X VARCHAR(256) NOT NULL, END_Y VARCHAR(256) NOT NULL, COUNT BIGINT NOT NULL);";

    def createInsertString(self, a, b, x, y, count):
        aHex = toHex(a)
        bHex = toHex(b)
        xHex = toHex(x)
        yHex = toHex(y)

        return "INSERT INTO POINTS(START_A, START_B, END_X, END_Y, COUNT) VALUES('%s', '%s', '%s', '%s', '%d');" % (aHex, bHex, xHex, yHex, count)

    def createReadString(self, x, y):
        xHex = toHex(x)
        yHex = toHex(y)
        return "SELECT * FROM POINTS WHERE END_X='%s' AND END_Y='%s';" % (xHex, yHex)

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

        cursor.execute(self.createTableString())
        db.close()

    '''
    Insert distinguished point into database
    '''
    def insert(self, a, b, x, y, count):
        db = self.connectToDB()
        cursor = db.cursor()

        cursor.execute(self.createInsertString(a, b, x, y, count))
        db.commit()
        db.close()

    '''
    Read a distinguished point from the database.
    Returns a dictinary containing:
    'a', 'b', 'x', 'y', 'count'
    '''
    def get(self, x, y):
        db = self.connectToDB()
        cursor = db.cursor()
        cursor.execute(self.createReadString(x,y))

        results = cursor.fetchone()

        if results == None:
            return None

        dp = {}
        dp['a'] = int(results[0], 16)
        dp['b'] = int(results[1], 16)
        dp['x'] = int(results[2], 16)
        dp['y'] = int(results[3], 16)
        dp['count'] = int(results[4])
        db.close()

        return dp

    '''
    Check if the databse contains a particular distinguished point
    '''
    def contains(self, x, y):

        db = self.connectToDB()
        cursor = db.cursor()
        exists = True
        try:
            cursor.execute(createReadString(x))
        except:
            exists = False

        db.close()
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
        db.autocommit(True)
        return db
