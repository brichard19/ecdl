
class PointDatabase:

    def createContext(self, name, params, rPoints):
        raise NotImplementedError("This should be implemented in subclass")

    def getConnection(self, name):
        raise NotImplementedError("This should be implemented in subclass")

'''
Abstract class for hashtable to storing distinguished points
'''
class PointDatabaseConnection:

    '''
    Check if the database contains a distinguished point with the given
    x value
    '''
    def contains(self, x, y):
        raise NotImplementedError("This should be implemented in subclass")

    '''
    Insert a distinguished point into the database
    a: 'a' value of the starting point
    b: 'b' value of the starting point
    x: 'x' value of the distinguished point
    y: 'y' value of the distinguished point
    '''
    def insert(self, a, b, x, y):
        raise NotImplementedError("This should be implemented in subclass")

    '''
    Look up a distinguished point based on the x value
    Returns an object containing a, b, x, y
    '''
    def get(self, x, y):
        raise NotImplementedError("This should be implemented in subclass")

    def getSize():
        raise NotImplementedError("This should be implemented in subclass")


    def getRPoints(self):
        raise NotImplementedError("This shou,d be implemented in subclass")

    def getParams(self):
        raise NotImplementedError("This shou,d be implemented in subclass")

    def open(self):
        raise NotImplementedError("This shou,d be implemented in subclass")

    def close(self):
        raise NotImplementedError("This shou,d be implemented in subclass")
