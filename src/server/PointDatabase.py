'''
Abstract class for hashtable to storing distinguished points
'''
class PointDatabase:

    '''
    Check if the database contains a distinguished point with the given
    x value
    '''
    def contains(x):
        raise NotImplementedError("This should be implemented in subclass")

    '''
    Insert a distinguished point into the database
    a: 'a' value of the starting point
    b: 'b' value of the starting point
    x: 'x' value of the distinguished point
    y: 'y' value of the distinguished point
    '''
    def insert(a, b, x, y):
        raise NotImplementedError("This should be implemented in subclass")

    '''
    Look up a distinguished point based on the x value
    Returns an object containing a, b, x, y
    '''
    def get(x):
        raise NotImplementedError("This should be implemented in subclass")
