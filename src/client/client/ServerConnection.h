#ifndef _SERVER_H
#define _SERVER_H

#include <string>
#include "BigInteger.h"

#define DEFAULT_PORT 9999

enum {
    SERVER_STATUS_RUNNING,
    SERVER_STATUS_STOPPED
};

/**
 * Stores all the parameters that the server sends to the client
 */
class ParamsMsg {

public:
    unsigned int dBits;
    BigInteger p;
    BigInteger a;
    BigInteger b;
    BigInteger n;
    BigInteger gx;
    BigInteger gy;
    BigInteger qx;
    BigInteger qy;
    BigInteger rx[32];
    BigInteger ry[32];
};

/**
 * Represents a distinguished point
 */
class DistinguishedPoint {

public:
    DistinguishedPoint(BigInteger &a, BigInteger &b, BigInteger &x, BigInteger &y, unsigned int length)
    {
        this->a = a;
        this->b = b;
        this->x = x;
        this->y = y;
        this->length = length;
    }

    BigInteger a;
    BigInteger b;
    BigInteger x;
    BigInteger y;
    unsigned int length;
};

/**
 * Represents a connection the server
 */
class ServerConnection {

private:
    int _port;
    std::string _host;
    std::string _url;

public:
    ServerConnection(std::string host, int port=DEFAULT_PORT);

    int getStatus(std::string id);
    ParamsMsg getParameters(std::string id);
    void submitPoints(std::string id, std::vector<DistinguishedPoint> &points);
};

#endif
