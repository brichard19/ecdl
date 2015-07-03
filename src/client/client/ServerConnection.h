#ifndef _SERVER_H
#define _SERVER_H

#include <string>
#include "BigInteger.h"

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

class DistinguishedPoint {

public:
    DistinguishedPoint(BigInteger &a, BigInteger &b, BigInteger &x, BigInteger &y)
    {
        this->a = a;
        this->b = b;
        this->x = x;
        this->y = y;
    }

    BigInteger a;
    BigInteger b;
    BigInteger x;
    BigInteger y;
};

/**
 * Stores all the values that the client sends to the server when it
 * finds a distinguished point
 */

class ServerConnection {

private:
    unsigned short _port;
    std::string _host;
    std::string _url;
public:
    ServerConnection(std::string, unsigned short port);
    int getStatus(std::string id);
    ParamsMsg getParameters(std::string id);
    void submitPoints(std::string id, std::vector<DistinguishedPoint> &points);
};

#endif
