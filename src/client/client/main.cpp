#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include "client.h"
#include "util.h"
#include "logger.h"
#include "threads.h"
#include "ServerConnection.h"
#include "config.h"
#include "client.h"
#include "ECDLContext.h"
#include "ecc.h"

#ifdef _CUDA
#include "ECDLCuda.h"
#else
#include "ECDLCPU.h"
#endif

ECDLContext *_context;

#define NUM_R_POINTS 32

// X and Y values for the random walk points
BigInteger _rx[NUM_R_POINTS];
BigInteger _ry[NUM_R_POINTS];

// Problem parameters
ECDLPParams _params;

// problem id
std::string _id;

Mutex _pointsMutex;

ServerConnection *_serverConnection = NULL;

// Declared extern in client.h
ClientConfig _config;

std::vector<DistinguishedPoint> _pointsCache;

bool _running = true;

/**
 * Sends points to server
 */
void sendPointsToServer()
{
    Logger::logInfo("Sending %d points to server", _pointsCache.size());
    try {
        _serverConnection->submitPoints(_id, _pointsCache);
    } catch(std::string err) {
        printf("Error sending points to server: %s\n", err.c_str());
        return;
    }
    _pointsCache.clear();
}

bool verifyPoint(BigInteger &x, BigInteger &y)
{
    ECCurve curve(_params.p, _params.n, _params.a, _params.b, _params.gx, _params.gy);
    ECPoint p(x, y);

    return curve.pointExists(p);
}

/**
 * Adds distinguished point to cache
 */
void addPointToCache(BigInteger a, BigInteger b, BigInteger x, BigInteger y)
{
    DistinguishedPoint p(a, b, x, y);
    _pointsMutex.grab();
    _pointsCache.push_back(p);
    _pointsMutex.release();
    
}

void callback(struct CallbackParameters *p)
{
    // TODO: Should be in new thread so worker thread is not blocked

    // Check if point is valid
    if(!verifyPoint(p->x, p->y)) {
        printf("INVALID POINT\n");
        printf("a: %s\n", p->aStart.toString(16).c_str());
        printf("b: %s\n", p->bStart.toString(16).c_str());
        printf("x: %s\n", p->x.toString(16).c_str());
        printf("y: %s\n", p->y.toString(16).c_str());
        printf("\n\n" );
        return;
    }
    addPointToCache(p->aStart, p->bStart, p->x, p->y);
}

/**
 * Gets the problem parameters and R points from the server
 */
bool getParameters(ECDLPParams &params, BigInteger *rx, BigInteger *ry)
{
    ParamsMsg paramsMsg;

    try {
        paramsMsg = _serverConnection->getParameters(_id);
    } catch(std::string e) {
        printf("Error: %s\n", e.c_str()); 
        return false;
    }

    params.p = paramsMsg.p;
    params.n = paramsMsg.n;
    params.a = paramsMsg.a;
    params.b = paramsMsg.b;
    params.gx = paramsMsg.gx;
    params.gy = paramsMsg.gy;
    params.qx = paramsMsg.qx;
    params.qy = paramsMsg.qy;
    params.dBits = paramsMsg.dBits;
    
    for(int i = 0; i < 32; i++) {
        rx[ i ] = paramsMsg.rx[i];
        ry[ i ] = paramsMsg.ry[i];
    }

    return true;
}

/**
 * Thread to poll the cache of distinguished points. Will send them to the server
 * when there are enough
 */
void *sendPointsThread(void *p)
{

    while(_running) {
        _pointsMutex.grab();

        if(_pointsCache.size() >= _config.pointCacheSize) {
            Logger::logInfo("Sending %d points to server", _pointsCache.size());
            bool success = true;

            try {
                _serverConnection->submitPoints(_id, _pointsCache);
            } catch(std::string err) {
                success = false;
                printf("Error sending points to server: %s. Will try again later\n", err.c_str());
            }

            if(success) {
                _pointsCache.clear();
            }
        }
        _pointsMutex.release();

        sleep(30);
    }
}

/**
 * Holding thread for running the context
 */
void *runningThread(void *p)
{
    // This is a blocking call
   _context->run();

   return NULL;
}

/**
 * Main loop of the program. It periodically polls the server
 */
void pollConnections()
{
    _running = true;

    Thread pointsThread(sendPointsThread, NULL);

    while(_running) {

        unsigned int status = 0;
      
        // Attempt to connect to the server 
        try {
            status = _serverConnection->getStatus(_id);
        }catch(std::string s) {
            printf("Connection error: %s\n", s.c_str());
            printf("Retrying in 60 seconds...\n");
            fflush(stdout); 
            sleep(60);
            continue;
        }

        // If not currently running, then get the parameters and start
        if(status == SERVER_STATUS_RUNNING) {
            if(_context == NULL) {

                // Get parameters from the server
                if(!getParameters(_params, _rx, _ry)) {
                    printf("Error getting the parameters from server\n");
                } else {

                    printf("Received parameters from server\n");
                    printf("GF(p) = %s\n", _params.p.toString().c_str());
                    printf("y^2 = x^3 + %sx + %s\n", _params.a.toString().c_str(), _params.b.toString().c_str());
                    printf("G = [%s, %s]\n", _params.gx.toString().c_str(), _params.gy.toString().c_str());
                    printf("Q = [%s, %s]\n", _params.qx.toString().c_str(), _params.qy.toString().c_str());

                    #ifdef _CUDA
                    _context = new ECDLCudaContext(_config.device, _config.blocks, _config.threads, _config.pointsPerThread, &_params, _rx, _ry, NUM_R_POINTS, callback);
                    #endif
                    
                    #ifdef _CPU
                    _context = new ECDLCpuContext(_config.threads, _config.pointsPerThread, &_params, _rx, _ry, NUM_R_POINTS, callback);
                    #endif

                    _context->init();

                    Thread t(runningThread, NULL);
                }
            } else if(!_context->isRunning()) {
                Thread t(runningThread, NULL);
            }
        } else {
            printf("Stopping\n"); 
            if(_context != NULL) {
                _context->stop();
            }
        }

        // Sleep 5 minutes
        sleep(300);
    }
}

/**
 * Program entry point
 */
int main(int argc, char **argv)
{
    // TODO: Use proper RNG
    srand(util::getSystemTime());

    // Check for CUDA devices
#ifdef _CUDA
    unsigned int major;
    unsigned int minor;
    unsigned int mpCount;
    unsigned long long mem;

    if(CUDA::getDeviceCount() == 0) {
        Logger::logError("No CUDA devices detected\n");
        return 1;
    }

    // Get device info
    try {
        CUDA::getDeviceInfo(0, &major, &minor, &mpCount, &mem);
    }catch(cudaError_t cudaError) { 
        Logger::logError("Error getting device info: %s\n", cudaGetErrorString(cudaError));
        return 1;
    }
  
    Logger::logInfo("Device info:");
    Logger::logInfo("version:  %d.%d", major, minor);
    Logger::logInfo("MP count: %d", mpCount);
    Logger::logInfo("Memory:   %lldMB", mem/1048576);
    Logger::logInfo("");
#endif
    
    // Load configuration
    try {
        _config = loadConfig("settings.json");
    }catch(std::string err) {
        Logger::logError("Error loading settings: " + err);
        return 1; 
    }

//TODO: Properly parse arguments (getopt?)
    // Run benchmark on -b option
    if(argc >= 2 && strcmp(argv[1], "-b") == 0) {
        doBenchmark();
        return 0;
    } else {
        if(argc < 2) {
            printf("usage: [options] id\n");
            return 0;
        } else {
            _id = std::string(argv[1]);
        }
    }

    // Enter main loop
    _serverConnection = new ServerConnection(_config.serverHost, _config.serverPort);
    pollConnections();

    return 0;
}
