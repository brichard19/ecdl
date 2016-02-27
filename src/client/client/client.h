#ifndef _CLIENT_H
#define _CLIENT_H

#include <string>

typedef struct {

    std::string serverHost;
    unsigned short serverPort;
    unsigned int pointCacheSize;

#ifdef _CUDA
    int device;
    int blocks;
    int threads;
    int pointsPerThread;
    int totalPoints;
#else
    int threads;
    int pointsPerThread;
#endif

}ClientConfig;


extern ClientConfig _config;

ClientConfig loadConfig(std::string fileName);
void doBenchmark();

#endif
