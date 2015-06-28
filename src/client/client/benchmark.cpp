#include "client.h"

#ifdef _CUDA
#include "ECDLCuda.h"
#include "kernels.h"
#include "cudapp.h"
#else
#include "ECDLCPU.h"
#endif

#include "BigInteger.h"
#include "logger.h"

#include "util.h"

void doBenchmark()
{
    #ifdef _CUDA
    ECDLCudaContext *ctx;
    #else
    ECDLCpuContext *ctx;
    #endif

    ECDLPParams params[1];
    BigInteger rx[ 32 ];
    BigInteger ry[ 32 ];
    params[0].p = BigInteger("48E1D43F293469E33194C43186B3ABC0B", 16);
    params[0].a = BigInteger("41CB121CE2B31F608A76FC8F23D73CB66", 16);
    params[0].b = BigInteger("2F74F717E8DEC90991E5EA9B2FF03DA58", 16);
    params[0].n = BigInteger("48E1D43F293469E317F7ED728F6B8E6F1", 16);
    params[0].gx = BigInteger("03DF84A96B5688EF574FA91A32E197198A", 16);
    params[0].gy = BigInteger("014721161917A44FB7B4626F36F0942E71", 16);
    params[0].qx = BigInteger("03AA6F004FC62E2DA1ED0BFB62C3FFB568", 16);
    params[0].qy = BigInteger("009C21C284BA8A445BB2701BF55E3A67ED", 16);
    params[0].dBits = 32;

    ECCurve curve(params[0].p, params[0].n, params[0].a, params[0].b, params[0].gx, params[0].gy);
    generateRPoints(curve, ECPoint(params[0].qx, params[0].qy), NULL, NULL, rx, ry, 32);

    Logger::logInfo("Running benchmark...");
    #ifdef _CUDA
    ctx = new ECDLCudaContext(_config.device, _config.blocks, _config.threads, _config.pointsPerThread, &params[0], rx, ry, 32, NULL);
    ctx->init();
    ctx->benchmark(NULL);
    #else
    ctx = new ECDLCpuContext(_config.threads, _config.pointsPerThread, &params[0], rx, ry, 32, NULL);
    ctx->init();
    ctx->benchmark(NULL);
    #endif
}
