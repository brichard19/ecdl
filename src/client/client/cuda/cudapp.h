#ifndef _CUDAPP_H
#define _CUDAPP_H

#include<cuda.h>
#include<cuda_runtime.h>

namespace CUDA {

    void memcpy(void *dest, const void *src, size_t count, enum cudaMemcpyKind kind);
    void *malloc(size_t);
    void *hostAlloc(size_t size, unsigned int flags );
    void free(void *ptr);
    void setDevice(int device);
    void setDeviceFlags(unsigned int flags);
    void *getDevicePointer(void *hostPtr, unsigned int flags);
    int getDeviceCount();
    void getDeviceInfo(int device,
                           unsigned int *deviceMajor,
                           unsigned int *deviceMinor,
                           unsigned int *mpCount,
                           unsigned long long *globalMemory);
}

#endif
