#ifndef _CUDAPP_H
#define _CUDAPP_H

#include<cuda.h>
#include<cuda_runtime.h>
#include<string>

namespace CUDA {
    typedef struct {
      unsigned int major;
      unsigned int minor;
      unsigned int mpCount;
      unsigned int cores;
      unsigned long long globalMemory;
      std::string name;
    }DeviceInfo;

    void memcpy(void *dest, const void *src, size_t count, enum cudaMemcpyKind kind);
    void *malloc(size_t);
    void *hostAlloc(size_t size, unsigned int flags);
    void free(void *ptr);
    void setDevice(int device);
    void setDeviceFlags(unsigned int flags);
    void *getDevicePointer(void *hostPtr, unsigned int flags);
    int getDeviceCount();
    void getDeviceInfo(int device, DeviceInfo &devInfo);
}

#endif
