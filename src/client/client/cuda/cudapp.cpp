#include"cudapp.h"

namespace CUDA {

    void memcpy(void *dest, const void *src, size_t count, enum cudaMemcpyKind kind)
    {
        cudaError_t cudaError = cudaMemcpy(dest, src, count, kind);
        if(cudaError != cudaSuccess) {
            throw cudaError;
        }
    }

    void *malloc(size_t size)
    {
        void *ptr;
        cudaError_t cudaError = cudaMalloc(&ptr, size);

        if(cudaError != cudaSuccess) {
            throw cudaError;
        }

        return ptr;
    }

    void *hostAlloc(size_t size, unsigned int flags )
    {
        void *ptr;
        cudaError_t cudaError = cudaHostAlloc(&ptr, size, flags);
        
        if(cudaError != cudaSuccess) {
            throw cudaError;
        }

        return ptr;
    }

    void free(void *ptr)
    {
        cudaError_t cudaError = cudaFree(ptr);
        
        if(cudaError != cudaSuccess) {
            throw cudaError;
        }
    }

    void setDevice(int device)
    {
        cudaError_t cudaError = cudaSetDevice(device);
        if(cudaError != cudaSuccess) {
            throw cudaError;
        }
    }

    void setDeviceFlags(unsigned int flags)
    {
        cudaError_t cudaError = cudaSetDeviceFlags(flags);
        if(cudaError != cudaSuccess) {
            throw cudaError;
        }
    }

    void *getDevicePointer(void *hostPtr, unsigned int flags)
    {
        void *devPtr = NULL;
        cudaError_t cudaError = cudaHostGetDevicePointer(&devPtr, hostPtr, flags);
        
        if(cudaError != cudaSuccess) {
            throw cudaError;
        }

        return devPtr;
    }
  
    /**
     * Gets the number of available CUDA devices
     */
    int getDeviceCount()
    {
        int count = 0;
        cudaError_t cudaError = cudaSuccess;

        cudaError = cudaGetDeviceCount(&count);

        if(cudaError != cudaSuccess) {
            return 0;
        }

        return count;
    }

    /**
     * Function to get device information
     */
    void getDeviceInfo(int device,
                        unsigned int *deviceMajor,
                        unsigned int *deviceMinor,
                        unsigned int *mpCount,
                        unsigned long long *globalMemory)
    {
        cudaDeviceProp properties;
        cudaError_t cudaError = cudaSuccess;

        cudaError = cudaSetDevice(device);
        if(cudaError != cudaSuccess) {
            throw cudaError;
        }

        cudaError = cudaGetDeviceProperties(&properties, device);
        if(cudaError != cudaSuccess) {
            throw cudaError;
        }

        *deviceMajor = properties.major;
        *deviceMinor = properties.minor;
        *mpCount = properties.multiProcessorCount;
        *globalMemory = properties.totalGlobalMem;
    }
}
