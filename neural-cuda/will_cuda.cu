
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.hpp"
#undef __CUDA_INTERNAL_COMPILATION__
#include <stdio.h>
#include "will_cuda.h"
#define blockMax 1024

#define cal_i() (blockIdx.x * blockDim.x + threadIdx.x)

inline int blockNum(unsigned int size) { return (size + blockMax - 1) / blockMax; }

inline int getError(const char* content)
{
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "%s Kernel launch failed: %s\n", content, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

__global__ void AddNumberDoubleKernel(double* dst, double* src, double v, double scale, int N)
{
    int i = cal_i();
    if (i < N) { dst[i] = v + scale * src[i]; }
}

__global__ void AddNumberSingleKernel(float* dst, float* src, float v, float scale, int N)
{
    int i = cal_i();
    if (i < N) { dst[i] = v + scale * src[i]; }
}

__global__ void ReciprocalDoubleKernel(double* dst, double* src, double scale, int N)
{
    int i = cal_i();
    if (i < N) { dst[i] = scale / src[i]; }
}

__global__ void ReciprocalSingleKernel(float* dst, float* src, float scale, int N)
{
    int i = cal_i();
    if (i < N) { dst[i] = scale / src[i]; }
}



HBAPI int MYTHAPI cuda_addnumber(double *dst, double *src, unsigned int size, double v, double scale)
{ 
    AddNumberDoubleKernel << < blockNum(size), blockMax >> >(dst, src, v, scale, size);
    return getError("add number double");
}

HBAPI int MYTHAPI cuda_addnumber(float *dst, float *src, unsigned int size, float v, float scale)
{
    AddNumberSingleKernel << < blockNum(size), blockMax >> >(dst, src, v, scale, size);
    return getError("add number float");
}

HBAPI int MYTHAPI cuda_reciprocal(float *dst, float *src, unsigned int size, float scale)
{
    ReciprocalSingleKernel << < blockNum(size), blockMax >> >(dst, src, scale, size);
    return getError("reciprocal number double");
}

HBAPI int MYTHAPI cuda_reciprocal(double *dst, double *src, unsigned int size, double scale)
{
    ReciprocalDoubleKernel << < blockNum(size), blockMax >> >(dst, src, scale, size);
    return getError("reciprocal number float");
}
