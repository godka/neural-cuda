
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.hpp"
#undef __CUDA_INTERNAL_COMPILATION__
#include <stdio.h>
#include "neural-cuda.h"
#define blockMax 1024
__global__ void AddNumberDoubleKernel(double* dst, double* src, double v,double scale, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
		dst[i] = v + scale * src[i];
}
__global__ void AddNumberSingleKernel(float* dst, float* src, float v, float scale, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
		dst[i] = v + scale * src[i];
}
__global__ void ReciprocalDoubleKernel(double* dst, double* src, double scale, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
		dst[i] = scale / src[i];
}
__global__ void ReciprocalSingleKernel(float* dst, float* src, float scale, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
		dst[i] = scale / src[i];
}
__global__ void MatrixMulKernel(const double* A, const double* B, double* C, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
		C[i] = A[i] * B[i];
}

__global__ void SigmoidKernel(double* A, double* B, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
	{
		B[i] = 1 / (1 + exp(-A[i]));
	}
}
__global__ void DsigmoidKernel(double* A, double* B, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
	{
		double a = 1 + exp(-A[i]);
		B[i] = (a - 1) / (a*a);
	}
}

__global__ void ExpKernel(double* A, double* B, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
	{
		B[i] = exp(A[i]);
	}
}

HBAPI int MYTHAPI cuda_hadamardProduct(const double *A, const double *B, double *R, unsigned int size)
{
	int blockNum = (size + blockMax - 1) / blockMax;

	MatrixMulKernel << < blockNum, blockMax >> >(A, B, R, size);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	return cudaStatus;
	//}
	return 0;
}

HBAPI int MYTHAPI cuda_addnumber_double(double *dst, double *src, unsigned int size, double v, double scale){
	int blockNum = (size + blockMax - 1) / blockMax;

	AddNumberDoubleKernel << < blockNum, blockMax >> >(dst, src, v,scale, size);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	return cudaStatus;
	//}
	return 0;
}
HBAPI int MYTHAPI cuda_addnumber_float(float *dst, float *src, unsigned int size, float v, float scale){
	int blockNum = (size + blockMax - 1) / blockMax;

	AddNumberSingleKernel << < blockNum, blockMax >> >(dst, src, v, scale, size);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	return cudaStatus;
	//}
	return 0;
}
HBAPI int MYTHAPI cuda_reciprocal_float(float *dst, float *src, unsigned int size, float scale)
{
	int blockNum = (size + blockMax - 1) / blockMax;

	ReciprocalSingleKernel << < blockNum, blockMax >> >(dst, src, scale, size);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	return cudaStatus;
	//}
	return 0;

}
HBAPI int MYTHAPI cuda_reciprocal_double(double *dst, double *src, unsigned int size,double scale)
{
	int blockNum = (size + blockMax - 1) / blockMax;

	ReciprocalDoubleKernel << < blockNum, blockMax >> >(dst, src, scale,size);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	return cudaStatus;
	//}
	return 0;

}

HBAPI int MYTHAPI cuda_dsigmoid(double *A, double *B, unsigned int size)
{
	int blockNum = (size + blockMax - 1) / blockMax;
	DsigmoidKernel << < blockNum, blockMax >> >(A, B, size);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	return 1;
	//}
	return 0;
}

HBAPI int MYTHAPI  cuda_test(){
	printf("hello cuda!\n");
	return 0;
}
HBAPI int MYTHAPI cuda_sigmoid(double *A, double *B, unsigned int size)
{
	int blockNum = (size + blockMax - 1) / blockMax;

	SigmoidKernel << < blockNum, blockMax >> >(A, B, size);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	return 1;
	//}
	return 0;
}
HBAPI int MYTHAPI cuda_exp(double *A, double *B, unsigned int size)
{
	int blockNum = (size + blockMax - 1) / blockMax;

	SigmoidKernel << < blockNum, blockMax >> >(A, B, size);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return 1;
	}
	return 0;
}