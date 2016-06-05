#pragma once
/* Includes, cuda */
#include "lib/cblas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#define MAXN 3000	//100m
class mythCuda
{
public:
	cublasHandle_t handle;
	static mythCuda* _mythcuda;
	mythCuda();
	int myth_idamax(const int N, const double *X, const int incX);
	double myth_sumColAbs(const int N, const double *X, const int incX);
	~mythCuda();

	static mythCuda* GetInstance(){
		if (_mythcuda)
			return _mythcuda;
		else
			return new mythCuda();
	}
	static bool HasDevice();
	double myth_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
	void myth_dgemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
	void myth_dgemv(const CBLAS_TRANSPOSE TransA, const int M, const int N, 
		const double alpha, const double  *A, const int lda, const double  *X, const int incX, const double beta, double  *Y, const int incY);
private:
	double* d_A;
	double* d_B;
	double* d_C;
	void bind(const double* A, int sizeA, const double* B, int sizeB, double* R, int sizeR);
};