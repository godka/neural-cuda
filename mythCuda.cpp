#include "mythCuda.h"
mythCuda* mythCuda::_mythcuda = NULL;
mythCuda::mythCuda()
{
	auto status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
	}
	else{
		_mythcuda = this;

		/* Allocate device memory for the matrices */
		if (cudaMalloc((void **) &d_A, MAXN *MAXN * sizeof(d_A[0])) != cudaSuccess)
		{
			fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
		}

		if (cudaMalloc((void **) &d_B, 6*MAXN *MAXN * sizeof(d_B[0])) != cudaSuccess)
		{
			fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
		}

		if (cudaMalloc((void **) &d_C, MAXN *MAXN * sizeof(d_C[0])) != cudaSuccess)
		{
			fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
		}
	}
}

int mythCuda::myth_idamax(const int N, const double *X, const int incX){
	int ret;
	bind(X, N, NULL, NULL, NULL, NULL);
	cublasIdamax(handle, N, d_A, incX, &ret);
	return ret - 1;
}
double mythCuda::myth_sumColAbs(const int N, const double *X, const int incX){
	double ret;
	bind(X, N, NULL, NULL, NULL, NULL);
	cublasDasum(handle, N, d_A, incX, &ret);
	return ret;
}

bool mythCuda::HasDevice()
{
	int dev = findCudaDevice(0, NULL);
	if (dev == -1)
	{
		return false;
	}
	else
		return true;
}

double mythCuda::myth_ddot(const int N, const double *X, const int incX,
	const double *Y, const int incY)
{
	double ret = 0;
	bind(X, N, Y, N, NULL, NULL);
	cublasDdot(handle,N, d_A, incX, d_B, incY,&ret);
	return ret;
}
void mythCuda::myth_dgemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
{
	int sd = sizeof(double);
	int a = K, b = N;
	cublasOperation_t ta = CUBLAS_OP_N, tb = CUBLAS_OP_N;
	if (TransA == CblasTrans) { a = M; ta = CUBLAS_OP_T; }
	if (TransB == CblasTrans) { b = K; tb = CUBLAS_OP_T; }
	
	bind(A, lda*a, B, ldb*b, C, ldc*N);
	auto status = cublasDgemm(handle, ta, tb, M, N, K, &alpha,
		d_A, lda, d_B, ldb, &beta, d_C, ldc);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! kernel execution error.\n");
	}
	/* Read the result back */
	status = cublasGetVector(ldc*N, sizeof(double), d_C, 1, C, 1);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! device access error (read C)\n");
	}
}


void mythCuda::myth_dgemv(const CBLAS_TRANSPOSE TransA, const int M, const int N, 
	const double alpha, const double *A, const int lda, const double *X, const int incX, const double beta, double *Y, const int incY)
{
	int sd = sizeof(double);
	int a = N;
	cublasOperation_t ta = CUBLAS_OP_N;
	if (TransA == CblasTrans) { a = M; ta = CUBLAS_OP_T; }

	bind(A, lda*a, X, N*incX, Y, N*incY);
	auto status = cublasDgemv(handle, ta, M, N, &alpha, d_A, lda, d_B, incX, &beta, d_C, incY);
	/* Read the result back */
	status = cublasGetVector(N*incY, sizeof(double), d_C, incY, Y, incY);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! device access error (read C)\n");
	}
}

void mythCuda::bind(const double* A, int sizeA, const double* B, int sizeB, double* R, int sizeR)
{

	/* Initialize the device matrices with the host matrices */
	cublasStatus_t status;
	if (A){
		status = cublasSetVector(sizeA, sizeof(double), A, 1, d_A, 1);

		if (status != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "!!!! device access error (write A)\n");
		}
	}
	if (B){
		status = cublasSetVector(sizeB, sizeof(double), B, 1, d_B, 1);

		if (status != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "!!!! device access error (write B)\n");
		}
	}
	if (R){
		status = cublasSetVector(sizeR, sizeof(double), R, 1, d_C, 1);

		if (status != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "!!!! device access error (write C)\n");
		}
	}

}
mythCuda::~mythCuda(){
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);
	_mythcuda = NULL;
}
