#include "MatrixFunctions.h"


float d_matrix::ddot()
{
	float ret;
	cublasSdot(max_script, data, 1, data, 1, &ret);
	return ret;
}

void d_matrix::print()
{
#ifdef _DEBUG
	for (int i1 = 0; i1 < m; i1++)
	{
		for (int i2 = 0; i2 < n; i2++)
		{
			fprintf(stderr, "%11.5lf ", getData(i1, i2));
		}
		fprintf(stderr, "\n");
	}
	// 	for (int i = 0; i < m*n; i++)
	// 	{
	// 			printf("%11.5lf ", getData(i));
	// 	}
	fprintf(stderr, "\n");
#endif
}

void d_matrix::memcpyDataIn(float* src, int size)
{
	memcpy(data, src, std::min(size, int(sizeof(float)*max_script)));
}

void d_matrix::memcpyDataOut(float* dst, int size)
{
	memcpy(dst, data, std::min(size, int(sizeof(float)*max_script)));
}

//这两个的操作没有数学道理
//将第一列复制到整个矩阵
void d_matrix::expand()
{
#pragma loop(hint_parallel(8))
	for (int i = 1; i < n; i++)
	{
		memcpy(getDataPointer(0, i), getDataPointer(0, 0), sizeof(float)*m);
	}
}

int d_matrix::indexColMaxAbs(int c)
{
	int ret;
	cublasIsamax(m, getDataPointer(0, c),1,&ret);
	return ret;
}

float d_matrix::sumColAbs(int c)
{
	float ret;
	cublasSasum(m, getDataPointer(0, c), 1, &ret);
	return ret;
}

void d_matrix::initData(float v)
{
#pragma loop(hint_parallel(8))
	for (int i = 0; i < max_script; i++)
	{
		data[i] = v;
	}
}

void d_matrix::initRandom()
{
	//#pragma loop(hint_parallel(8))
	for (int i = 0; i < max_script; i++)
	{
		data[i] = 2.0 * rand() / RAND_MAX - 1;
	}
}

void d_matrix::multiply(float v)
{
#pragma loop(hint_parallel(8))
	for (int i = 0; i < max_script; i++)
	{
		data[i] *= v;
	}
}

void d_matrix::colMultiply(float v, int c)
{
#pragma loop(hint_parallel(8))
	for (int i = 0; i < n; i++)
	{
		getData(i, c) *= v;
	}
}

void d_matrix::applyFunction(std::function<float(float)> f)
{
#pragma loop(hint_parallel(8))
	for (int i = 0; i < max_script; i++)
	{
		data[i] = f(data[i]);
	}
}


//复制数据，只处理较少的
void d_matrix::cpyData(d_matrix* dst, d_matrix* src)
{
	memcpy(dst->data, src->data, sizeof(float)*std::min(dst->m*dst->n, src->m*src->n));
}

void d_matrix::product(d_matrix* A, d_matrix* B, d_matrix* R,
	float a /*= 1*/, float c /*= 0*/, CBLAS_TRANSPOSE ta /*= CblasNoTrans*/, CBLAS_TRANSPOSE tb /*= CblasNoTrans*/)
{
	float *d_a;
	float *d_b;
	float *d_c;
	int m = R->m;
	int n = R->n;
	int lda = A->m;
	int k = A->n;
	int ldb = B->m;
	if (ta == CblasTrans) { k = A->m; }
	
	if (cudaMalloc((void **) &d_a, A->max_script*sizeof(float) != cudaSuccess))
		fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
	if (cudaMalloc((void **) &d_b, B->max_script*sizeof(float)) != cudaSuccess)
		fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
	if (cudaMalloc((void **) &d_c, R->max_script*sizeof(float)) != cudaSuccess)
		fprintf(stderr, "!!!! device memory allocation error (allocate R)\n");
	

	auto ret = cublasSetVector(A->max_script, sizeof(float), A->data, 1, d_a, 1);
	ret = cublasSetVector(B->max_script, sizeof(float), B->data, 1, d_b, 1);
	//cublasSetVector(R->max_script, sizeof(float), R->data, 1, d_c, 1);
	cublasSgemm(ta, tb, m, n, k, a, d_a, lda, d_b, ldb, c, d_c, m);
	auto status2 = cublasGetVector(R->max_script, sizeof(float), d_c, 1, R->data, 1);
	//status = cublasGetVector(R->m*R->n, sizeof(R->data[0]), d_c, 1, R->data, 1);
	if (status2 != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! device access error (read R)\n");
	}
	if (cudaFree(d_a) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (A)\n");
	}

	if (cudaFree(d_b) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (B)\n");
	}

	if (cudaFree(d_c) != cudaSuccess)
	{
		fprintf(stderr, "!!!! memory free error (C)\n");
	}
	//cblas_dgemm(CblasColMajor, ta, tb, m, n, k, a, A->data, lda, B->data, ldb, c, R->data, m);
}

void d_matrix::productVector(d_matrix* A, d_matrix* B, d_matrix* R, float a /*= 1*/, float c /*= 0*/, CBLAS_TRANSPOSE ta /*= CblasNoTrans*/)
{
	int m = A->m, n = A->n;
	if (ta == CblasTrans) { std::swap(m, n); };
	cublasSgemv(mythCuda::GetInstance()->handle, ta, m, n, &a, A->data, A->m, B->data, 1, &c, R->data, 1);
}

void d_matrix::hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R)
{
#pragma loop(hint_parallel(8))
	for (int i = 0; i < R->max_script; i++)
	{
		R->data[i] = A->data[i] * B->data[i];
	}
}

void d_matrix::minus(d_matrix* A, d_matrix* B, d_matrix* R)
{
#pragma loop(hint_parallel(8))
	for (int i = 0; i < R->max_script; i++)
	{
		R->data[i] = A->data[i] - B->data[i];
	}
}

void d_matrix::applyFunction(d_matrix* A, d_matrix* R, std::function<float(float)> f)
{
#pragma loop(hint_parallel(8))
	for (int i = 0; i < std::min(A->max_script, R->max_script); i++)
	{
		R->data[i] = f(A->data[i]);
	}
}