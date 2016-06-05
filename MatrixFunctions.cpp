#include "MatrixFunctions.h"


double d_matrix::ddot()
{
	return mythCuda::GetInstance()->myth_ddot(max_script, data, 1, data, 1);
}

void d_matrix::print()
{
#ifdef _DEBUG
	for (int i1 = 0; i1 < row; i1++)
	{
		for (int i2 = 0; i2 < col; i2++)
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

void d_matrix::memcpyDataIn(double* src, int size)
{
	memcpy(data, src, std::min(size, int(sizeof(double)*max_script)));
}

void d_matrix::memcpyDataOut(double* dst, int size)
{
	memcpy(dst, data, std::min(size, int(sizeof(double)*max_script)));
}

//这两个的操作没有数学道理
//将第一列复制到整个矩阵
void d_matrix::expand()
{
	for (int i = 1; i < col; i++)
	{
		memcpy(getDataPointer(0,i), getDataPointer(0,0), sizeof(double)*row);
	}
}

int d_matrix::indexColMaxAbs(int c)
{
	return mythCuda::GetInstance()->myth_idamax(row, getDataPointer(0, c), 1);
	//return cblas_idamax(row, getDataPointer(0, c), 1);
}

double d_matrix::sumColAbs(int c)
{
	return mythCuda::GetInstance()->myth_sumColAbs(row, getDataPointer(0, c), 1);
}

void d_matrix::initData(double v)
{
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

void d_matrix::multiply(double v)
{
	for (int i = 0; i < max_script; i++)
	{
		data[i] *= v;
	}
}

void d_matrix::colMultiply(double v, int c)
{
	for (int i = 0; i < row; i++)
	{
		getData(i, c) *= v;
	}
}

void d_matrix::applyFunction(std::function<double(double)> f)
{
//#pragma omp parallel for
	for (int i = 0; i < max_script; i++)
	{
		data[i] = f(data[i]);
	}
}


//复制数据，只处理较少的
void d_matrix::cpyData(d_matrix* dst, d_matrix* src)
{
	memcpy(dst->data, src->data, sizeof(double)*std::min(dst->row*dst->col, src->row*src->col));
}


void d_matrix::hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R)
{
	for (int i = 0; i < R->max_script; i++)
	{
		R->data[i] = A->data[i] * B->data[i];
	}
}

void d_matrix::minus(d_matrix* A, d_matrix* B, d_matrix* R)
{
	for (int i = 0; i < R->max_script; i++)
	{
		R->data[i] = A->data[i] - B->data[i];
	}
}

void d_matrix::applyFunction(d_matrix* A, d_matrix* R, std::function<double(double)> f)
{
	for (int i = 0; i < std::min(A->max_script, R->max_script); i++)
	{
		R->data[i] = f(A->data[i]);
	}
}

void d_matrix::product(d_matrix* A, d_matrix* B, d_matrix* R,
	double a /*= 1*/, double c /*= 0*/, CBLAS_TRANSPOSE ta /*= CblasNoTrans*/, CBLAS_TRANSPOSE tb /*= CblasNoTrans*/)
{
	int m = R->row;
	int n = R->col;
	int lda = A->row;
	int k = A->col;
	int ldb = B->row;
	if (ta == CblasTrans) { k = A->row; }

	mythCuda::GetInstance()->myth_dgemm(ta, tb, m, n, k, a, A->data, lda, B->data, ldb, c, R->data, m);
	//cblas_dgemm(CblasColMajor, ta, tb, m, n, k, a, A->data, lda, B->data, ldb, c, R->data, m);

}

void d_matrix::productVector(d_matrix* A, d_matrix* B, d_matrix* R, double a /*= 1*/, double c /*= 0*/, CBLAS_TRANSPOSE ta /*= CblasNoTrans*/)
{
	int m = A->row, n = A->col;
	if (ta == CblasTrans) { std::swap(m, n); };
	mythCuda::GetInstance()->myth_dgemv(ta, m, n, a, A->data, A->row, B->data, 1, c, R->data, 1);
	//cblas_dgemv(CblasColMajor, ta, m, n, a, A->data, A->row, B->data, 1, c, R->data, 1);
}