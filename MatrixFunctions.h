#pragma once
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <omp.h>
#include "mythCuda.h"
// #define CBLAS_TRANSPOSE cublasOperation_t
// #define CblasNoTrans CUBLAS_OP_N
// #define CblasTrans CUBLAS_OP_T
class d_matrix
{
private:
	bool insideData = true;
public:
	int row = 0;
	int col = 0;
	double* data = nullptr;
	int max_script;
	d_matrix(int x, int y, bool insideData = true)
	{
		row = x;
		col = y;
		this->insideData = insideData;
		if (insideData)
			data = new double[row*col + 1];
		max_script = row*col;
	}
	~d_matrix()
	{
		if(insideData) delete[] data;
	}
	int getRow()
	{
		return row;
	}
	int getCol()
	{
		return col;
	}
	double& getData(int x, int y)
	{
		return data[std::min(x + y*row, max_script)];
	}
	double& getData(int i)
	{
		return data[std::min(i, max_script)];
	}
	double* getDataPointer(int x, int y)
	{
		return &getData(x, y);
	}
	double* getDataPointer(int i)
	{
		return &getData(i);
	}
	double* getDataPointer()
	{
		return data;
	}
	//这个函数可能不安全，慎用！！
	void resetDataPointer(double* d)
	{
		data = d;
	}
	double& operator [] (int i)
	{
		return data[i];
	}
	double ddot();

	void print();
	void memcpyDataIn(double* src, int size);
	void memcpyDataOut(double* dst, int size);
	void expand();
	int indexColMaxAbs(int c);
	double sumColAbs(int c);

	void initData(double v);
	void initRandom();
	void multiply(double v);
	void colMultiply(double v, int c);
	void applyFunction(std::function<double(double)> f);

	static void cpyData(d_matrix* dst, d_matrix* src);

	static void product(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, CBLAS_TRANSPOSE ta = CblasNoTrans, CBLAS_TRANSPOSE tb = CblasNoTrans);
	static void productVector(d_matrix* A, d_matrix* B, d_matrix* R,
		double a = 1, double c = 0, CBLAS_TRANSPOSE ta = CblasNoTrans);
	static void hadamardProduct(d_matrix* A, d_matrix* B, d_matrix* R);
	static void minus(d_matrix* A, d_matrix* B, d_matrix* R);
	static void applyFunction(d_matrix* A, d_matrix* R, std::function<double(double)> f);
};


