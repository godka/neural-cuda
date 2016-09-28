#pragma once
//#include <Windows.h>
#define MYTHAPI _stdcall
#ifdef __cplusplus   
#define HBAPI /*extern "C"*/ __declspec (dllexport)   
#else   
#define HBAPI __declspec (dllexport)   
#endif   

HBAPI int MYTHAPI cuda_reciprocal(double *dst, double *src, unsigned int size, double scale);
HBAPI int MYTHAPI cuda_reciprocal(float *dst, float *src, unsigned int size, float scale);

HBAPI int MYTHAPI cuda_addnumber(double *dst, double *src, unsigned int size, double v,double scale);
HBAPI int MYTHAPI cuda_addnumber(float *dst, float *src, unsigned int size, float v,float scale);

/*
HBAPI int MYTHAPI  cuda_dsigmoid(double *A, double *B, unsigned int size);
HBAPI int MYTHAPI  cuda_hadamardProduct(const double *A, const double *B, double *R, unsigned int size);
HBAPI int MYTHAPI  cuda_sigmoid(double *A, double *B, unsigned int size);
HBAPI int MYTHAPI  cuda_dsigmoid(double *A, double *B, unsigned int size);
HBAPI int MYTHAPI  cuda_exp(double *A, double *B, unsigned int size);
HBAPI int MYTHAPI  cuda_test();
*/
