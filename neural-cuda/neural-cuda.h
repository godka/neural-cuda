#pragma once
//#include <Windows.h>
#define MYTHAPI _stdcall
#ifdef __cplusplus   
#define HBAPI extern "C" __declspec (dllexport)   
#else   
#define HBAPI __declspec (dllexport)   
#endif   

HBAPI int MYTHAPI  cuda_hadamardProduct(const double *A, const double *B, double *R, unsigned int size);
HBAPI int MYTHAPI  cuda_sigmoid(double *A, double *B, unsigned int size);
HBAPI int MYTHAPI  cuda_dsigmoid(double *A, double *B, unsigned int size);
HBAPI int MYTHAPI  cuda_exp(double *A, double *B, unsigned int size);
HBAPI int MYTHAPI  cuda_test();
