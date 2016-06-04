#pragma once
#include <math.h>
//#include "NeuralNode.h"

namespace MyMath
{
	//static xoid setFunctions(class NeuralNode* node, std::function<double(double)> activeFunction, std::function<double(double)> feedbackFunction);

	static double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }
	static double dsigmoid(double x) { double a = 1 + exp(-x); return exp(-x) / (a*a); }
	static double linear(double x) { return x; }
	static double dlinear(double x) { return 1; }
	static double exp1(double x) { return exp(x); }
	static double dexp1(double x) { return exp(x); }
	static double tanh1(double x) { return tanh(x); }
	static double dtanh1(double x) { return 1 / cosh(x) / cosh(x); }

	static double sign1(double x) { return x > 0 ? 1 : -1; }
	static double dsign1(double x) { return 1; }

	static double is(double x) { return x > 0.5 ? 1 : 0; }
	static double dis(double x) { return 1; }

	//static int min(int a, int b) { return (a < b) ? a : b; }
	//static void swap(int &a, int &b) { auto t = a; a = b; b = t; }

};

