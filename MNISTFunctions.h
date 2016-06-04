#pragma once
#include "lib/libconvert.h"
#include <string.h>

class MNISTFunctions
{
public:
	MNISTFunctions();
	virtual ~MNISTFunctions();

	static unsigned char* readFile(const char* filename);
	static int readImageFile(const char* filename, double*& input);
	static int readLabelFile(const char* filename, double*& expect);
	static void BE2LE(unsigned char* c, int n);
};

