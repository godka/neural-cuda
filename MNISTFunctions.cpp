#include "MNISTFunctions.h"

MNISTFunctions::MNISTFunctions()
{
}


MNISTFunctions::~MNISTFunctions()
{
}

unsigned char* MNISTFunctions::readFile(const char* filename)
{
	FILE *fp = fopen(filename, "rb");
	if (!fp)
	{
		fprintf(stderr, "Can not open file %s\n", filename);
		return nullptr;
	}
	fseek(fp, 0, SEEK_END);
	int length = ftell(fp);
	fseek(fp, 0, 0);
	auto s = new unsigned char[length + 1];
	for (int i = 0; i <= length; s[i++] = '\0');
	fread(s, length, 1, fp);
	fclose(fp);
	return s;
}

int MNISTFunctions::readImageFile(const char* filename, double*& input)
{
	auto content = readFile(filename);
	BE2LE(content + 4, 4);
	BE2LE(content + 8, 4);
	BE2LE(content + 12, 4);
	int count = *(int*)(content + 4);
	int w = *(int*)(content + 8);
	int h = *(int*)(content + 12);
	//fprintf(stderr, "%-30s%d groups data, w = %d, h = %d\n", filename, count, w, h);
	int size = count*w*h;
	input = new double[size];
	memset(input, 0, sizeof(double)*size);
	for (int i = 0; i < size; i++)
	{
		auto v = *(content + 16 + i);
		input[i] = v/255.0;
	}

//  	for (int i = 0; i < 784*10; i++)
// 	{
// 		if (input[i] != 0)
// 			printf("%2.1f ", input[i]);
// 		else
// 			printf("    ");
// 		if (i % 28 == 27)
// 			printf("\n");
// 	}

	delete[] content;
	return w*h;
}

int MNISTFunctions::readLabelFile(const char* filename, double*& expect)
{
	auto content = readFile(filename);
	BE2LE(content + 4, 4);
	int count = *(int*)(content + 4);
	//fprintf(stderr, "%-30s%d groups data\n", filename, count);
	expect = new double[count*10];
	memset(expect, 0, sizeof(double)*count*10);
	for (int i = 0; i < count; i++)
	{
		int pos = *(content + 8 + i);
		expect[i * 10 + pos] = 1;
	}
	delete[] content;
	return 10;
}

void MNISTFunctions::BE2LE(unsigned char* c, int n)
{
	for (int i = 0; i < n / 2; i++)
	{
		auto& a = *(c + i);
		auto& b = *(c + n - 1 - i);
		auto t = b;
		b = a;		
		a = t;
	}
}
