#pragma once
#include <stdio.h>
#include <vector>
#include <string.h>
#include <cmath>
#include "NeuralLayer.h"
#include "lib/libconvert.h"
#include "MNISTFunctions.h"
#include "Option.h"



//学习模式
typedef enum 
{
	Batch = 0,
	Online = 1,
	MiniBatch = 2,
	//输入向量如果0的项较多，在线学习会比较快
	//通常情况下批量学习会考虑全局优先，应为首选
	//在线学习每次都更新所有键结值，批量学习每一批数据更新一次键结值
} NeuralNetLearnMode;

//计算模式（no use）
/*
typedef enum 
{
	ByLayer,
	ByNode,
} NeuralNetCalMode;
*/

//工作模式
typedef enum
{
	Fit = 0,            //拟合
	Classify = 1,       //分类，会筛选最大值设为1，其他设为0
	Probability = 2,    //几率，结果会归一化	
} NeuralNetWorkMode;


//神经网
class NeuralNet
{
public:
	NeuralNet();
	virtual ~NeuralNet();
	int Id;

	Option _option;
	void loadOptoin(const char* filename);

	void run();

	//神经层
	std::vector<NeuralLayer*> Layers;
	std::vector<NeuralLayer*>& getLayerVector() { return Layers; }
	NeuralLayer*& getLayer(int number) { return Layers[number]; }
	NeuralLayer*& getFirstLayer() { return Layers[0]; }
	NeuralLayer*& getLastLayer() { return Layers.back(); }
	int getLayerCount() { return Layers.size(); };

	int InputNodeCount;
	int OutputNodeCount;

	NeuralNetLearnMode LearnMode = Batch;
	int MiniBatchCount = -1;
	void setLearnMode(NeuralNetLearnMode lm, int lb = -1);

	double LearnSpeed = 0.5;  //学习速度
	void setLearnSpeed(double s) { LearnSpeed = s; }

	double Lambda = 0.0;      //正则化参数，防止过拟合
	void setRegular(double l) { Lambda = l; }

	NeuralNetWorkMode WorkMode = Fit;
	void setWorkMode(NeuralNetWorkMode wm);

	void createLayers(int layerCount);  //包含输入和输出层

	void learn();

	void train(int times = 1000000, int interval = 1000, double tol = 1e-3, double dtol = 0);  //训练过程
	
	void activeOutputValue(double* input, double* output, int groupCount);  //计算一组输出

	void setInputData(double* input, int nodeCount, int groupid);
	void setExpectData(double* expect, int nodeCount, int groupid);

	void getOutputData(int nodeCount, int groupCount, double* output);

	//数据
	double* _train_inputData = nullptr;
	double* _train_expectData = nullptr;
	int _train_groupCount = 0;   //实际的数据量
	
	typedef enum { Train, Test } DateMode;
	void readData(const char* filename, DateMode dm = Train);
	void resetGroupCount(int n);

	double* _test_inputData = nullptr;
	double* _test_expectData = nullptr;
	int _test_groupCount = 0;

	//具体设置
	virtual void createByData(int layerCount = 3, int nodesPerLayer = 7); //具体的网络均改写这里
	void outputBondWeight(const char* filename = nullptr); 
	void createByLoad(const char* filename);

	//NeuralNetCalMode activeMode = ByNode;
	//NeuralNetCalMode backPropageteMode = ByNode;

	void readMNIST();

	bool TestMax = false;
	void selectTest();
	void test();
	void printResult(int nodeCount, int groupCount, double* output, double* expect);
};

