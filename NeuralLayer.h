#pragma once
#include <vector>
#include <functional>
#include <string>
#include "MyMath.h"
#include "MatrixFunctions.h"
#include "mythCuda.h"

//���أ����룬���
typedef enum
{
	Hidden,
	Input,
	Output,
} NeuralLayerType;

typedef enum
{
	FullConnection,
	Convolution,
	Pooling,
} NeuralLayerWorkMode;

//�񾭲�
class NeuralLayer
{
public:
	NeuralLayer();
	virtual ~NeuralLayer();

	int Id;

	int NodeCount;
	static int GroupCount;
	static int Step;

	NeuralLayerType Type = Hidden;

	NeuralLayerWorkMode WorkMode = FullConnection;
	
	//�⼸��������ʽ��ͬ�������ǽڵ�������������������
	d_matrix *InputMatrix = nullptr, *OutputMatrix = nullptr, *ExpectMatrix = nullptr, *DeltaMatrix = nullptr;

	void deleteData();
	
	//weight���������Ǳ���Ľڵ�������������һ��Ľڵ���
	d_matrix* WeightMatrix = nullptr;
	//ƫ��������ά��Ϊ����ڵ���
	d_matrix* BiasVector = nullptr;
	//����ƫ�������ĸ�������������ֵΪ1��ά��Ϊ��������
	d_matrix* _asBiasVector = nullptr;

	NeuralLayer *PrevLayer, *NextLayer;

	void initData(int nodeCount, int groupCount, NeuralLayerType type = Hidden);
	void resetData(int groupCount);
	d_matrix*& getOutputMatrix() { return OutputMatrix; }	
	d_matrix*& getExpectMatrix() { return ExpectMatrix; }
	d_matrix*& getDeltaMatrix() { return DeltaMatrix; }

	double& getOutputValue(int x, int y) { return OutputMatrix->getData(x, y); }

	static void connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer);
	void connetPrevlayer(NeuralLayer* prevLayer);
	void connetNextlayer(NeuralLayer* nextLayer);
	//void connet(NueralLayer nextLayer);
	void markMax();
	void normalized();

	//dactive��active�ĵ���
	std::function<double(double)> _activeFunction = MyMath::sigmoid;
	std::function<double(double)> _dactiveFunction = MyMath::dsigmoid;

	void setFunctions(std::function<double(double)> active, std::function<double(double)> dactive);

	void activeOutputValue();

	void updateDelta();
	void backPropagate(double learnSpeed = 0.5, double lambda = 0.1);

};

