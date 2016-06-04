#pragma once
#include <vector>
#include <functional>
#include <string>
#include "MyMath.h"
#include "MatrixFunctions.h"
#include "mythCuda.h"

//隐藏，输入，输出
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

//神经层
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
	
	//这几个矩阵形式相同，行数是节点数，列数是数据组数
	d_matrix *InputMatrix = nullptr, *OutputMatrix = nullptr, *ExpectMatrix = nullptr, *DeltaMatrix = nullptr;

	void deleteData();
	
	//weight矩阵，行数是本层的节点数，列数是上一层的节点数
	d_matrix* WeightMatrix = nullptr;
	//偏移向量，维度为本层节点数
	d_matrix* BiasVector = nullptr;
	//更新偏移向量的辅助向量，所有值为1，维度为数据组数
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

	//dactive是active的导数
	std::function<double(double)> _activeFunction = MyMath::sigmoid;
	std::function<double(double)> _dactiveFunction = MyMath::dsigmoid;

	void setFunctions(std::function<double(double)> active, std::function<double(double)> dactive);

	void activeOutputValue();

	void updateDelta();
	void backPropagate(double learnSpeed = 0.5, double lambda = 0.1);

};

