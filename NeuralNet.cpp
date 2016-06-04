#include "NeuralNet.h"


NeuralNet::NeuralNet()
{
}


NeuralNet::~NeuralNet()
{
	for (auto& layer : getLayerVector())
	{
		delete layer;
	}
	if (_train_inputData) 
		delete[] _train_inputData;
	if (_train_expectData)	
		delete[] _train_expectData;
	if (_test_inputData) 
		delete[] _test_inputData;
	if (_test_expectData) 
		delete[] _test_expectData;
}

//���У�ע���ݴ�������
void NeuralNet::run()
{
	LearnMode = NeuralNetLearnMode(_option.LearnMode);
	MiniBatchCount = _option.MiniBatch;
	WorkMode = NeuralNetWorkMode(_option.WorkMode);

	LearnSpeed = _option.LearnSpeed;
	Lambda = _option.Regular;
	TestMax = _option.TestMax;

	if (_option.UseMNIST == 0)
	{
		if (_option.TrainDataFile != "")
		{
			readData(_option.TrainDataFile.c_str(), Train);
		}
	}
	else
	{
		readMNIST();
	}

	//�������ļ�ǿ�����´������磬��̫����
	//if (readStringFromFile(_option.LoadFile) == "")
	//	_option.LoadNet == 0;

	if (_option.LoadNet == 0)
		createByData(_option.Layer, _option.NodePerLayer);
	else
		createByLoad(_option.LoadFile.c_str());

	//net->selectTest();
	train(int(_option.TrainTimes), int(_option.OutputInterval), _option.Tol, _option.Dtol);
	test();

	if (_option.SaveFile != "")
		outputBondWeight(_option.SaveFile.c_str());
	if (_option.TestDataFile != "")
	{
		readData(_option.TestDataFile.c_str(), Test);
		test();
	}
}

//����ѧϰģʽ
void NeuralNet::setLearnMode(NeuralNetLearnMode lm, int lb /*= -1*/)
{
	LearnMode = lm;
	//����ѧϰʱ���ڵ�����������ʵ��������
	if (LearnMode == Online)
	{
		MiniBatchCount = 1;
	}
	//�����������������
	if (LearnMode == MiniBatch)
	{
		MiniBatchCount = lb;
	}
}

void NeuralNet::setWorkMode(NeuralNetWorkMode wm)
{
	 WorkMode = wm; 
	 if (wm == Probability)
	 {
		 getLastLayer()->setFunctions(MyMath::exp1, MyMath::dexp1);
	 }
}

//�����񾭲�
void NeuralNet::createLayers(int layerCount)
{
	Layers.resize(layerCount);
	for (int i = 0; i < layerCount; i++)
	{
		auto layer = new NeuralLayer();
		layer->Id = i;
		Layers[i] = layer;
	}
}



//�������
//���ﰴ��ǰ������Ӧ�����𲽻��ݼ��㣬ʹ��ջ��������˳�򣬴����ƺ��޸�
void NeuralNet::activeOutputValue(double* input, double* output, int groupCount)
{	
	if (input)
	{
		setInputData(input, InputNodeCount, 0);
	}	
	for (int i = 1; i < getLayerCount(); i++)
	{
		Layers[i]->activeOutputValue();
	}

	if (WorkMode == Probability)
	{
		getLastLayer()->normalized();
	}
	else if (WorkMode == Classify)
	{
		getLastLayer()->markMax();
	}

	if (output)
	{
		getOutputData(OutputNodeCount, groupCount, output);
	}
}

void NeuralNet::setInputData(double* input, int nodeCount, int groupid)
{
	getFirstLayer()->getOutputMatrix()->resetDataPointer(input + nodeCount*groupid);
}

void NeuralNet::setExpectData(double* expect, int nodeCount, int groupid)
{
	getLastLayer()->getExpectMatrix()->resetDataPointer(expect + nodeCount*groupid);
}


void NeuralNet::getOutputData(int nodeCount, int groupCount, double* output)
{
	getLastLayer()->getOutputMatrix()->memcpyDataOut(output, sizeof(double)*nodeCount*groupCount);
}

//ѧϰ����
void NeuralNet::learn()
{
	//�������
	activeOutputValue(nullptr, nullptr, _train_groupCount);
	//���򴫲�
	for (int i = getLayerCount() - 1; i > 0; i--)
	{
		Layers[i]->backPropagate(LearnSpeed, Lambda);
	}
}

//ѵ��һ�����ݣ��������������ѵ������Ϊ0�������Ϊ������ģʽ
void NeuralNet::train(int times /*= 1000000*/, int interval /*= 1000*/, double tol /*= 1e-3*/, double dtol /*= 1e-9*/)
{
	if (times <= 0) return;
	//��������ʼ��������㹻С�Ͳ�ѵ����
	setInputData(_train_inputData, InputNodeCount, 0);
	setExpectData(_train_expectData, OutputNodeCount, 0);
	activeOutputValue(nullptr, nullptr, _train_groupCount);
	getLastLayer()->updateDelta();
	double e = getLastLayer()->getDeltaMatrix()->ddot() / (_train_groupCount*OutputNodeCount);
	fprintf(stdout, "step = %e, mse = %e\n", 0.0, e);
	if (e < tol) return;
	double e0 = e;
	
	if (LearnMode == Online)
	{
		resetGroupCount(1);
		MiniBatchCount = 1;
	}
	else if (LearnMode == Batch)
	{
		//resetGroupCount(_train_groupCount);
		MiniBatchCount = _train_groupCount;
		e0 = 0;  //�����ʹ����������ѵ������ʼ�������һ��ֵ�������жϴ���
	}
	else if (LearnMode == MiniBatch)
	{
		if (MiniBatchCount > 0)
			resetGroupCount(MiniBatchCount);
	}

	//ѵ������
	for (int count = 1; count <= times; count++)
	{
		//getFirstLayer()->step = count;
		if (LearnMode == Batch)
		{
			learn();
			//������ʵ������һ����
			if (count % interval == 0)
			{
				e = getLastLayer()->getDeltaMatrix()->ddot();
			}
		}
		else
		{
			for (int i = 0; i < _train_groupCount; i += MiniBatchCount)
			{
				setInputData(_train_inputData, InputNodeCount, i);
				setExpectData(_train_expectData, OutputNodeCount, i);
				learn();
				//������ע������㷨����minibatch���ϸ�
				if (count % interval == 0)
				{
					e += getLastLayer()->getDeltaMatrix()->ddot();
				}
			}
		}
		if (count % interval == 0)
		{
			e /= (_train_groupCount*OutputNodeCount);
			fprintf(stdout, "step = %e, mse = %e, diff(mse) = %e\n", double(count), e, e0 - e);
			if (e < tol || std::abs(e - e0) < dtol) break;
			e0 = e;
			e = 0;
		}
	}
}

//��ȡ����
//����Ĵ�����ܲ��Ǻܺ�
void NeuralNet::readData(const char* filename, DateMode dm/*= Train*/)
{
	_train_groupCount = 0;
	_test_groupCount = 0;

	int mark = 3;
	//���ݸ�ʽ��ǰ����������������������������֮��������ÿ��������������Ƿ��лس�����Ҫ
	std::string str = readStringFromFile(filename) + "\n";
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	if (n <= 0) return;
	InputNodeCount = int(v[0]);
	OutputNodeCount = int(v[1]);

	auto groupCount = &_train_groupCount;
	auto inputData = &_train_inputData;
	auto expectData = &_train_expectData;
	if (dm == Test)
	{
		groupCount = &_test_groupCount;
		inputData = &_test_inputData;
		expectData = &_test_expectData;
	}

	*groupCount = (n - mark) / (InputNodeCount + OutputNodeCount);
	*inputData = new double[InputNodeCount * (*groupCount)];
	*expectData = new double[OutputNodeCount * (*groupCount)];

	//д��̫�ѿ���
	int k = mark, k1 = 0, k2 = 0;

	for (int i_data = 1; i_data <= (*groupCount); i_data++)
	{
		for (int i = 1; i <= InputNodeCount; i++)
		{
			(*inputData)[k1++] = v[k++];
		}
		for (int i = 1; i <= OutputNodeCount; i++)
		{
			(*expectData)[k2++] = v[k++];
		}
	}
// 	for (int i = 0; i < 784 * 22; i++)
// 	{
// 		if ((*inputData)[i] > 0.5)
// 			printf("%2.1f ", (*inputData)[i]);
// 		else
// 		{
// 			printf("    ");
// 			(*inputData)[i] = 0;
// 		}
// 		if (i % 28 == 27)
// 			printf("\n");
// 	}
}

void NeuralNet::resetGroupCount(int n)
{
	for (auto l : Layers)
	{
		l->resetData(n);
	}
}

//�������ֵ
void NeuralNet::outputBondWeight(const char* filename)
{
	FILE *fout = stdout;
	if (filename)
		fout = fopen(filename, "w+t");

	fprintf(fout,"Net information:\n");
	fprintf(fout,"%d\tlayers\n", Layers.size());
	for (int i_layer = 0; i_layer < getLayerCount(); i_layer++)
	{
		fprintf(fout,"layer %d has %d nodes\n", i_layer, Layers[i_layer]->NodeCount);
	}

	fprintf(fout,"---------------------------------------\n");
	for (int i_layer = 0; i_layer < getLayerCount() - 1; i_layer++)
	{
		auto& layer1 = Layers[i_layer];
		auto& layer2 = Layers[i_layer + 1];
		fprintf(fout, "weight for layer %d to %d\n", i_layer + 1, i_layer);
		for (int i2 = 0; i2 < layer2->NodeCount; i2++)
		{
			for (int i1 = 0; i1 < layer1->NodeCount; i1++)
			{
				fprintf(fout, "%14.11lf ", layer2->WeightMatrix->getData(i2, i1));
			}
			fprintf(fout, "\n");
		}
		fprintf(fout, "bias for layer %d\n", i_layer + 1);
		for (int i2 = 0; i2 < layer2->NodeCount; i2++)
		{
			fprintf(fout, "%14.11lf ", layer2->BiasVector->getData(i2));
		}
		fprintf(fout, "\n");
	}

	if (filename)
		fclose(fout);
}

//�����������ݴ�������������Ľڵ���ֻ�����ز�����
//�˴��Ǿ��������ṹ
void NeuralNet::createByData(int layerCount /*= 3*/, int nodesPerLayer /*= 7*/)
{
	this->createLayers(layerCount);

	getFirstLayer()->initData(InputNodeCount, _train_groupCount, Input);
	for (int i = 1; i < layerCount - 1; i++)
	{
		getLayer(i)->initData(nodesPerLayer, _train_groupCount, Hidden);
	}	
	getLastLayer()->initData(OutputNodeCount, _train_groupCount, Output);

	for (int i = 1; i < layerCount; i++)
	{
		Layers[i]->connetPrevlayer(Layers[i - 1]);
	}
}

//���ݼ���ֵ��������
void NeuralNet::createByLoad(const char* filename)
{
	std::string str = readStringFromFile(filename) + "\n";
	if (str == "")
		return;
	std::vector<double> v;
	int n = findNumbers(str, v);
	// 	for (int i = 0; i < n; i++)
	// 		printf("%14.11lf\n",v[i]);
	// 	printf("\n");
	std::vector<int> v_int;
	v_int.resize(n);
	for (int i_layer = 0; i_layer < n; i_layer++)
	{
		v_int[i_layer] = int(v[i_layer]);
	}
	int k = 0;
	int layerCount = v_int[k++];
	this->createLayers(layerCount);
	getFirstLayer()->Type = Input;
	getLastLayer()->Type = Output;
	k++;
	for (int i_layer = 0; i_layer < layerCount; i_layer++)
	{
		getLayer(i_layer)->initData(v_int[k], _train_groupCount, getLayer(i_layer)->Type);
		k += 2;
	}
	k = 1 + layerCount * 2;
	for (int i_layer = 0; i_layer < layerCount - 1; i_layer++)
	{
		auto& layer1 = Layers[i_layer];
		auto& layer2 = Layers[i_layer + 1];
		layer2->connetPrevlayer(layer1);
		k += 2;
		for (int i2 = 0; i2 < layer2->NodeCount; i2++)
		{
			for (int i1 = 0; i1 < layer1->NodeCount; i1++)
			{
				layer2->WeightMatrix->getData(i2, i1) = v[k++];
			}
		}
		k += 1;
		for (int i2 = 0; i2 < layer2->NodeCount; i2++)
		{
			layer2->BiasVector->getData(i2) = v[k++];
		}
	}
}

void NeuralNet::readMNIST()
{
	InputNodeCount = MNISTFunctions::readImageFile("train-images.idx3-ubyte", _train_inputData);
	OutputNodeCount = MNISTFunctions::readLabelFile("train-labels.idx1-ubyte", _train_expectData);
	_train_groupCount = 60000;

	MNISTFunctions::readImageFile("t10k-images.idx3-ubyte", _test_inputData);
	MNISTFunctions::readLabelFile("t10k-labels.idx1-ubyte", _test_expectData);
	_test_groupCount = 10000;
}

void NeuralNet::loadOptoin(const char* filename)
{
	_option.loadIni(filename);
}


//�����һ��������Ϊ�������ݣ�д����hack����
void NeuralNet::selectTest()
{
	//����ԭ��������
	auto input = new double[InputNodeCount*_train_groupCount];
	auto output = new double[OutputNodeCount*_train_groupCount];
	memcpy(input, _train_inputData, sizeof(double)*InputNodeCount*_train_groupCount);
	memcpy(output, _train_expectData, sizeof(double)*OutputNodeCount*_train_groupCount);

	_test_inputData = new double[InputNodeCount*_train_groupCount];
	_test_expectData = new double[OutputNodeCount*_train_groupCount];

	std::vector<bool> isTest;
	isTest.resize(_train_groupCount);

	_test_groupCount = 0;
	int p = 0, p_data = 0, p_test = 0;
	int it = 0, id = 0;
	for (int i = 0; i < _train_groupCount; i++)
	{
		isTest[i] = (0.9 < 1.0*rand() / RAND_MAX);
		if (isTest[i])
		{
			memcpy(_test_inputData + InputNodeCount*it, input + InputNodeCount*i, sizeof(double)*InputNodeCount);
			memcpy(_test_expectData + OutputNodeCount*it, output + OutputNodeCount*i, sizeof(double)*OutputNodeCount);
			_test_groupCount++;
			it++;
		}
		else
		{
			memcpy(_train_inputData + InputNodeCount*id, input + InputNodeCount*i, sizeof(double)*InputNodeCount);
			memcpy(_train_expectData + OutputNodeCount*id, output + OutputNodeCount*i, sizeof(double)*OutputNodeCount);
			id++;
		}
	}
	_train_groupCount -= _test_groupCount;
	resetGroupCount(_train_groupCount);
}

//�����ϵĽ���Ͳ��Լ��Ľ��
void NeuralNet::test()
{
	if (_train_groupCount > 0)
	{
		resetGroupCount(_train_groupCount);
		auto train_output = new double[OutputNodeCount*_train_groupCount];
		activeOutputValue(_train_inputData, train_output, _train_groupCount);
		fprintf(stdout, "\n%d groups train data:\n---------------------------------------\n", _train_groupCount);
		printResult(OutputNodeCount, _train_groupCount, train_output, _train_expectData);
		delete[] train_output;
	}
	if (_test_groupCount > 0)
	{
		resetGroupCount(_test_groupCount);
		auto test_output = new double[OutputNodeCount*_test_groupCount];
		activeOutputValue(_test_inputData, test_output, _test_groupCount);
		fprintf(stdout, "\n%d groups test data:\n---------------------------------------\n", _test_groupCount);
		printResult(OutputNodeCount, _test_groupCount, test_output, _test_expectData);
		delete[] test_output;
	}
}

void NeuralNet::printResult(int nodeCount, int groupCount, double* output, double* expect)
{
	if (groupCount <= 100)
	{
		for (int i = 0; i < groupCount; i++)
		{
			for (int j = 0; j < nodeCount; j++)
			{
				fprintf(stdout, "%8.4lf ", output[i*nodeCount + j]);
			}
			fprintf(stdout, " --> ");
			for (int j = 0; j < nodeCount; j++)
			{
				fprintf(stdout, "%8.4lf ", expect[i*nodeCount + j]);
			}
			fprintf(stdout, "\n");
		}
	}

	if (TestMax)
	{
		getLastLayer()->markMax();
		auto outputMax = new double[nodeCount*groupCount];
		getOutputData(nodeCount, groupCount, outputMax);

		if (groupCount <= 100)
		{
			for (int i = 0; i < groupCount; i++)
			{
				for (int j = 0; j < nodeCount; j++)
				{
					if (outputMax[i*nodeCount + j] == 1)
						fprintf(stdout, "%3d (%6.4lf) ", j, output[i*nodeCount + j]);
				}
				fprintf(stdout, " --> ");
				for (int j = 0; j < nodeCount; j++)
				{
					if (expect[i*nodeCount + j] == 1)
						fprintf(stdout, "%3d ", j);
				}
				fprintf(stdout, "\n");
			}
		}

		double n = 0;
		for (int i = 0; i < nodeCount*groupCount; i++)
			n += std::abs(outputMax[i] - expect[i]);
		n /= 2;
		delete[] outputMax;
		fprintf(stdout, "Error of max value position: %d, %5.2lf%%\n", int(n), n / groupCount * 100);
	}
}


