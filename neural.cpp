#include "NeuralNet.h"
#include "lib/Timer.h"
#include "mythCuda.h"
#include "MatrixFunctions.h"
int main(int argc, char* argv[])
{


	NeuralNet net;
	Timer t;
	if (!mythCuda::HasDevice()){
		fprintf(stdout, "No Device Found\n");
		exit(0);
	}
	else{
		mythCuda::GetInstance();
	}
	if (argc > 1)
	{
		net.loadOptoin(argv[1]);
	}
	else
		net.loadOptoin("mnist.ini");
	
	t.start();
	net.run();
	t.stop();
	
	fprintf(stdout, "Run neural net end. Time is %lf s.\n", t.getElapsedTime());
	delete mythCuda::GetInstance();
#ifdef _WIN32
	getchar();
#endif
	return 0;
}


