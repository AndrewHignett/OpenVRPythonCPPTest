// steamVRSideInteractions.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


//https://stackoverflow.com/questions/60917800/how-to-get-the-opencv-image-from-python-and-use-it-in-c-in-pybind11
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <cstdlib>
using namespace std;
#define PY_SSIZE_T_CLEAN
//#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>  // python interpreter
#include <pybind11/stl.h>  // type conversion
namespace py = pybind11;

int main(int argc, char *argv[])
{
	//should really be set up with a UI
	//system("pwd");
	//system("py postEstimationPythonTest.py");


	/*
	while (true)
	{
		//check location in specific memory location for whether it's changed
		//if changed, then update
		//might be faster and easier to just update all the time?
		string line;
		ifstream myfile("C:/tmp/data.txt");
		if (myfile.is_open())
		{
			while (getline(myfile, line))
			{
				//cout << "%p" (void*)line << '\n';
				printf("%s\n", line);
			}
			myfile.close();
		}

		else cout << "Unable to open file";
	}
	*/
	std::cout << "Starting pybind" << std::endl;
	py::scoped_interpreter guard{}; // start interpreter, dies when out of scope
	//py::exec("import test");
					
	py::module::import("cv2");
	py::function min_rosen =
		py::reinterpret_borrow<py::function>(   // cast from 'object' to 'function - use `borrow` (copy) or `steal` (move)
			py::module::import("test").attr("functionTest")  // import method "min_rosen" from python "module"
			);
			
	py::object result = min_rosen();  // automatic conversion from `std::vector` to `numpy.array`, imported in `pybind11/stl.h`
	//bool success = result.attr("success").cast<bool>();
	//int num_iters = result.attr("nit").cast<int>();
	//double obj_value = result.attr("fun").cast<double>();

	while (true)
	{

	}
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file