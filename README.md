# EE-769
Building Neural Network from scratch and accelerating on GPU using PyOpenCl 

Instructions:
1. The file NN.py is a pure python file which does not require pyopencl. It consists of implementation of Multi Layer Neural    Network from Scratch.
2. The file NN_Matrix is again a pure Python file. However, few optimizations have been done. In many places for loops have been replaced by using broadcasting in numpy.
3. The file NN_Matrix_opencl.py depends on the module matmul_opencl.py which contains a routine for matrix multiplication. This routine executes on GPU through pyopencl interface and hence requires installation of pyopencl. To install it in ubuntu/mint use the command sudo apt install python3-pyopencl
