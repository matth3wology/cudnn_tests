CC=nvcc
INCLUDE=-I"C:\Program Files (x86)\opencv\build\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include"
LIB=-L"C:\Program Files (x86)\opencv\build\x64\vc15\lib" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib"
LIBRARY_FLAGS=-lopencv_world430 -lcublas -lcudnn

all:
	$(CC) main.cpp $(INCLUDE) $(LIB) $(LIBRARY_FLAGS) -o main

clean:
	DEL *.ex*
	DEL *.lib