OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

NVCC_FLAGS := -diag-suppress <warning-number>


all: pgm.o hough houghConstant houghShared

hough: houghBase.cu pgm.o
	nvcc $(NVCC_FLAGS) $(OPENCV_CFLAGS) houghBase.cu pgm.o -ljpeg $(OPENCV_LIBS) -o hough

houghConstant: houghConstant.cu pgm.o
	nvcc $(NVCC_FLAGS) $(OPENCV_CFLAGS) houghConstant.cu pgm.o -ljpeg $(OPENCV_LIBS) -o houghConstant

houghShared: houghShared.cu pgm.o
	nvcc $(NVCC_FLAGS) $(OPENCV_CFLAGS) houghShared.cu pgm.o -ljpeg $(OPENCV_LIBS) -o houghShared

pgm.o:	pgm.cpp
	g++ -std=c++17 $(OPENCV_CFLAGS) -c pgm.cpp -o pgm.o

run1:
	./hough runway.pgm

run2:
	./houghConstant runway.pgm

run3:
	./houghShared runway.pgm