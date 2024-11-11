OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)


all: pgm.o houghBase houghConstant houghShared

houghBase: houghBase.cu pgm.o
	nvcc $(OPENCV_CFLAGS) houghBase.cu pgm.o -ljpeg $(OPENCV_LIBS) -o base

houghConstant: houghConstant.cu pgm.o
	nvcc $(OPENCV_CFLAGS) houghConstant.cu pgm.o -ljpeg $(OPENCV_LIBS) -o constant

houghShared: houghShared.cu pgm.o
	nvcc $(OPENCV_CFLAGS) houghShared.cu pgm.o -ljpeg $(OPENCV_LIBS) -o shared

pgm.o:	pgm.cpp
	g++ -std=c++17 $(OPENCV_CFLAGS) -c pgm.cpp -o pgm.o

run1:
	./base runway.pgm

run2:
	./constant runway.pgm

run3:
	./shared runway.pgm