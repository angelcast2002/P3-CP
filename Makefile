all: pgm.o hough houghConstant houghShared

hough: houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -ljpeg -o hough

houghConstant: houghConstant.cu pgm.o
	nvcc houghConstant.cu pgm.o -ljpeg -o houghConstant

houghShared: houghShared.cu pgm.o
	nvcc houghShared.cu pgm.o -ljpeg -o houghShared

pgm.o:	pgm.cpp
	g++ -std=c++17 -c pgm.cpp -o pgm.o

run1:
	./hough runway.pgm

run2:
	./houghConstant runway.pgm

run3:
	./houghShared runway.pgm
