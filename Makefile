# Makefile for CUDA project

# Specify the compiler for CUDA
NVCC = nvcc

# Flags for CUDA, assuming a common architecture, but this can be adjusted for specific hardware
NVCC_FLAGS = -arch=sm_50 -O2 --compiler-options "-Wall"

# Flags for OpenCV (using pkg-config to get the required flags)
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# Define the target executable
TARGET = hough

# Object files
OBJS = pgm.o

# Default rule to build the project
all: $(TARGET)

# Rule for building the main CUDA file
$(TARGET): houghBase.cu $(OBJS)
	$(NVCC) $(NVCC_FLAGS) houghBase.cu $(OBJS) -o $(TARGET) $(OPENCV_FLAGS)

# Rule for pgm.o compilation
pgm.o: pgm.cpp
	g++ -c pgm.cpp -o pgm.o $(OPENCV_FLAGS)

# Clean rule to remove object files and executable
clean:
	rm -f $(TARGET) $(OBJS)
