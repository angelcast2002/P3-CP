
# Makefile for CUDA project

# Specify the compiler for CUDA
NVCC = nvcc

# Flags for CUDA, assuming a common architecture, but this can be adjusted for specific hardware
NVCC_FLAGS = -arch=sm_50 -O2 --compiler-options "-Wall"

# Define the target executable
TARGET = hough

# Object files
OBJS = pgm.o

# Default rule to build the project
all: $(TARGET)

# Rule for building the main CUDA file
$(TARGET): hough.cu $(OBJS)
	$(NVCC) $(NVCC_FLAGS) hough.cu $(OBJS) -o $(TARGET)

# Rule for pgm.o compilation
pgm.o: common/pgm.cpp
	g++ -c common/pgm.cpp -o pgm.o

# Clean rule to remove object files and executable
clean:
	rm -f $(TARGET) $(OBJS)
