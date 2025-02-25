# Makefile for compiling the C++20 code

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -Werror

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = test

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/**/*.cpp)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))

# Target executable
TARGET = $(BIN_DIR)/clarabel

# CUDA libraries
CUDA_LIBS = -lcudart -lcusparse -lcublas -lcusolver

# Thrust library
THRUST_LIBS = -lthrust

# Include directories
INCLUDES = -I$(SRC_DIR)

# Default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(CUDA_LIBS) $(THRUST_LIBS)

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Clean up build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Run tests
test: $(TARGET)
	$(TARGET) --test

.PHONY: all clean test
