CXX = nvcc
CXXFLAGS = -O3 -Xcompiler -fopenmp -I./include

SRC_DIR = src
OBJ_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))

TARGET = a2

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)