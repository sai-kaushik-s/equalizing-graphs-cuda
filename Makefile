NVCC = nvcc

ARCH_FLAGS := -gencode arch=compute_35,code=sm_35 \
              -gencode arch=compute_35,code=compute_35

ARCH_TEST := $(shell nvcc $(ARCH_FLAGS) -x cu /dev/null -o /dev/null >/dev/null 2>&1; echo $$?)
ifneq ($(ARCH_TEST),0)
    ARCH_FLAGS := -arch=sm_86
endif

CFLAGS = -O3 -Xcompiler "-fopenmp -MMD -MP" -I./include $(ARCH_FLAGS)
LDFLAGS = -Xcompiler "-fopenmp"

SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin

COMMON_OBJS = $(OBJ_DIR)/common.o \
              $(OBJ_DIR)/kMeans.o \
              $(OBJ_DIR)/knnApprox.o \
              $(OBJ_DIR)/knn.o

MAIN_BIN = $(BIN_DIR)/main
COMPARE_BIN = $(BIN_DIR)/compare
ALGO_RUN_BIN = $(BIN_DIR)/runAlgo
A2_BIN = a2

default: $(A2_BIN) $(MAIN_BIN)

all: $(A2_BIN) $(MAIN_BIN) $(COMPARE_BIN) $(ALGO_RUN_BIN)

$(MAIN_BIN): $(OBJ_DIR)/main.o $(COMMON_OBJS) | $(BIN_DIR)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(A2_BIN): $(OBJ_DIR)/main.o $(COMMON_OBJS)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(COMPARE_BIN): $(OBJ_DIR)/compare.o $(COMMON_OBJS) | $(BIN_DIR)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ALGO_RUN_BIN): $(OBJ_DIR)/runAlgo.o $(COMMON_OBJS) | $(BIN_DIR)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

-include $(OBJ_DIR)/*.d

compare: $(COMPARE_BIN)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(A2_BIN)

.PHONY: default all clean compare