CUDA_FILES = proposal.cu
CPP_FILES = mh_gpu.cpp mh_cpu.cpp main.cpp

CUDA_PATH = /usr/local/cuda-9.1
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr
NVCC_INCLUDE =
NVCC_LIBS = 
NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

CUDA_OBJ_FILES = $(addsuffix .o, $(CUDA_FILES))
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++0x -pthread
INCLUDE = -I$(CUDA_INC_PATH)
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft

OBJ_FILES = $(addsuffix .o, $(CPP_FILES))

all: $(OBJ_FILES) cuda.o $(CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o main $(INCLUDE) $^ $(LIBS) 

cuda.o: $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^

%.cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

%.cpp.o: %.cpp
	g++ $(FLAGS) -c -o $@ $(INCLUDE) $< 

clean:
	rm -f *.o all

.PHONY: clean

