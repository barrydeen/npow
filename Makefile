CUDA   := /opt/cuda
NVCC   := $(CUDA)/bin/nvcc
CXX    := g++-15
ARCH   := -gencode arch=compute_86,code=sm_86 \
          -gencode arch=compute_86,code=compute_86

all: npow

main.o: src/main.cpp src/json.h src/bech32.h src/sha256.cuh src/miner.h
	$(CXX) -O3 -std=c++17 -Wall -I/opt/cuda/include -c src/main.cpp -o main.o

miner.o: src/miner.cu src/sha256.cuh src/miner.h
	$(NVCC) -O3 -std=c++17 $(ARCH) -ccbin $(CXX) -c src/miner.cu -o miner.o

npow: main.o miner.o
	$(NVCC) -ccbin $(CXX) -o npow main.o miner.o -lsecp256k1

clean:
	rm -f npow main.o miner.o

.PHONY: all clean
