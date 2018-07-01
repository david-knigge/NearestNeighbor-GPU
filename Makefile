NVCC 				= nvcc
CPP					= g++

CU_FLAGS 		= -O3 -std=c++11

CU_SOURCES_1_0  = cuda_version_1_0_gpu.cu
CU_SOURCES_1_1  = cuda_version_1_1_gpu.cu
CU_SOURCES_2_0  = cuda_version_2_0_gpu.cu
CU_SOURCES_2_1  = cuda_version_2_1_gpu.cu
CU_SOURCES_3_0  = cuda_version_3_0_gpu.cu

CU_OBJECTS_1_0 	= $(CU_SOURCES_1_0:%.cu=%.o)
CU_OBJECTS_1_1 	= $(CU_SOURCES_1_1:%.cu=%.o)
CU_OBJECTS_2_0 	= $(CU_SOURCES_2_0:%.cu=%.o)
CU_OBJECTS_2_1 	= $(CU_SOURCES_2_1:%.cu=%.o)
CU_OBJECTS_3_0 	= $(CU_SOURCES_3_0:%.cu=%.o)

%.o:			%.cu
					$(NVCC) $(CU_FLAGS) -c $< -o $@

# %.o:			%.cpp
# 				$(CPP) $(CPP_FLAGS) -c $< -o $@

all: 			nns_gpu_version_1_0 nns_gpu_version_1_1 nns_gpu_version_2_0 nns_gpu_version_2_1 nns_gpu_version_3_0

nns_gpu_version_1_0:		$(CU_OBJECTS_1_0)
					$(NVCC) $^ -o $@

nns_gpu_version_1_1:		$(CU_OBJECTS_1_1)
					$(NVCC) $^ -o $@

nns_gpu_version_2_0: 		$(CU_OBJECTS_2_0)
					$(NVCC) $^ -o $@

nns_gpu_version_2_1:		$(CU_OBJECTS_2_1)
					$(NVCC) $^ -o $@

nns_gpu_version_3_0:		$(CU_OBJECTS_3_0)
					$(NVCC) $^ -o $@

clean:
					rm -f *.o nns_gpu_version_1_0 nns_gpu_version_1_1 nns_gpu_version_2_0 nns_gpu_version_2_1 nns_gpu_version_3_0 *~
