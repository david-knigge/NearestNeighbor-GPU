NVCC 				= nvcc
CPP					= g++

CU_FLAGS 		= -O3 -std=c++11

CU_SOURCES_1_0  = cuda_version_1_0_gpu.cu
CU_SOURCES_1_1  = cuda_version_1_1_gpu.cu
CU_SOURCES_2_0  = cuda_version_2_0_gpu.cu
CU_SOURCES_2_1  = cuda_version_2_1_gpu.cu

CU_OBJECTS_1_0 	= $(CU_SOURCES_1_0:%.cu=%.o)
CU_OBJECTS_1_1 	= $(CU_SOURCES_1_1:%.cu=%.o)
CU_OBJECTS_2_0 	= $(CU_SOURCES_2_0:%.cu=%.o)
CU_OBJECTS_2_1 	= $(CU_SOURCES_2_1:%.cu=%.o)

%.o:			%.cu
					$(NVCC) $(CU_FLAGS) -c $< -o $@

# %.o:			%.cpp
# 				$(CPP) $(CPP_FLAGS) -c $< -o $@

all: 			nns10 nns11 nns20 nns21

nns10:		$(CU_OBJECTS_1_0)
					$(NVCC) $^ -o $@

nns11:		$(CU_OBJECTS_1_1)
					$(NVCC) $^ -o $@

nns20: 		$(CU_OBJECTS_2_0)
					$(NVCC) $^ -o $@

nns21:		$(CU_OBJECTS_2_1)
					$(NVCC) $^ -o $@

clean:
					rm -f *.o nns10 nns11 nns20 nns21 *~
