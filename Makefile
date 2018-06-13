NVCC 				= nvcc
CPP					= g++

CU_FLAGS 		= -O3 -std=c++11

CU_SOURCES  = nns.cu nns2.cu 

CU_OBJECTS 	= $(CU_SOURCES:%.cu=%.o)

%.o:			%.cu
				$(NVCC) $(CU_FLAGS) -c $< -o $@

# %.o:			%.cpp
# 				$(CPP) $(CPP_FLAGS) -c $< -o $@

all: 		nns

nns:		$(CU_OBJECTS)
				$(NVCC) $^ -o $@

clean:
			rm -f *.o nns *~
