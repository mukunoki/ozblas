# Makefile
include ../../make.inc

NVCC	= $(CUDA_PATH)/bin/nvcc 
AR		= ar
LIB		:= ../../lib/libcuozblas.a
FLAGS   := -Wall -MD -O0 -Wno-maybe-uninitialized -I$(INC_PATH) 
NVCCFLAGS   := -O0 --expt-relaxed-constexpr #--ptxas-options=-v
#--ptxas-options=-v #--compiler-options='-finstrument-functions' -g

ifeq ($(CC30),yes)
NVCCFLAGS += -gencode arch=compute_30,code=sm_30
endif
ifeq ($(CC32),yes)
NVCCFLAGS += -gencode arch=compute_32,code=sm_32
endif
ifeq ($(CC35),yes)
NVCCFLAGS += -gencode arch=compute_35,code=sm_35
endif
ifeq ($(CC37),yes)
NVCCFLAGS += -gencode arch=compute_37,code=sm_37
endif
ifeq ($(CC50),yes)
NVCCFLAGS += -gencode arch=compute_50,code=sm_50
endif
ifeq ($(CC52),yes)
NVCCFLAGS += -gencode arch=compute_52,code=sm_52
endif
ifeq ($(CC53),yes)
NVCCFLAGS += -gencode arch=compute_53,code=sm_53
endif
ifeq ($(CC60),yes)
NVCCFLAGS += -gencode arch=compute_60,code=sm_60
endif
ifeq ($(CC61),yes)
NVCCFLAGS += -gencode arch=compute_61,code=sm_61
endif
ifeq ($(CC70),yes)
NVCCFLAGS += -gencode arch=compute_60,code=sm_70
endif
ifeq ($(CC75),yes)
NVCCFLAGS += -gencode arch=compute_60,code=sm_75
endif
ifeq ($(CC80),yes)
NVCCFLAGS += -gencode arch=compute_60,code=sm_80
endif
ifeq ($(CC86),yes)
NVCCFLAGS += -gencode arch=compute_60,code=sm_86
endif

all: blas

ALLSRC += \
cuozblas_blas_wrapper.cu \
cuozblas_split.cu \
cuozblas_sum.cu \
cuozblas_aux.cu 

ALLHDR = \
cuozblas_common.h \
cuozblas_internal.h \
cueft.h 

#ifeq ($(GEMM),yes) common
ALLSRC += cuozblas_gemm.cu 
#endif
ifeq ($(AXPY),yes)
ALLSRC += cuozblas_axpy.cu 
endif
ifeq ($(DOT),yes)
ALLSRC += cuozblas_dot.cu 
endif
ifeq ($(NRM2),yes)
ALLSRC += cuozblas_nrm2.cu 
endif
ifeq ($(GEMV),yes)
ALLSRC += cuozblas_gemv.cu 
endif
ifeq ($(CSRMV),yes)
ALLSRC += cuozblas_csrmv.cu 
endif
ifeq ($(CG),yes)
ALLSRC += cuozblas_cg.cu 
endif

ALLOBJ = cuozblas.o
ALLOBJ += $(ALLSRC:.cu=.o)

$(ALLOBJ) : $(ALLHDR)

blas: $(ALLOBJ) 
	$(AR) rv $(LIB) $(ALLOBJ) 

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $< 
cuozblas.o: cuozblas.cpp
	$(CXX) $(FLAGS) -o $@ -c $< -I$(CUDA_PATH)/include

clean :
	$(RM) *.o *.d
	$(RM) $(LIB)
