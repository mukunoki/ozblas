# Makefile
include ../../make.inc

AR		= ar
LIB		:= ../../lib/libozblas.a
ifneq ($(CXX),FCC)
FLAGS	:= -Wall -Wno-maybe-uninitialized
endif
FLAGS   += -MD -O3 -fPIC -I$(INC_PATH) 

#--------------------
# ARCH and COMPILER
#--------------------
ifeq ($(CXX),icpc)
	FLAGS += -fma -fp-model source -fprotect-parens -qopenmp #-Qoption,cpp,--extended_float_types
ifeq ($(CPU_ARCH),KNL)
	FLAGS += -xMIC-AVX512
endif
ifeq ($(CPU_ARCH),SKYLAKE)
	FLAGS += -xCORE-AVX2 -mtune=skylake-avx512 
endif
ifeq ($(CPU_ARCH),ICELAKE)
	FLAGS += -march=icelake-server
endif
endif
	# Warning 1: fprotect-parens MUST be used (g++ enables it in default)
	# Warning 2: fp-model source MUST be used (disable 80-bit register)
ifeq ($(CXX),g++)
	FLAGS += -fopenmp 
ifeq ($(CPU_ARCH),SKYLAKE)
	FLAGS += -march=skylake
endif
ifeq ($(CPU_ARCH),ICELAKE)
	FLAGS += -march=icelake-server
endif
endif
ifeq ($(CXX),FCC)
	FLAGS += -Kfenv_access,openmp,SVE 
endif

ifeq ($(CPU_ARCH),ARM)
	FLAGS += -DARM
#	FLAGS += -I/opt/FJSVxtclanga/tcsds-1.2.33/include
endif

#--------------------
# BLAS
#--------------------
ifeq ($(CPU_BLAS),MKL)
	FLAGS += -DMKL 
endif
ifeq ($(CPU_BLAS),SSL2)
	FLAGS += -DSSL2 
endif
ifeq ($(CUBLAS),yes)
	FLAGS += -DCUBLAS -lcudart -lcublas -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64
endif
ifeq ($(MPLAPACK),yes)
	FLAGS += -DMPLAPACK
endif

#--------------------
# PREC
#--------------------
ifeq ($(PREC_Q_D),yes)
	PREC_Q=yes
endif
ifeq ($(PREC_Q_S),yes)
	PREC_Q=yes
endif
ifeq ($(PREC_Q),yes)
ifeq ($(CPU_ARCH),ARM)
else
	FLAGS += -lquadmath 
endif
endif

all: blas

ALLSRC += \
ozblas.cpp \
ozblas_blas_wrapper.cpp \
ozblas_split.cpp \
ozblas_sum.cpp \
ozblas_aux.cpp 

ALLHDR = \
ozblas_common.h \
ozblas_internal.h 

#ifeq ($(GEMM),yes) // common
ALLSRC += ozblas_gemm.cpp 
#endif
ifeq ($(AXPY),yes)
ALLSRC += ozblas_axpy.cpp 
endif
ifeq ($(DOT),yes)
ALLSRC += ozblas_dot.cpp 
endif
ifeq ($(NRM2),yes)
ALLSRC += ozblas_nrm2.cpp 
endif
ifeq ($(GEMV),yes)
ALLSRC += ozblas_gemv.cpp 
endif
ifeq ($(GEMV),yes)
ALLSRC += ozblas_gemv.cpp 
endif
ifeq ($(CSRMV),yes)
ALLSRC += ozblas_csrmv.cpp 
endif
ifeq ($(CG),yes)
ALLSRC += ozblas_cg.cpp 
endif

ALLOBJ += $(ALLSRC:.cpp=.o)

$(ALLOBJ) : $(ALLHDR)

blas: $(ALLOBJ) 
	$(AR) rv $(LIB) $(ALLOBJ) 

%.o: %.cpp
	$(CXX) $(FLAGS) -o $@ -c $< 

clean :
	$(RM) *.o *.d
	$(RM) $(LIB)
