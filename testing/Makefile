# Makefile
include ../make.inc

# for performance comparison
PREC_DD = no
GEMMEX  = no
PREC_H_H = no
POWER_API = no

# common -------------------------------------
ifneq ($(CXX),FCC)
FLAGS	:= -Wall -Wno-unused-but-set-variable 
endif
FLAGS	+= -O2 -MD 
INCS	+= -I$(INC_PATH) -I../include
LDFLAGS += -L$(LIB_PATH) -L../lib

# -------------------------------------------
ifeq ($(CUDA),yes)
OBJS	+= $(CUOZBLAS_OBJS) 
OBJS	+= $(CUBLAS_OBJS) 
INCS	+= -I$(CUDA_PATH)/include 
LDFLAGS	+= -L$(CUDA_PATH)/lib64
LIBS	+= -lcublasLt -lcudart -lcublas -lcusparse -lcuozblas
endif
ifeq ($(CUBLAS),yes)
INCS	+= -I$(CUDA_PATH)/include 
LDFLAGS	+= -L$(CUDA_PATH)/lib64 
LIBS	+= -lcudart -lcublas
FLAGS	+= -DCUBLAS
endif

ifeq ($(CPU),yes)
OBJS	+= $(CBLAS_OBJS) 
OBJS	+= $(OZBLAS_OBJS) 
LIBS	+= -lozblas 
ifeq ($(CPU_BLAS),MKL)
FLAGS	+= -DMKL
ifeq ($(LINKAGE),static)
LIBS	+= -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group 
else
LIBS	+= -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core #-littnotify#-liomp5 
# note: gcc with MKL may not work with OMP?
endif
#===========FOR VTUNE=========================
#LIBS	+= -littnotify
#FLAGS 	+= -I/work/opt/local/x86_64/cores/intel/2022.1.2/vtune/2022.0.0/include -L/work/opt/local/x86_64/cores/intel/2022.1.2/vtune/2022.0.0/lib64
#===========FOR VTUNE=========================
endif
ifeq ($(CPU_BLAS),OpenBLAS)
FLAGS	+= -DOpenBLAS
ifeq ($(LINKAGE),static)
LIBS	+= -Wl,-Bstatic,-lopenblas
else
LIBS	+= -lopenblas
endif
#LIBS 	+= -lgfortran 
endif
ifeq ($(CPU_BLAS),SSL2)
ifeq ($(CXX),FCC)
FLAGS	+= -DSSL2 -SSL2BLAMP
else
LIBS	+= -lfjsrcinfo -lc++abi -lclang_rt.asan-aarch64 -lfj90pxrscif -lfjstringsve -lclang_rt.hwasan-aarch64 -lfj90rt -lfjlapackexsve -lfmtl -lclang_rt.scudo-aarch64 -lfjarmci -loptf -lclang_rt.ubsan_minimal-aarch64 -lfjc++abi -lfjpthr -lstdtl -lclang_rt.ubsan_standalone-aarch64 -lfjcex -lcpubind -lfjcrt -lfjrtcl -lc++ -lfjc++ -lfjompcrt -lfj90f -lfjdemgl -lfjomphk -lfj90icaf -lfjomp -lfjsrcinfo -lfj90i
#LIBS	+= -lc++abi -lfj90i -lfjlapackex -lfjprofcore -lfjstring_internal -lopen-pal_mt -lclang_rt.asan-aarch64 -lfj90pxrscif -lfjlapackexsve_ilp64 -lfjprofmpif -lfjstringsve -lopen-pal -lclang_rt.hwasan-aarch64 -lfj90rt -lfjlapackexsve -lfjprofmpi -lfmtl -lopen-rte -lclang_rt.scudo-aarch64 -lfjarmci -lfjlapack_ilp64 -lfjprofomp -lmpi_cxx -loptf -lclang_rt.ubsan_minimal-aarch64 -lfjc++abi -lfjlapack -lfjpthr -lmpi_java -lstdtl -lclang_rt.ubsan_standalone-aarch64 -lfjcex -lfjlapacksve_ilp64 -lfjrtcl_io -lmpi_mpifh -lcpubind -lfjcrt -lfjlapacksve -lfjrtcl -lmpi_mt -lc++ -lfjc++ -lfjompcrt -lfjscalapack -lmpi -lfj90f -lfjdemgl -lfjomphk -lfjscalapacksve -lmpi_usempif08 -lfj90icaf -lfjlapackex_ilp64 -lfjomp -lfjsrcinfo -lmpi_usempi_ignore_tkr
endif
else
PREC_H_H = no
endif
endif

ifeq ($(CPU_ARCH),ARM)
FLAGS	+= -DARM 
ifeq ($(CXX),FCC)
FLAGS	+= -Nclang -Kopenmp -std=c++11 #-Nfjomplib -Nclang -Kopemp -march=armv8-a+sve 
endif
LDFLAGS	+= -L/opt/FJSVxtclanga/tcsds-1.2.33/lib64
INCS	+= -I/opt/FJSVxtclanga/tcsds-1.2.33/include
ifeq ($(POWER_API),yes)
FLAGS	+= -DPOWER_API
LIBS	+= -Wl,-Bdynamic,-lpwr
LDFLAGS	+= -L/opt/FJSVtcs/pwrm/aarch64/lib64
INCS	+= -I/opt/FJSVtcs/pwrm/aarch64/include
endif
endif

#=========================================
# SPARSE, FLOAT128
ifeq ($(CSRMV),yes)
SPARSE=yes
endif
ifeq ($(CG),yes)
SPARSE=yes
endif
ifeq ($(MPLAPACK),no)
PREC_Q_D = no
PREC_Q_S = no
PREC_DD = no
endif

ifeq ($(MPLAPACK),yes)
ifeq ($(PREC_Q_D),yes)
FLOAT128=yes
endif
ifeq ($(PREC_Q_S),yes)
FLOAT128=yes
endif
endif

ifeq ($(PREC_DD),yes)
ifeq ($(MPLAPACK),yes)
ifeq ($(LINKAGE),static)
LIBS	+= -Wl,-Bstatic,-lqd,-lmpblas_dd_opt
else
LIBS	+= -lqd -lmpblas_dd_opt
endif
endif
endif

ifeq ($(FLOAT128),yes)
ifeq ($(LINKAGE),static)
LIBS	+= -Wl,-Bstatic,-lmpblas__Float128_opt
else
LIBS	+= -lmpblas__Float128_opt
endif
ifeq ($(CPU_ARCH),ARM)
else
LIBS	+= -Wl,-Bdynamic,-lquadmath 
endif
endif

ifeq ($(SPARSE),yes)
ifeq ($(LINKAGE),static)
LIBS    += -Wl,-Bstatic,-lsparse_matrix_converter,-lbebop_util
else
LIBS    += -lsparse_matrix_converter -lbebop_util
endif
endif
#=========================================

#=========================================
ifeq ($(MPLAPACK),yes)
OBJS	+= $(MPBLAS_OBJS) 
ifeq ($(LINKAGE),static)
LIBS	+= -Wl,-Bstatic,-lmpblas_mpfr_opt -lmpfr
else
LIBS	+= -lmpblas_mpfr_opt -lmpfr
endif
LIBS	+= -lmpc -lgmp -std=c++11 -std=gnu++17
FLAGS	+= -DMPLAPACK
endif

ifeq ($(LINKAGE),static)
LIBS	+= -Wl,-Bdynamic
endif
LIBS	+= -lstdc++ -lm -lc -ldl 

# compiler -----------------------------------
ifeq ($(CXX),icpc)
FLAGS	+= -qopenmp
ifeq ($(CPU_ARCH),KNL)
	FLAGS += -xMIC-AVX512
endif
ifeq ($(CPU_ARCH),SKYLAKE)
	FLAGS += -xCORE-AVX2 -mtune=skylake-avx512 
endif
endif
ifeq ($(CXX),g++)
	FLAGS += -fopenmp 
ifeq ($(CPU_ARCH),SKYLAKE)
	FLAGS += -march=skylake
endif
endif
#=========================================

# DOT  ---------------------------------------
ifeq ($(DOT),yes)
ifeq ($(PREC_H_H),yes)
CBLAS_OBJS += testing_cblas_hdot
endif
ifeq ($(PREC_S_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_sdot
OZBLAS_OBJS += testing_ozblas_sdot
CUBLAS_OBJS += testing_cublas_sdot
CBLAS_OBJS += testing_cblas_sdot
endif
ifeq ($(PREC_S_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_sddot
OZBLAS_OBJS += testing_ozblas_sddot
endif
ifeq ($(PREC_D_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_dsdot
OZBLAS_OBJS += testing_ozblas_dsdot
endif
ifeq ($(PREC_D_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_ddot
OZBLAS_OBJS += testing_ozblas_ddot
CUBLAS_OBJS += testing_cublas_ddot
CBLAS_OBJS += testing_cblas_ddot
endif
ifeq ($(PREC_Q_D),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qddot
OZBLAS_OBJS += testing_ozblas_qddot
MPBLAS_OBJS += testing_mpblas_rdot_b128
endif
ifeq ($(PREC_Q_S),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qsdot
OZBLAS_OBJS += testing_ozblas_qsdot
endif
ifeq ($(PREC_DD),yes)
MPBLAS_OBJS += testing_mpblas_rdot_dd
endif
endif

# GEMV ---------------------------------------
ifeq ($(GEMV),yes)
ifeq ($(PREC_H_H),yes)
CBLAS_OBJS += testing_cblas_hgemv
endif
ifeq ($(PREC_S_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_sgemv
OZBLAS_OBJS += testing_ozblas_sgemv
CUBLAS_OBJS += testing_cublas_sgemv
CBLAS_OBJS += testing_cblas_sgemv
endif
ifeq ($(PREC_S_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_sdgemv
OZBLAS_OBJS += testing_ozblas_sdgemv
endif
ifeq ($(PREC_D_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_dsgemv
OZBLAS_OBJS += testing_ozblas_dsgemv
endif
ifeq ($(PREC_D_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_dgemv
OZBLAS_OBJS += testing_ozblas_dgemv
CUBLAS_OBJS += testing_cublas_dgemv
CBLAS_OBJS += testing_cblas_dgemv
endif
ifeq ($(PREC_Q_D),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qdgemv
OZBLAS_OBJS += testing_ozblas_qdgemv
MPBLAS_OBJS += testing_mpblas_rgemv_b128
endif
ifeq ($(PREC_Q_S),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qsgemv
OZBLAS_OBJS += testing_ozblas_qsgemv
endif
ifeq ($(PREC_DD),yes)
MPBLAS_OBJS += testing_mpblas_rgemv_dd
endif
endif

# CSRMV ---------------------------------------
ifeq ($(CSRMV),yes)
ifeq ($(PREC_H_H),yes)
#CBLAS_OBJS += testing_cblas_hcsrmv
endif
ifeq ($(PREC_S_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_scsrmv
OZBLAS_OBJS += testing_ozblas_scsrmv
CBLAS_OBJS += testing_cblas_scsrmv
endif
ifeq ($(PREC_S_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_sdcsrmv
OZBLAS_OBJS += testing_ozblas_sdcsrmv
endif
ifeq ($(PREC_D_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_dscsrmv
OZBLAS_OBJS += testing_ozblas_dscsrmv
endif
ifeq ($(PREC_D_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_dcsrmv
OZBLAS_OBJS += testing_ozblas_dcsrmv
CBLAS_OBJS += testing_cblas_dcsrmv
endif
ifeq ($(PREC_Q_D),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qdcsrmv
OZBLAS_OBJS += testing_ozblas_qdcsrmv
endif
ifeq ($(PREC_Q_S),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qscsrmv
OZBLAS_OBJS += testing_ozblas_qscsrmv
endif
endif

# CSRMM ---------------------------------------
ifeq ($(CSRMM),yes)
ifeq ($(PREC_H_H),yes)
#CBLAS_OBJS += testing_cblas_hcsrmm
endif
ifeq ($(PREC_S_S),yes)
#CUOZBLAS_OBJS += testing_cuozblas_scsrmm
#OZBLAS_OBJS += testing_ozblas_scsrmm
CBLAS_OBJS += testing_cblas_scsrmm
endif
ifeq ($(PREC_S_D),yes)
#CUOZBLAS_OBJS += testing_cuozblas_sdcsrmm
#OZBLAS_OBJS += testing_ozblas_sdcsrmm
endif
ifeq ($(PREC_D_S),yes)
#CUOZBLAS_OBJS += testing_cuozblas_dscsrmm
#OZBLAS_OBJS += testing_ozblas_dscsrmm
endif
ifeq ($(PREC_D_D),yes)
#CUOZBLAS_OBJS += testing_cuozblas_dcsrmm
#OZBLAS_OBJS += testing_ozblas_dcsrmm
CBLAS_OBJS += testing_cblas_dcsrmm
endif
ifeq ($(PREC_Q_D),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qdcsrmm
#OZBLAS_OBJS += testing_ozblas_qdcsrmm
endif
ifeq ($(PREC_Q_S),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qscsrmm
#OZBLAS_OBJS += testing_ozblas_qscsrmm
endif
endif

# GEMM ---------------------------------------
ifeq ($(GEMM),yes)
ifeq ($(PREC_H_H),yes)
CBLAS_OBJS += testing_cblas_hgemm
endif
ifeq ($(PREC_S_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_sgemm
OZBLAS_OBJS += testing_ozblas_sgemm
CUBLAS_OBJS += testing_cublas_sgemm
CBLAS_OBJS += testing_cblas_sgemm
endif
ifeq ($(PREC_S_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_sdgemm
OZBLAS_OBJS += testing_ozblas_sdgemm
endif
ifeq ($(PREC_D_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_dsgemm
OZBLAS_OBJS += testing_ozblas_dsgemm
endif
ifeq ($(PREC_D_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_dgemm
OZBLAS_OBJS += testing_ozblas_dgemm
CUBLAS_OBJS += testing_cublas_dgemm
CBLAS_OBJS += testing_cblas_dgemm
endif
ifeq ($(PREC_Q_D),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qdgemm
OZBLAS_OBJS += testing_ozblas_qdgemm
MPBLAS_OBJS += testing_mpblas_rgemm_b128
endif
ifeq ($(PREC_Q_S),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qsgemm
OZBLAS_OBJS += testing_ozblas_qsgemm
endif
ifeq ($(PREC_DD),yes)
MPBLAS_OBJS += testing_mpblas_rgemm_dd
endif
endif
ifeq ($(GEMMEX),yes)
CUBLAS_OBJS += \
testing_cublasGemmEx_32f_pdt_32f \
testing_cublasGemmEx_32f_tc_32f \
testing_cublasGemmEx_32f_tc_32tf \
testing_cublasGemmEx_64f_pdt_64f \
testing_cublasGemmEx_64f_tc_64f \
testing_cublasGemmEx_16f_tc_32f \
testing_cublasGemmEx_16f_pdt_32f \
testing_cublasGemmEx_16f_tc_16f \
testing_cublasGemmEx_16f_pdt_16f 
endif

# CG -----------------------------------------
ifeq ($(CG),yes)
ifeq ($(PREC_S_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_scg
OZBLAS_OBJS += testing_ozblas_scg
endif
ifeq ($(PREC_S_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_sdcg
OZBLAS_OBJS += testing_ozblas_sdcg
endif
ifeq ($(PREC_D_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_dscg
OZBLAS_OBJS += testing_ozblas_dscg
endif
ifeq ($(PREC_D_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_dcg
OZBLAS_OBJS += testing_ozblas_dcg
endif
ifeq ($(PREC_Q_D),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qdcg
OZBLAS_OBJS += testing_ozblas_qdcg
endif
ifeq ($(PREC_Q_S),yes)
#CUOZBLAS_OBJS += testing_cuozblas_qscg
OZBLAS_OBJS += testing_ozblas_qscg
endif
endif

default: $(OBJS)
ifeq ($(CPU),yes)
$(OBJS): ../lib/libozblas.a
endif
ifeq ($(CUDA),yes)
$(OBJS): ../lib/libcuozblas.a
endif
$(OBJS): \
testing_common.cpp \
testing_common.h \
testing_setting.h

SRC_L1B	= testing_level1b.cpp 
SRC_L2	= testing_level2.cpp 
SRC_L2S	= testing_level2s.cpp 
SRC_L3	= testing_level3.cpp 
SRC_L3S	= testing_level3s.cpp 
SRC_SS	= testing_solver.cpp 

### CBLAS ###################################################################################################
testing_cblas_hgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DGEMM -DPREC_H_H
testing_cblas_hgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DGEMV -DPREC_H_H
#testing_cblas_hcsrmv: $(SRC_L2S)
#	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DCSRMV -DPREC_H_H
testing_cblas_hdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DDOT -DPREC_H_H
#testing_cblas_hcg: $(SRC_SS)
#	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DCG -DPREC_H_H 

testing_cblas_sgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DGEMM -DPREC_S_S
testing_cblas_sgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DGEMV -DPREC_S_S
testing_cblas_scsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DCSRMV -DPREC_S_S
testing_cblas_sdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DDOT -DPREC_S_S
#testing_cblas_scg: $(SRC_SS)
#	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DCG -DPREC_S_S 

testing_cblas_dgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DGEMM -DPREC_D_D
testing_cblas_dgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DGEMV -DPREC_D_D
testing_cblas_dcsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DCSRMV -DPREC_D_D
testing_cblas_dcsrmm: $(SRC_L3S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DCSRMM -DPREC_D_D
testing_cblas_ddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DDOT -DPREC_D_D
#testing_cblas_dcg: $(SRC_SS)
#	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCBLAS -DCG -DPREC_D_D 

### CUBLAS ###################################################################################################
testing_cublas_sgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMM -DPREC_S_S
testing_cublas_sgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMV -DPREC_S_S
#testing_cublas_scsrmv: $(SRC_L2S)
#	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DCSRMV -DPREC_S_S
testing_cublas_sdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DDOT -DPREC_S_S
#testing_cublas_scg: $(SRC_SS)
#	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DCG -DPREC_S_S 

testing_cublas_dgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMM -DPREC_D_D
testing_cublas_dgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMV -DPREC_D_D
#testing_cublas_dcsrmv: $(SRC_L2S)
#	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DCSRMV -DPREC_D_D
testing_cublas_ddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DDOT -DPREC_D_D
#testing_cublas_dcg: $(SRC_SS)
#	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DCG -DPREC_D_D 

testing_cublasGemmEx_32f_pdt_32f: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMMEX -DPREC_32F_PDT_32F
testing_cublasGemmEx_32f_tc_32f: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMMEX -DPREC_32F_TC_32F
testing_cublasGemmEx_32f_tc_32tf: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMMEX -DPREC_32F_TC_32TF
testing_cublasGemmEx_16f_tc_32f: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMMEX -DPREC_16F_TC_32F 
testing_cublasGemmEx_16f_pdt_32f: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMMEX -DPREC_16F_PDT_32F
testing_cublasGemmEx_16f_tc_16f: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMMEX -DPREC_16F_TC_16F
testing_cublasGemmEx_16f_pdt_16f: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMMEX -DPREC_16F_PDT_16F
testing_cublasGemmEx_64f_pdt_64f: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMMEX -DPREC_64F_PDT_64F
testing_cublasGemmEx_64f_tc_64f: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUBLAS -DGEMMEX -DPREC_64F_TC_64F 

### CUOZBLAS ###################################################################################################
testing_cuozblas_sgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMM -DPREC_S_S
testing_cuozblas_sgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMV -DPREC_S_S
testing_cuozblas_scsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DCSRMV -DPREC_S_S
testing_cuozblas_sdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DDOT -DPREC_S_S
testing_cuozblas_scg: $(SRC_SS)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DCG -DPREC_S_S 

testing_cuozblas_sdgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMM -DPREC_S_D
testing_cuozblas_sdgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMV -DPREC_S_D
testing_cuozblas_sdcsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DCSRMV -DPREC_S_D
testing_cuozblas_sddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DDOT -DPREC_S_D
testing_cuozblas_sdcg: $(SRC_SS)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DCG -DPREC_S_D 

testing_cuozblas_dsgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMM -DPREC_D_S
testing_cuozblas_dsgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMV -DPREC_D_S
testing_cuozblas_dscsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DCSRMV -DPREC_D_S
testing_cuozblas_dsdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DDOT -DPREC_D_S
testing_cuozblas_dscg: $(SRC_SS)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DCG -DPREC_D_S 

testing_cuozblas_dgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMM -DPREC_D_D
testing_cuozblas_dgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMV -DPREC_D_D
testing_cuozblas_dcsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DCSRMV -DPREC_D_D
testing_cuozblas_ddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DDOT -DPREC_D_D
testing_cuozblas_dcg: $(SRC_SS)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DCG -DPREC_D_D 

### OZBLAS ###################################################################################################
testing_ozblas_sgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMM -DPREC_S_S
testing_ozblas_sgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMV -DPREC_S_S
testing_ozblas_scsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCSRMV -DPREC_S_S
testing_ozblas_sdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DDOT -DPREC_S_S
testing_ozblas_scg: $(SRC_SS)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCG -DPREC_S_S 

testing_ozblas_sdgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMM -DPREC_S_D
testing_ozblas_sdgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMV -DPREC_S_D
testing_ozblas_sdcsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCSRMV -DPREC_S_D
testing_ozblas_sddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DDOT -DPREC_S_D
testing_ozblas_sdcg: $(SRC_SS)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCG -DPREC_S_D

testing_ozblas_dsgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMM -DPREC_D_S
testing_ozblas_dsgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMV -DPREC_D_S
testing_ozblas_dscsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCSRMV -DPREC_D_S
testing_ozblas_dsdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DDOT -DPREC_D_S
testing_ozblas_dscg: $(SRC_SS)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCG -DPREC_D_S

testing_ozblas_dgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMM -DPREC_D_D
testing_ozblas_dgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMV -DPREC_D_D
testing_ozblas_dcsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCSRMV -DPREC_D_D
testing_ozblas_ddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DDOT -DPREC_D_D
testing_ozblas_dcg: $(SRC_SS)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCG -DPREC_D_D

testing_ozblas_qdgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMM -DPREC_Q_D
testing_ozblas_qdgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMV -DPREC_Q_D
testing_ozblas_qdcsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCSRMV -DPREC_Q_D
testing_ozblas_qddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DDOT -DPREC_Q_D
testing_ozblas_qdcg: $(SRC_SS)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCG -DPREC_Q_D

testing_ozblas_qsgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMM -DPREC_Q_S
testing_ozblas_qsgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMV -DPREC_Q_S
testing_ozblas_qscsrmv: $(SRC_L2S)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCSRMV -DPREC_Q_S
testing_ozblas_qsdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DDOT -DPREC_Q_S
testing_ozblas_qscg: $(SRC_SS)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DCG -DPREC_Q_S

### MPBLAS ###################################################################################################
testing_mpblas_rdot_b128: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DMPBLAS -DDOT -DPREC_Q_D
testing_mpblas_rgemv_b128: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DMPBLAS -DGEMV -DPREC_Q_D
testing_mpblas_rgemm_b128: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DMPBLAS -DGEMM -DPREC_Q_D

testing_mpblas_rdot_dd: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DMPBLAS -DDOT -DPREC_DD
testing_mpblas_rgemv_dd: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DMPBLAS -DGEMV -DPREC_DD
testing_mpblas_rgemm_dd: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DMPBLAS -DGEMM -DPREC_DD


clean:
	$(RM) $(OBJS)
	$(RM) *.d 

-include *.d
.PHONY: all clean

