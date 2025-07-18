##############################################

#--- System ----------------------------------
#CPU_ARCH: KNL, ICELAKE, SKYLAKE, ARM
CPU_ARCH =
# -> If specified, compile with architecture-specific options

#LINKAGE: static, dynamic
LINKAGE = static

#--- PATH ------------------------------------
CUDA_PATH	= /usr/local/cuda
LIB_PATH	= $(HOME)/local/lib
INC_PATH	= $(HOME)/local/include
# -> installation path of external libraries

#--- Compiler --------------------------------
CXX = g++
# -> Intel LLVM Compiler (icpx) is not supported

#--- CPU (OzBLAS) / CUDA (cuOzBLAS) ----------
CPU  = yes
CUDA = yes
# CPU must be yes also for CUDA

#--- BLAS (CPU) ------------------------------
#CPU_BLAS: OpenBLAS, MKL
#CPU_BLAS = MKL
CPU_BLAS = OpenBLAS
# CSRMV & CG require MKL because OpenBLAS does not have sparse routines.

#--- MPLAPACK (for testing) ------------------
MPLAPACK = yes
# -> yes or no. 'yes' uses MPLAPACK for verifying computational result
# -> If no, binary128 (PREC_Q_D and PREC_Q_S) cannot be used.
# MPLAPACK does not work with SSL2 on Fugaku

#--- CUDA Compute Capability -----------------
CC50 = no
CC52 = no
CC53 = no
CC60 = no
CC61 = no
CC62 = no
CC70 = no
CC75 = no
CC80 = no
CC86 = no
CC90 = yes
# -> Tested only on GPUs with CC70, CC75, and CC86

#--- Precision -------------------------------
# PREC_X_Y: X is I/O-precision, Y is compute-precision
# Q: FP128, D: FP64, S: FP32, H: FP16 
PREC_Q_D = yes
PREC_Q_S = yes
PREC_D_D = yes
PREC_D_S = yes
PREC_D_H = yes
PREC_S_S = yes
PREC_S_H = yes
PREC_S_D = yes
# -> D_H and S_H: GPU only
# -> Q_D and Q_S: CPU only

#--- Routine ---------------------------------
# level-1
DOT = yes
NRM2 = yes
AXPY = yes

# level-2
GEMV = yes
CSRMV = yes
# -> CSRMV requires MKL on CPUs

# level-3
GEMM = yes

# solver
CG = yes
# -> CG requires CSRMV, DOT, NRM2, and AXPY

##############################################

