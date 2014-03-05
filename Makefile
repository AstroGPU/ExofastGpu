
# Variables to set for local configuration
# Compile just for Fermi-based GPUs
# COMPILER = nvcc
# GENCODE = -gencode arch=compute_20,\"code=sm_20,compute_20\"
# For printing more info about memory requirements
# GENCODE = -gencode arch=compute_20,\"code=sm_20,compute_20\" --ptxas-options=-v
# Compile for either Fermi or GT200-based GPUs (gives lots of warning messages you can ignore)
# GENCODE = -gencode arch=compute_13,\"code=sm_13,compute_13\" -gencode arch=compute_20,\"code=sm_20,compute_20\"
# If not using the GPU, can use OpenMP in place of GPU
# GENCODE = -Xcompiler -fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP -lgomp 
# If not using the GPU or CUDA, then can use gcc with OpenMP
COMPILER = g++
GENCODE = -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp

#INCPATHS =  -I/scratch/eford/NVIDIA_GPU_Computing_SDK/C/common/inc/ -I. 
INCPATHS =  -I/home/ebf11/include
#LIBS = -L/scratch/eford/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_x86_64 
LIBS = 
OPTS = -O3
IDLCMDS = -shared  --compiler-options ' -fPIC '

# short cuts
all:   demos libc idl
demos: keplereq_test.exe occultuniform_test.exe occultquad_test.exe occultnl_test.exe flux_model_test.exe
libc:   exofast_cuda_c.so
idl:   exofast_cuda_idl.so

clean: 
	rm -f keplereq_test.exe occultquad_test.exe flux_model_test.exe exofast_cuda_*.so
tarball:
	tar czf exofast_cuda_xxx.tgz *.cu *.cuh  *.c *.cpp *.hpp *_test.pro Makefile

# Executables demonstrating C wrapper functions
flux_model_test.exe: flux_model_test.cpp 
	$(COMPILER) flux_model_test.cpp $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -o flux_model_test.exe

keplereq_test.exe:   keplereq_test.cpp  keplereq.cu
	$(COMPILER) keplereq_test.cpp $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -o keplereq_test.exe

occultquad_test.exe: occultquad_test.cpp occultquad.cu
	$(COMPILER) occultquad_test.cpp $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -o occultquad_test.exe

occultuniform_test.exe: occultuniform_test.cpp occultuniform.cu
	$(COMPILER) occultuniform_test.cpp $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -o occultuniform_test.exe

occultnl_test.exe: occultnl_test.cpp occultnl.cu
	$(COMPILER) occultnl_test.cpp $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -o occultnl_test.exe

# Dependancies for executables demonstrating C wrapper functions
keplereq_test.cu:   keplereq_c_wrapper.cpp   keplereq.cu
occultquad_test.cu: occultquad_c_wrapper.cpp occultquad.cu
occultuniform_test.cu: occultuniform_c_wrapper.cpp occultuniform.cu
occultnl_test.cu: occultnl_c_wrapper.cpp occultnl.cu
flux_model_test.cu: flux_model_c_wrapper.cpp occultquad.cu getb.cu keplereq.cu bjd.cu chisq.cu

# C Wrapper functions
bjd_c_wrapper.cpp: bjd.cu
keplereq_c_wrapper.cpp: keplereq.cu
getb_c_wrapper.cpp: getb.cu keplereq.cu
occultquad_c_wrapper.cpp: occultquad.cu
occultuniform_c_wrapper.cpp: occultuniform.cu
occultnl_c_wrapper.cpp: occultnl.cu
flux_model_c_wrapper.cpp: occultquad.cu getb.cu keplereq.cu bjd.cu chisq.cu
chisq_c_wrapper.cpp: chisq.cu

# Library for accessing via C
exofast_cuda_c.so: exofast_cuda_all_c_wrappers.cpp  flux_model_c_wrapper.cpp occultuniform_c_wrapper.cpp occultquad_c_wrapper.cpp occultnl_c_wrapper.cpp keplereq_c_wrapper.cpp getb_c_wrapper.cpp bjd_c_wrapper.cpp chisq_c_wrapper.cpp
	$(COMPILER) exofast_cuda_all_c_wrappers.cpp $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -c   -o exofast_cuda_c.so


# IDL wrapper functions
bjd_idl_wrapper.cu: bjd.cu
keplereq_idl_wrapper.cu: keplereq.cu
getb_idl_wrapper.cu: getb.cu
occultquad_idl_wrapper.cu: occultquad.cu
occultuniform_idl_wrapper.cu: occultuniform.cu
occultnl_idl_wrapper.cu: occultnl.cu
flux_model_idl_wrapper.cu: occultquad.cu getb.cu keplereq.cu bjd.cu chisq.cu
chisq_idl_wrapper.cu: chisq.cu

# Library for accessing via IDL 
exofast_cuda_idl.so: exofast_cuda_all_idl_wrappers.cu  flux_model_idl_wrapper.cu occultuniform_idl_wrapper.cu occultquad_idl_wrapper.cu occultnl_idl_wrapper.cu keplereq_idl_wrapper.cu getb_idl_wrapper.cu bjd_idl_wrapper.cu chisq_idl_wrapper.cu
	$(COMPILER) exofast_cuda_all_idl_wrappers.cu $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) $(IDLCMDS)  -o exofast_cuda_idl.so

