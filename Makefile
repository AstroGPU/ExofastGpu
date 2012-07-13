
# Variables to set for local configuration
# Compile just for Fermi-based GPUs
GENCODE = -gencode arch=compute_20,\"code=sm_20,compute_20\"
# For printing more info about memory requirements
# GENCODE = -gencode arch=compute_20,\"code=sm_20,compute_20\" --ptxas-options=-v
# Compile for either Fermi or GT200-based GPUs (gives lots of warning messages you can ignore)
# GENCODE = -gencode arch=compute_13,\"code=sm_13,compute_13\" -gencode arch=compute_20,\"code=sm_20,compute_20\"
# If not using the GPU, can use OpenMP in place of GPU
# GENCODE = -Xcompiler -fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP -lgomp -O3

INCPATHS =  -I/scratch/eford/NVIDIA_GPU_Computing_SDK/C/common/inc/ -I. 
LIBS = -L/scratch/eford/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_x86_64 
OPTS = -O
IDLCMDS = -shared  --compiler-options ' -fPIC '

# short cuts
all:   demos idl
demos: keplereq_test.exe occultuniform_test.exe occultquad_test.exe occultnl_test.exe flux_model_test.exe
idl:   exofast_cuda.so
clean: 
	rm -f keplereq_test.exe occultquad_test.exe flux_model_test.exe exofast_cuda.so
tarball:
	tar czf exofast_cuda_xxx.tgz *.cu *.cuh  *.c *.cpp *.hpp *_test.pro Makefile

# Executables demonstrating C wrapper functions
flux_model_test.exe: flux_model_test.cu 
	nvcc flux_model_test.cu $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -o flux_model_test.exe

keplereq_test.exe:   keplereq_test.cu  keplereq.cu
	nvcc keplereq_test.cu $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -o keplereq_test.exe

occultquad_test.exe: occultquad_test.cu occultquad.cu
	nvcc occultquad_test.cu $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -o occultquad_test.exe

occultuniform_test.exe: occultuniform_test.cu occultuniform.cu
	nvcc occultuniform_test.cu $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -o occultuniform_test.exe

occultnl_test.exe: occultnl_test.cu occultnl.cu
	nvcc occultnl_test.cu $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) -o occultnl_test.exe

# Dependancies for executables demonstrating C wrapper functions
keplereq_test.cu:   keplereq_c_wrapper.cu   keplereq.cu
occultquad_test.cu: occultquad_c_wrapper.cu occultquad.cu
occultuniform_test.cu: occultuniform_c_wrapper.cu occultuniform.cu
occultnl_test.cu: occultnl_c_wrapper.cu occultnl.cu
flux_model_test.cu: flux_model_c_wrapper.cu occultquad.cu getb.cu keplereq.cu bjd.cu chisq.cu

# C Wrapper functions
bjd_c_wrapper.cu: bjd.cu
keplereq_c_wrapper.cu: keplereq.cu
getb_c_wrapper.cu: getb.cu keplereq.cu
occultquad_c_wrapper.cu: occultquad.cu
occultuniform_c_wrapper.cu: occultuniform.cu
occultnl_c_wrapper.cu: occultnl.cu
flux_model_c_wrapper.cu: occultquad.cu getb.cu keplereq.cu bjd.cu chisq.cu
chisq_c_wrapper.cu: chisq.cu

# IDL wrapper functions
bjd_idl_wrapper.cu: bjd_c_wrapper.cu
keplereq_idl_wrapper.cu: keplereq_c_wrapper.cu
getb_idl_wrapper.cu: getb_c_wrapper.cu
occultquad_idl_wrapper.cu: occultquad_c_wrapper.cu
occultuniform_idl_wrapper.cu: occultuniform_c_wrapper.cu
occultnl_idl_wrapper.cu: occultnl_c_wrapper.cu
flux_model_idl_wrapper.cu: flux_model_c_wrapper.cu
chisq_idl_wrapper.cu: chisq_c_wrapper.cu

# Library for accessing via IDL 
exofast_cuda.so: exofast_cuda_all_idl_wrappers.cu  flux_model_idl_wrapper.cu occultuniform_idl_wrapper.cu occultquad_idl_wrapper.cu occultnl_idl_wrapper.cu keplereq_idl_wrapper.cu getb_idl_wrapper.cu bjd_idl_wrapper.cu chisq_idl_wrapper.cu
	nvcc exofast_cuda_all_idl_wrappers.cu $(GENCODE) $(INCPATHS) $(LIBS) $(OPTS) $(IDLCMDS)  -o exofast_cuda.so

