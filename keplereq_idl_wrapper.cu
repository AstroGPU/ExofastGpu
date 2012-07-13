#include "exofast_cuda_util.cuh"

#include "keplereq_c_wrapper.cu"

extern "C" {
  #include <stdio.h>
  #include "idl_export.h"

  double keplereq_cuda(int argc, void *argv[]) 
  {
    keplereq_wrapper_c((double *) argv[0], (double *) argv[1], (IDL_LONG64) argv[2], (IDL_LONG64) argv[3], (double *) argv[4]);  
   return -1; 
  }

} // end extern "C"

