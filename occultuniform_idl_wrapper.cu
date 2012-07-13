#include "exofast_cuda_util.cuh"

#include "occultuniform_c_wrapper.cu"

extern "C" {
  #include <stdio.h>
  #include "idl_export.h"

  double occult_uniform_cuda(int argc, void *argv[]) 
  {
    occult_uniform_wrapper_c((double *) argv[0], (double *) argv[1],  (IDL_LONG64) argv[2], (IDL_LONG64) argv[3], (double *) argv[4] );  
   return -1; 
  }

} // end extern "C"

