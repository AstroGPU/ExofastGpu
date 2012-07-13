#include "exofast_cuda_util.cuh"

#include "occultquad_c_wrapper.cu"

extern "C" {
  #include <stdio.h>
  #include "idl_export.h"

  double occultquad_cuda(int argc, void *argv[]) 
  {
    occultquad_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (IDL_LONG64) argv[5], (IDL_LONG64) argv[6], (double *) argv[7], (double *) argv[8] );  
   return -1; 
  }

  double occultquad_only_cuda(int argc, void *argv[]) 
  {
    occultquad_only_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (IDL_LONG64) argv[5], (IDL_LONG64) argv[6], (double *) argv[7] );  
   return -1; 
  }

  double occult_uniform_slow_cuda(int argc, void *argv[]) 
  {
    occult_uniform_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (IDL_LONG64) argv[5], (IDL_LONG64) argv[6], (double *) argv[7] );  
   return -1; 
  }

} // end extern "C"

