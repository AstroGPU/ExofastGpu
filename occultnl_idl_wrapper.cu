#include "exofast_cuda_util.cuh"

#include "occultnl_c_wrapper.cu"

extern "C" {
  #include <stdio.h>
  #include "idl_export.h"

  double occultnl_cuda(int argc, void *argv[]) 
  {
    occultnl_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (double *) argv[4],  (double *) argv[5],  (IDL_LONG64) argv[6], (IDL_LONG64) argv[7], (double *) argv[8] );  
   return -1; 
  }

} // end extern "C"

