#include "exofast_cuda_util.cuh"

#include "chisq_c_wrapper.cu"

extern "C" {
  #include <stdio.h>
  #include "idl_export.h"

  double calc_chisq_cuda(int argc, void *argv[]) 
  {
    calc_chisq_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (IDL_LONG64) argv[3], (IDL_LONG64) argv[4], (double *) argv[5]);  
   return -1; 
  }

} // end extern "C"

