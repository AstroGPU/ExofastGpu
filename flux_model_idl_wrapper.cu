#include "exofast_cuda_util.cuh"

#include "flux_model_c_wrapper.cu"

extern "C" {
  #include <stdio.h>
  #include "idl_export.h"

  double flux_model_wltt_cuda(int argc, void *argv[]) 
  {
    flux_model_wltt_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (double *) argv[4],  (double *) argv[5],  (double *) argv[6],  (double *) argv[7],  (double *) argv[8],  (double *) argv[9], (double *) argv[10], (IDL_LONG64) argv[11], (IDL_LONG64) argv[12], (double *) argv[13]);  
   return -1; 
  }

  double flux_model_cuda(int argc, void *argv[]) 
  {
    flux_model_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (double *) argv[4],  (double *) argv[5],  (double *) argv[6],  (double *) argv[7],  (double *) argv[8],  (double *) argv[9], (IDL_LONG64) argv[10], (IDL_LONG64) argv[11], (double *) argv[12]);  
   return -1; 
  }

  double chisq_flux_model_wltt_cuda(int argc, void *argv[]) 
  {
    chisq_flux_model_wltt_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (double *) argv[4],  (double *) argv[5],  (double *) argv[6],  (double *) argv[7],  (double *) argv[8],  (double *) argv[9], (double *) argv[10], (double *) argv[11], (double *) argv[12], (IDL_LONG64) argv[13], (IDL_LONG64) argv[14], (double *) argv[15]);  
   return -1; 
  }

  double chisq_flux_model_cuda(int argc, void *argv[]) 
  {
    chisq_flux_model_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (double *) argv[4],  (double *) argv[5],  (double *) argv[6],  (double *) argv[7],  (double *) argv[8],  (double *) argv[9], (double *) argv[10], (double *) argv[11], (IDL_LONG64) argv[12], (IDL_LONG64) argv[13], (double *) argv[14]);  
   return -1; 
  }

} // end extern "C"

