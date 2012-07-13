#include "exofast_cuda_util.cuh"

#include "getb_c_wrapper.cu"

extern "C" {
  #include <stdio.h>
  #include "idl_export.h"

  double get_xyz_cuda(int argc, void *argv[]) 
  {
    get_xyz_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (double *) argv[4],  (double *) argv[5],  (double *) argv[6],  (IDL_LONG64) argv[7], (IDL_LONG64) argv[8], (double *) argv[9], (double *) argv[10], (double *) argv[11] );  
   return -1; 
  }

  double get_b_cuda(int argc, void *argv[]) 
  {
    get_b_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (double *) argv[4],  (double *) argv[5],  (double *) argv[6],  (IDL_LONG64) argv[7], (IDL_LONG64) argv[8], (double *) argv[9]);  
   return -1; 
  }

  double get_b_depth_cuda(int argc, void *argv[]) 
  {
    get_b_depth_wrapper_c((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (double *) argv[4],  (double *) argv[5],  (double *) argv[6],  (IDL_LONG64) argv[7], (IDL_LONG64) argv[8], (double *) argv[9], (double *) argv[10] );  
   return -1; 
  }

} // end extern "C"

