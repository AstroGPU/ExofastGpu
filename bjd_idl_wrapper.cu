#include "exofast_cuda_util.cuh"
#include "bjd_c_wrapper.cu"

extern "C" {
  #include <stdio.h>
  #include "idl_export.h"

  double bjd_target_secondary_cuda(int argc, void *argv[]) 
  {
    bjd_target_wrapper_c<false>((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (double *) argv[4],  (double *) argv[5],  (double *) argv[6],  (IDL_LONG64) argv[7], (IDL_LONG64) argv[8], (double *) argv[9] );  
   return -1; 
  }

  double bjd_target_primary_cuda(int argc, void *argv[]) 
  {
    bjd_target_wrapper_c<true>((double *) argv[0], (double *) argv[1],  (double *) argv[2],  (double *) argv[3],  (double *) argv[4],  (double *) argv[5],  (double *) argv[6],  (IDL_LONG64) argv[7], (IDL_LONG64) argv[8], (double *) argv[9] );  
   return -1; 
  }

  // I'm not sure if this is correct syntax for extracting an integer passed from IDL, so be double sure to test before using 
  double bjd_target_cuda(int argc, void *argv[])
  {
    int primary = 0;
    if(argc>=11) primary = (IDL_LONG64) argv[10]; 

    if(primary>0) 
       bjd_target_primary_cuda(argc,argv);
    else
       bjd_target_secondary_cuda(argc,argv);

   return -1; 
  }


} // end extern "C"

