#ifndef H_EXOFAST_CUDA_UTIL
#define H_EXOFAST_CUDA_UTIL

#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
//#include <thrust/for_each.h>
  
// ********************* BEGIN EXOFAST CUDA UTIL FUNCTIONS ****************************** 

#define SQR(x)      ((x)*(x))
#define CUBE(x)     ((x)*(x)*(x))
#define TOFOURTH(x) ((x)*(x)*(x)*(x))
#define TOFIFTH(x)  ((x)*(x)*(x)*(x)*(x))
#define TOSIXTH(x)  ((x)*(x)*(x)*(x)*(x))

// A bunch of junk to deal with querying GPU info
namespace ebf {
/**
        \brief Unrecoverable error exception.

        Throw an instance of this class to indicate an unrecoverable error
        was encountered. Do not throw it directly, but through the use of ERROR() macro.
*/
class runtime_error : public std::runtime_error
{
public:
        runtime_error(const std::string &msg) : std::runtime_error(msg) {}
        virtual ~runtime_error() throw() {};
};

#ifndef THROW_IS_ABORT
        #define ERROR(msg) throw ebf::runtime_error(msg);
#else
        #define ERROR(msg) { fprintf(stderr, "%s\n", std::string(msg).c_str()); abort(); }
#endif


/*!  Unrecoverable CUDA error, thrown by cudaErrCheck macro.
 *    Do not use directly. use cudaErrCheck macro instead.
 */
struct cudaException : public ebf::runtime_error
{
        cudaException(cudaError err) : ebf::runtime_error( cudaGetErrorString(err) ) {}

        static void check(cudaError err, const char *fun, const char *file, const int line) {
                if(err != cudaSuccess)
                        throw cudaException(err);
        }
};
/**
 *      cudaErrCheck macro -- aborts with message if the enclosed call returns != cudaSuccess
 */
#define cudaErrCheck(expr) \
        cudaException::check(expr, __PRETTY_FUNCTION__, __FILE__, __LINE__)

// selects GPU to use and returns gpu ID or -1 if using CPU
int init_cuda() 
{ 
    // Select the proper device
    const char* devstr = getenv("CUDA_DEVICE");
    const int env_dev = (devstr != NULL) ? atoi(devstr) : 0;
    int dev = env_dev;
    int devcnt; ebf::cudaErrCheck( cudaGetDeviceCount(&devcnt) );
    if( dev >= 0 && dev < devcnt )
       { 
       ebf::cudaErrCheck( cudaSetDevice(dev) ); 
       cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
       }
    else
       {
        dev = -1;
       	std::cerr << "# Cannot select the CUDA device. Using CPU!" << std::endl;
	}
    return dev;
}


struct nonprofiled_wrapper_c_base
{
   void start_timer_kernel()    { }
   void start_timer_upload()    { }
   void start_timer_download() { }
   void stop_timer_kernel()  { }
   void stop_timer_upload()  { }
   void stop_timer_download() { }
   void print_profile_info()  { }
};

struct profiled_wrapper_c_base
{
   // place to store results of timing code
   uint kernelTime, memoryUploadTime, memoryDownloadTime;
   
   profiled_wrapper_c_base() 
   {
   cutCreateTimer(&memoryUploadTime);
   cutCreateTimer(&memoryDownloadTime);
   cutCreateTimer(&kernelTime);	
   cutResetTimer(memoryUploadTime);    
   cutResetTimer(memoryDownloadTime);	
   cutResetTimer(kernelTime);
   }

   void start_timer_kernel() 
    {  
		cudaThreadSynchronize();
		cutStartTimer(kernelTime);
	}
		
   void start_timer_upload() 
    {  
		cudaThreadSynchronize();
		cutStartTimer(memoryUploadTime);
	}
		
	void start_timer_download() 
    {  
		cudaThreadSynchronize();
		cutStartTimer(memoryDownloadTime);
	}
		
   void stop_timer_kernel() 
    {  
		cudaThreadSynchronize();
		cutStopTimer(kernelTime);
	}
		
   void stop_timer_upload() 
    {  
		cudaThreadSynchronize();
		cutStopTimer(memoryUploadTime);
	}
		
	void stop_timer_download() 
    {  
		cudaThreadSynchronize();
		cutStopTimer(memoryDownloadTime);
	}
	
   
    void print_profile_info()
	{
	// report time spent on calculations and memory transfer
	std::cerr << "# Time for kernel: " << cutGetTimerValue(kernelTime) << " ms, Time for memory: " << cutGetTimerValue(memoryUploadTime) << " ms (upload) + " << cutGetTimerValue(memoryDownloadTime) << " ms (download)  Total time: " << cutGetTimerValue(kernelTime)+cutGetTimerValue(memoryUploadTime)+cutGetTimerValue(memoryDownloadTime) << " ms \n"; 
    }
	
};

}


// Should these be moved into a namespace?
        // For reducing memory transfer when multiple observations associated with one set of parameters
        struct inverse_stride_functor : public thrust::unary_function<int,int>
        {
        const int invstride;

        inverse_stride_functor(const int _invstride) : invstride(_invstride) {}
        __host__ __device__  int operator()(const int x) const
          { return x / invstride; }
        };


        // For reducing memory transfer when multiple observations associated with one set of parameters
        struct modulo_stride_functor : public thrust::unary_function<int,int>
        {
        const int stride;

        modulo_stride_functor(const int _stride) : stride(_stride) {}
        __host__ __device__  int operator()(const int x) const
          { return x % stride; }
        };




#endif

