#ifndef CU_KEPLEREQN
#define CU_KEPLEREQN

#include "exofast_cuda_util.cuh"

#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
  
// Code that gets turned into a GPU kernel by thrust
struct keplereq_functor  : public thrust::binary_function<double,double,double>
{
    typedef double return_type;
	static const double del_sq = 1.0e-12;
    static const double k = 0.85;
	static const int num_max_it = 20;
	static const double third = 1.0/3.0;

	__device__ __host__ keplereq_functor() { };

	__device__ __host__ inline return_type operator()(const double& M, const double& e) const
	{
#if DEBUG_CPU
	assert(M>=0.);
	assert(M<=2.*M_PI);
	assert(e>=0.);
	assert(e<=1.);
#endif
 	double x = (M<M_PI) ? M + k*e : M - k*e;
    double F = 1.;
    for(int i=0;i<num_max_it;++i)
	   {
	   double es, ec;
 	   sincos(x,&es,&ec);
       es *= e;
       F = (x-es)-M;
       if(fabs(F)<del_sq) break;
       ec *= e;
 	   const double Fp = 1.-ec;
       const double Fpp = es;
       const double Fppp = ec;
       double Dx = -F/Fp;
       Dx = -F/(Fp+0.5*Dx*Fpp);
       Dx = -F/(Fp+0.5*Dx*(Fpp+third*Dx*Fppp));
       x += Dx;
       }
	return x;
	};

};

#endif

