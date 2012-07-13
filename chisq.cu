#ifndef CU_CHISQ
#define CU_CHISQ

#include "exofast_cuda_util.cuh"

struct calc_ressq_functor : public thrust::unary_function< thrust::tuple<const double&, const double&, const double&, double& >, double >
{

	__host__ __device__ inline double operator()( const thrust::tuple< const double&, const double&, const double& >  val) const
	 {
	 double model  = thrust::get<0>(val);
	 double obs    = thrust::get<1>(val);
	 double sigma  = thrust::get<2>(val);
         double dif = (model-obs)/sigma;
	 return dif*dif;
         }

};

#endif

