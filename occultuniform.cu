#ifndef CU_OCCULT_UNIFORM
#define CU_OCCULT_UNIFORM

#include "exofast_cuda_util.cuh"


struct occult_uniform_functor : public thrust::unary_function< thrust::tuple<double, double >,  double >
{
      typedef double return_type;

	 static const double tol = 1.e-6;
	 static const int  z_index = 0;
	 static const int p0_index = 1;

	__host__ __device__ inline return_type operator()( const double& z, const double& p) const
	{
	 if(z>=1.+p) return 1.;  // source is unocculted
	 if( (p>=1.) && (z<= p-1.) ) return 0.;  // source is completely occulted
	 //	if( SQRT(p-0.5) < tol) p = 0.5; // why? I want to avoid creating unnecessary variables
	 if( (z*z>=(1.-p)*(1.-p)) && (z<=1.+p) )  // source is partially occulted and occulting object crosses the limb 
	    {
		double lambdae = 0.;
	   	double tmp1 = (1.-p*p+z*z)/(2.*z);
	   	if(tmp1<-1.) lambdae += M_PI;
	   	else if (tmp1<1.) lambdae += acos(tmp1);
	   	double tmp2 = (p*p+z*z-1.)/(2.*p*z);
	   	if(tmp2<-1.) lambdae += p*p*M_PI;
	   	else if(tmp2<1.) lambdae += p*p*acos(tmp2);
	   	double tmp3 = 4.*z*z-SQR(1.+z*z-p*p);
	   	if(tmp3>0.) lambdae -= 0.5*sqrt(tmp3);
	   	lambdae /= M_PI;
		return 1.-lambdae;
		}
	 // if(z<=1.-p) // occulting object transits the source star and inscribed in side (can't we remove this if statement?)
	    return 1.-p*p; 
	}

	__host__ __device__ inline return_type operator()( const thrust::tuple< const double&, const double& >  val) const
	 {
	 double z = thrust::get<z_index>(val);
	 double p =  thrust::get<p0_index>(val);
	 return operator()(z,p);
	}
};

#endif

