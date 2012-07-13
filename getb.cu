#ifndef CU_GETB
#define CU_GETB

#include "exofast_cuda_util.cuh"
#include "keplereq.cu"

// Code that gets turned into a GPU kernel by thrust
struct get_xyz_functor : public thrust::unary_function< thrust::tuple<const double&, const double&, const double&, const double&, const double&, const double&, const double& >, thrust::tuple<double, double, double> >
{
	typedef thrust::tuple<double, double, double> result_type;

	__device__ __host__ inline result_type operator()( thrust::tuple<const double&, const double&, const double&, const double&, const double&, const double&, const double& >  val ) const
	{
	double time = thrust::get<0>(val);
	double time_peri = thrust::get<1>(val);
	double period = thrust::get<2>(val);
	double a_over_rstar = thrust::get<3>(val);
	double inc = thrust::get<4>(val);
	double ecc = thrust::get<5>(val);
	double omega = thrust::get<6>(val);

	double mean_anom = 2.*M_PI*fmod(((time-time_peri)/period),1.);
	if(mean_anom<0.) mean_anom += mean_anom;

	keplereq_functor solve_kepler;
	double ecc_anom = solve_kepler(mean_anom,ecc);
#if 1
	double true_anom = 2.0*atan(rsqrt((1.-ecc)/(1.+ecc))*tan(0.5*ecc_anom));
	double r = a_over_rstar*(1.-ecc*ecc)/(1.+ecc*cos(true_anom));
	double cos_true_plus_omega, sin_true_plus_omega;
	sincos(true_anom+omega,&sin_true_plus_omega,&cos_true_plus_omega);
	double x = -r*cos_true_plus_omega;
	double tmp = r*sin_true_plus_omega;
	double cos_inc, sin_inc;
	sincos(inc,&sin_inc,&cos_inc);
	double y = -tmp*cos_inc;
	double z =  tmp*sin_inc;
#else // need to test before using
        double sin_ecc_anom, cos_ecc_anom;
	sincos(ecc_anom,&sin_ecc_anom,&cos_ecc_anom);
//	double r = a_over_rstar*(1.-ecc*cos_ecc_anom);
	double xstd = a_over_rstar*(cos_ecc_anom-ecc);
	double ystd = a_over_rstar*sqrt(1.-ecc*ecc)*sin_ecc_anom;
	double cos_omega, sin_omega;
	sincos(omega,&sin_omega,&cos_omega);
	double x = xstd*cos_omega - ystd*sin_omega;
	double tmp = xstd*sin_omega + ystd*cos_omega;
	double cos_inc, sin_inc;
	sincos(inc,&sin_inc,&cos_inc);
	double y = -tmp*cos_inc;
	double z =  tmp*sin_inc;
#endif
	return thrust::make_tuple(x,y,z);
	};
};


// Code that gets turned into a GPU kernel by thrust
struct get_b_functor : public thrust::unary_function< thrust::tuple<const double&, const double&, const double&, const double&, const double&, const double&, const double& >, double>
{
	typedef double result_type;
	__device__ __host__ inline result_type operator()( thrust::tuple<const double&, const double&, const double&, const double&, const double&, const double&, const double& >  val ) const
	{
	get_xyz_functor get_xyz;
	thrust::tuple<double, double, double> xyz = get_xyz(val);
	const double x = thrust::get<0>(xyz);
	const double y = thrust::get<1>(xyz);
	const double z = thrust::get<2>(xyz);
	double d_proj = sqrt(x*x+y*y);
	return ((z>0.) ? 1. : -1.) * d_proj;
	};
};

// Code that gets turned into a GPU kernel by thrust
struct get_depth_functor : public thrust::unary_function< thrust::tuple<const double&, const double&, const double&, const double&, const double&, const double&, const double& >, double>
{
	typedef double result_type;
	__device__ __host__ inline result_type operator()( thrust::tuple<const double&, const double&, const double&, const double&, const double&, const double&, const double&>  val ) const
	{
	get_xyz_functor get_xyz;
	thrust::tuple<double, double, double> xyz = get_xyz(val);
	return thrust::get<2>(xyz);
	};
};

// Code that gets turned into a GPU kernel by thrust
struct get_b_depth_functor : public thrust::unary_function< thrust::tuple<const double&, const double&, const double&, const double&, const double&, const double&, const double& >, thrust::tuple<double,double> >
{
	typedef thrust::tuple<double,double> result_type;
	__device__ __host__ inline result_type operator()( thrust::tuple<const double&, const double&, const double&, const double&, const double&, const double&, const double& >  val ) const
	{
	get_xyz_functor get_xyz;
	thrust::tuple<double, double, double> xyz = get_xyz(val);
	const double x = thrust::get<0>(xyz);
	const double y = thrust::get<1>(xyz);
	const double z = thrust::get<2>(xyz);
	double d_proj = ((z>0.) ? 1. : -1.) * sqrt(x*x+y*y);
        return thrust::make_tuple(d_proj, z);
	};
};


#endif

