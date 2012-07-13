#ifndef CU_BJD
#define CU_BJD

#include "exofast_cuda_util.cuh"
#include "keplereq.cu"

// Code that gets turned into a GPU kernel by thrust
template<bool Primary>
struct target2bjd_functor : public thrust::unary_function< thrust::tuple<const double&, const double&, const double&, const double&, const double&, const double&, const double& >, double >
{
	typedef double result_type;
        static const double speed_of_light = 173.144483; //  AU/day

	__device__ __host__ inline result_type operator()( thrust::tuple<const double&, const double&, const double&, const double&, const double&, const double&, const double& >  val ) const
	{
	double bjd_target = thrust::get<0>(val);
	double time_peri = thrust::get<1>(val);
	double period = thrust::get<2>(val);
	double a_in_au = thrust::get<3>(val);  // I'm letting this a be the semi-major axis for the body of interest, not a of the system, so it assumes you've already multiplied by Eastman's "factor"
	double inc = thrust::get<4>(val);
	double ecc = thrust::get<5>(val);
	double omega = thrust::get<6>(val);

	double mean_anom = 2.*M_PI*fmod(((bjd_target-time_peri)/period),1.);
	if(mean_anom<0.) mean_anom += mean_anom;

	keplereq_functor solve_kepler;
	double ecc_anom = solve_kepler(mean_anom,ecc);
#if 1
	double true_anom = 2.0*atan(rsqrt((1.-ecc)/(1.+ecc))*tan(0.5*ecc_anom));
	double r = a_in_au*(1.-ecc*ecc)/(1.+ecc*cos(true_anom));
	double cos_true, sin_true;
	sincos(true_anom,&sin_true,&cos_true);
	double x =  r*cos_true;
	double y =  r*sin_true;
#else   // need to test before using
        double sin_ecc_anom, cos_ecc_anom;
        sincos(ecc_anom,&sin_ecc_anom,&cos_ecc_anom);
        double x = a_in_au*(cos_ecc_anom-ecc);
        double y = a_in_au*sqrt(1.-ecc*ecc)*sin_ecc_anom;
#endif
	if(Primary) omega += M_PI;
	double cos_omega, sin_omega;
	sincos(omega,&sin_omega,&cos_omega);
	y = x*sin_omega + y*cos_omega;
	double z = y*sin(inc);
	
	return bjd_target - z/speed_of_light;
	};
};


// Code that gets turned into a GPU kernel by thrust
template<bool Primary>
struct bjd_target_functor : public thrust::unary_function< thrust::tuple<double&, const double&, const double&, const double&, const double&, const double&, const double& >, double>
{
	typedef double result_type;
	static const double tolsq_default = 1e-14;  // 10 ms seems plenty good
	static const int max_it = 6;
        double tolsq;

         bjd_target_functor(const double _tolsq = tolsq_default ) : tolsq(_tolsq) {};

	__device__ __host__ inline result_type operator()( thrust::tuple<double&, const double&, const double&, const double&, const double&, const double&, const double& >  val ) const
	{
	double bjd_tdb = thrust::get<0>(val);
	target2bjd_functor<Primary> target2bjd;
	for(int i=0;i<max_it;++i)
	   {
	   double target_new = target2bjd(val);	
	   double dif = bjd_tdb-target_new;
	   thrust::get<0>(val) += dif;
           if(dif*dif<=tolsq) break;
   	   }
	return thrust::get<0>(val);
	};
};

#endif

