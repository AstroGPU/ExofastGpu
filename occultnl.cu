#ifndef CU_OCCULTNL
#define CU_OCCULTNL

#include "exofast_cuda_util.cuh"
#include "occultuniform.cu"

#define CPU_DEBUG 0


struct occultnl_fast_functor : public thrust::unary_function< thrust::tuple<double, double, double, double, double, double >,  double >
{
     typedef double return_type;

	 static const double tol = 1.e-6;
	 static const int  z_index = 0;
	 static const int c1_index = 1;
	 static const int c2_index = 2;
	 static const int c3_index = 3;
	 static const int c4_index = 4;
	 static const int p0_index = 5;

	__host__ __device__ inline return_type operator()( const thrust::tuple< const double&, const double&, const double&, const double&, const double&, const double& >  val) const
	 {
	 const double z  = thrust::get<z_index>(val);
	 const double c1 = thrust::get<c1_index>(val);
	 const double c2 = thrust::get<c2_index>(val);
	 const double c3 = thrust::get<c3_index>(val);
	 const double c4 = thrust::get<c4_index>(val);
	 const double p0 = thrust::get<p0_index>(val);
	 occult_uniform_functor occult_uniform;
	 double mulimb0 = occult_uniform(z,p0);
	 if(mulimb0 == 1.) return 1.;       // if not during transit
	 else if (mulimb0 == 0.) return 0.; // if source fully occulted
	 const double omega = 4.*((1.-c1-c2-c3-c4)/4.+c1/5.+c2/6.+c3/7.+c4/8.);
	 double mulimb = mulimb0;
	 double mulimbp;
	 int nr = 2;
	 do {
		mulimbp = mulimb;
		nr *= 2;
		double dt = 0.5*M_PI/static_cast<double>(nr);
		double r_nrm1 = sin(dt*(nr-1.));
		double sig = sqrt(cos((nr-0.5)*dt));
		double mulimbterms = sig*sig*sig*(c1+sig*(c2+sig*(c3+sig*c4)))*mulimb0/(1.-r_nrm1);
	    double r_im1 = 0.;
		double r_i = sin(dt);
		double sig1 = sqrt(cos(dt*0.5));
		for(int i=1;i<=nr-1;++i)
		   {
		   double mu = occult_uniform(z/r_i,p0/r_i);
		   // double r_im1 = sin(dt*(i-1));
		   // double r_i = sin(dt*i);
		   double r_ip1 = sin(dt*(i+1));
		   // double sig1 = sqrt(cos(dt*(i-0.5)));
		   double sig2 = sqrt(cos(dt*(i+0.5)));
		   mulimbterms += r_i*r_i*mu* (   c1*(CUBE(sig1)/(r_i-r_im1)-CUBE(sig2)/(r_ip1-r_i)) + 
		   c2*(TOFOURTH(sig1)/(r_i-r_im1)-TOFOURTH(sig2)/(r_ip1-r_i)) +
		   c3*(TOFIFTH(sig1)/(r_i-r_im1)-TOFIFTH(sig2)/(r_ip1-r_i)) +
		   c4*(TOSIXTH(sig1)/(r_i-r_im1)-TOSIXTH(sig2)/(r_ip1-r_i)) );
		   sig1 = sig2;
		   r_im1 = r_i;
		   r_i = r_ip1;
		   }

		  mulimb = ((1.-c1-c2-c3-c4)*mulimb0+dt*mulimbterms)/omega;
	    } while ( SQR(mulimb-mulimbp) > tol*SQR(mulimb0-1.)*SQR(mulimb+mulimbp) );
	    return mulimb;	    
	}
};

// if above doesn't give right answers, then try replacing occultnl_functor with occultnl_literal_functor which is a more literal translation of the Agol & Mandel FORTRAN code
struct occultnl_literal_functor : public thrust::unary_function< thrust::tuple<double, double, double, double, double, double >,  double >
{
     typedef double return_type;

	 static const double tol = 1.e-6;
	 static const int  z_index = 0;
	 static const int c1_index = 1;
	 static const int c2_index = 2;
	 static const int c3_index = 3;
	 static const int c4_index = 4;
	 static const int p0_index = 5;

	__host__ __device__ inline return_type operator()( const thrust::tuple< const double&, const double&, const double&, const double&, const double&, const double& >  val) const
	 {
	 const double z  = thrust::get<z_index>(val);
	 const double c1 = thrust::get<c1_index>(val);
	 const double c2 = thrust::get<c2_index>(val);
	 const double c3 = thrust::get<c3_index>(val);
	 const double c4 = thrust::get<c4_index>(val);
	 const double p0 = thrust::get<p0_index>(val);
	 occult_uniform_functor occultuniform;
	 double mulimb0 = occultuniform(z,p0);
	 if(mulimb0 == 1.) return 1.;       // if not during transit
	 else if (mulimb0 == 0.) return 0.; // if source fully occulted
	 const double omega = 4.*((1.-c1-c2-c3-c4)/4.+c1/5.+c2/6.+c3/7.+c4/8.);
	 double mulimb = mulimb0;
	 double mulimbp;
	 int nr = 2;
	 do {
		mulimbp = mulimb;
		nr *= 2;
		double dt = 0.5*M_PI/static_cast<double>(nr);
		// double t[nr+1] = dt * (0...nr+1);
		// double th[nr+1] = t + 0.5*dt;
		// double r[nr+1] = sin(t);
		double r_nrm1 = sin(dt*(nr-1.));
		double sig = sqrt(cos((nr-0.5)*dt));
		double sig3        = sig*sig*sig;
		double mulimbhalf = sig3*mulimb0/(1.-r_nrm1);
		double mulimb1     = sig*mulimbhalf;
		double mulimb3half = sig*sig*mulimbhalf;
		double mulimb2     = sig3*mulimbhalf;
		for(int i=1;i<=nr-1;++i)
		   {
		   double r_im1 = sin(dt*(i-1));
		   double r_i = sin(dt*i);
		   double r_ip1 = sin(dt*(i+1));
		   double mu = occultuniform(z/r_i,p0/r_i);
		   double sig1 = sqrt(cos(dt*(i-0.5)));
		   double sig2 = sqrt(cos(dt*(i+0.5)));
		   mulimbhalf  += r_i*r_i*mu*(CUBE(sig1)/(r_i-r_im1)-CUBE(sig2)/(r_ip1-r_i));
		   mulimb1     += r_i*r_i*mu*(TOFOURTH(sig1)/(r_i-r_im1)-TOFOURTH(sig2)/(r_ip1-r_i));
		   mulimb3half += r_i*r_i*mu*(TOFIFTH(sig1)/(r_i-r_im1)-TOFIFTH(sig2)/(r_ip1-r_i));
		   mulimb2     += r_i*r_i*mu*(TOSIXTH(sig1)/(r_i-r_im1)-TOSIXTH(sig2)/(r_ip1-r_i));
		   }
	     mulimb = ((1.-c1-c2-c3-c4)*mulimb0+dt*(c1*mulimbhalf+c2*mulimb1+c3*mulimb3half+c4*mulimb2))/omega;
	    } while ( SQR(mulimb-mulimbp) > tol*SQR(mulimb0-1.)*SQR(mulimb+mulimbp) );

	 return mulimb;
	}
};

// pick one
struct occultnl_functor : public occultnl_literal_functor {};
//struct occultnl_functor : public occultnl_fast_functor {};

#endif

