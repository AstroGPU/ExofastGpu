#ifndef CU_OCCULTQUAD
#define CU_OCCULTQUAD

#include "exofast_cuda_util.cuh"

#define ACCEL_TRIVIAL_RETURN 1
#define GROUP_FUNC_CALLS_MINI 1
#define GROUP_FUNC_CALLS 0 // doesn't help performance

struct occultquad_functor : public thrust::unary_function< thrust::tuple<double, double, double, double >,  thrust::tuple<double,double> >
{
      typedef thrust::tuple<double,double> return_type;

	 static const double tol = 1.e-14;
	 static const int  z_index = 0;
	 static const int u1_index = 1;
	 static const int u2_index = 2;
	 static const int p0_index = 3;
	 static const int muo1_index = 4;
	 static const int mu0_index = 5;

	 struct ellke_functor
	 {
 	   __host__ __device__ inline thrust::tuple<double,double> operator()(const double k) const
      	   {
            double m1 = 1.0-k*k;
      	    double logm1 = log(m1);
      	    double ek, kk;
	    {
	    const double a1=0.44325141463;
	    const double a2=0.06260601220;
	    const double a3=0.04757383546;
	    const double a4=0.01736506451;
	    const double b1=0.24998368310;
	    const double b2=0.09200180037;
	    const double b3=0.04069697526;
	    const double b4=0.00526449639;
	    const double ee1=1.+m1*(a1+m1*(a2+m1*(a3+m1*a4)));
	    const double ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*(-logm1);
	    ek = ee1+ee2;
	    }	
	    {
	    const double a0=1.38629436112;
	    const double a1=0.09666344259;
	    const double a2=0.03590092383;
	    const double a4=0.01451196212;
	    const double a3=0.03742563713;
	    const double b0=0.5;
	    const double b1=0.12498593597;
	    const double b2=0.06880248576;
	    const double b3=0.03328355346;
	    const double b4=0.00441787012;
	    const double ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)));
	    const double ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*logm1;
	    kk = ek1-ek2;
	    }
      	 return thrust::make_tuple(ek,kk);
      	 }
      };

      __host__ __device__ inline double ellpic_bulirsch(const double n, const double k) const
      {
      double kc = sqrt(1.-k*k);
      double p = n+1.0;
#if CPU_DEBUG
      assert(p>=0.);
#endif
      double m0 = 1.0;
      double c =1.0;
      double d = rsqrt(p);
      p = 1.0/d;
      double e = kc;
      do {
       double f = c;
       c = d/p+c;
       double g = e/p;
       d = 2.*(f*g+d);
       p = g + p;
       g = m0;
       m0 = kc + m0;
       if(SQR(1.0-kc/g)>1.e-16)
          { kc = 2.*sqrt(e);   e = kc*m0; }
       else
          { return 0.5*M_PI*(c*m0+d)/(m0*(m0+p)); }
       } while(true);
      }


	__host__ __device__ inline return_type operator()( const thrust::tuple< const double&, const double&, const double&, const double& >  val) const
	 {
	 ellke_functor ellke;
	 double z  = thrust::get<z_index>(val);
	 const double p0 = thrust::get<p0_index>(val);
	 const double p = fabs(p0); // "to mesh with fitting routines"
#if ACCEL_TRIVIAL_RETURN 
	 if((p<=0.) || (z>=1.+p) || (z<0.) ) // case 0, 1
	   {
	   return return_type(1.,1.);
	   }
#endif 
	 const double u1 = thrust::get<u1_index>(val);
	 const double u2 = thrust::get<u2_index>(val);
	 double mu0, muo1;
	 const double omega = 1.0-(u1-0.5*u2)/3.;
	 double lambdad = 0.;
	 double lambdae = 0.;
	 double etad = 0.;

	 z = (fabs(p-z)<tol) ? p : z;
	 z = (fabs((p-1.)-z)<tol) ? p-1. : z;
	 z = (fabs((1.-p)-z)<tol) ? 1.-p : z;
	 z = (z<tol) ? 0. : z;
	 const double x1 = (p-z)*(p-z);
	 const double x2 = (p+z)*(p+z);

#if !ACCEL_TRIVIAL_RETURN 
         // case 0 and 1 moved up to reduce memory loads
	 if(z<0.)  // if planet behind star
	   {
	   return return_helper(1.,1.);
	   }
	 if(p<=0.)  // case 0
	   {
	   return return_helper(1.,1.);
	   }
	 else if(z>=1.+p) // case 1 // source is unocculted (why so much code?)
	   {
	   muo1 = 1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega;
	   mu0 = 1.-lambdae;
	   return return_helper(muo1,mu0);
	   }
	 else 
#endif
	 if( (p>=1.) && (z<=p-1.) ) // case 11 (source completely occulted)
	   {
	   etad = 0.5;
	   lambdae = 1.;
	   muo1 = 1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega;
	   mu0 = 1.-lambdae;
	   }
	else // partially occulted
	{
	if( (z>=fabs(1.-p)) && (z<1.+p) ) // case 2,7,8 (during ingress/egress)
	   {
	   double tmp1 = (1.-p*p+z*z)/(2.*z);
	   if(tmp1>1.) tmp1 = 1.;	   if(tmp1<-1.) tmp1 = -1.;
	   double kap1 = acos(tmp1);
	   double tmp2 = (p*p+z*z-1.)/(2.*p*z);
	   if(tmp2>1.) tmp2 = 1.;	   if(tmp2<-1.) tmp2 = -1.;
	   double kap0 = acos(tmp2);
	   double tmp3 = 4.*z*z-SQR(1.+z*z-p*p);
	   if(tmp3<0.) tmp3 = 0.;
	   lambdae = (p*p*kap0+kap1-0.5*sqrt(tmp3))/M_PI;
	   etad = (kap1+p*p*(p*p+2.*z*z)*kap0-0.25*(1.+5.*p*p+z*z)*sqrt((1.-x1)*(x2-1.)))/(2.*M_PI);
	   // don't return here!
	   }

#if GROUP_FUNC_CALLS 
         // I thought it would be good to parallelize computation of Ek, Kk and elliptic integral
	 // But on test case, it's slower, so I abandonded the idea
	 double q, n;
	 bool compute_EkKk = false, compute_bs = false;
	 if(z==p) // case 5, 6, 7 (edge of planet at origin of star)
	   {  q = (p<=0.5) ? 2.*p : 0.5/p;  compute_EkKk = true; }
	 else if( ((z>0.5+fabs(p-0.5)) && (z<1.+p)) || ((p>0.5) && (z>fabs(1.-p)) && (z<p) ) ) // case 2, 8 (during ingress/egress) (needs etad from uniform disk code)
	   {  
	   q = sqrt((1.-x1)/(4.*p*z));
	   n = 1./x1-1.;          
   	   compute_EkKk = true; compute_bs = true; 
	   }
	 else if((p<1.)&&(z!=1.-p)&&(z!=0.)) // case 3, 9 (planet completely inside star)
	   { 
	   q = rsqrt((1.-x1)/(x2-x1)); 
	   n = x2/x1-1.;
   	   compute_EkKk = true; compute_bs = true; 
	   }
	 thrust::tuple<double,double> EkKk = (compute_EkKk) ? ellke(q) : thrust::make_tuple(0.,0.);
	 double ellpic_bulrisch_n_q = (compute_bs) ? ellpic_bulirsch(n,q) : 0.;
#endif	 
 
	 if(z==p) // case 5, 6, 7 (edge of planet at origin of star)
	   {
#if GROUP_FUNC_CALLS_MINI
	   double q = (p<=0.5) ? 2.*p : 0.5/p;
	   thrust::tuple<double,double> EkKk = (p!=0.5) ? ellke(q) : thrust::make_tuple(0.,0.);
#endif
	   if(p<0.5) // case 5
	     {
#if !GROUP_FUNC_CALLS && !GROUP_FUNC_CALLS_MINI
	     double q = 2.*p;
	     thrust::tuple<double,double> EkKk = ellke(q);
#endif
	     lambdad = 1./3.+2.*(4.*(2.*p*p-1.)*EkKk.get<0>()+(1.-4.*p*p)*EkKk.get<1>())/(9.*M_PI);
	     etad = 0.5*p*p*(p*p+2.*z*z);
	     lambdae = p*p;
	     }
	   else if( p>0.5) // case 7 (need etad from uniform disk code)
	     {
#if !GROUP_FUNC_CALLS && !GROUP_FUNC_CALLS_MINI
	     double q = 0.5/p;
	     thrust::tuple<double,double> EkKk = ellke(q);
#endif
	     lambdad = 1./3.+(16.*p*(2.*p*p-1.)*EkKk.get<0>()-
	       (32.*p*p*p*p-20.*p*p+3.)/(p)*EkKk.get<1>())/(9.*M_PI);
	     }
	   else // case 6
	     {
	     lambdad = 1./3.-4./(9.*M_PI);
	     etad = 3./32.;
	     }
	   muo1 = 1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega;
	   mu0 = 1.-lambdae;
	   }
	 else if( ((z>0.5+fabs(p-0.5)) && (z<1.+p)) || ((p>0.5) && (z>fabs(1.-p)) && (z<p) ) ) // case 2, 8 (during ingress/egress) (needs etad from uniform disk code)
	   {
#if !GROUP_FUNC_CALLS
	   const double q = sqrt((1.-x1)/(4.*p*z));
	   thrust::tuple<double,double> EkKk = ellke(q);
	   const double n = 1./x1-1.;          
//	   const double n = 1./(p-z);  // from python version?!?
	   const double ellpic_bulrisch_n_q = ellpic_bulirsch(n,q);
#endif
	   const double x3 = p*p-z*z;
	   lambdad = 1./(9.*M_PI)*rsqrt(p*z)*
	     ( ((1.-x2)*(2.*x2+x1-3.)-3.*x3*(x2-2.))*EkKk.get<1>()
	     +4.*p*z*(z*z+7.*p*p-4.)*EkKk.get<0>()
	     -3.*x3/x1*ellpic_bulrisch_n_q );
	   muo1 = 1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega;
	   mu0 = 1.- lambdae;
	   }
	 else if(p<1.) // case 3, 4, 9, 10 (planet completely inside star)
	   {
#if CPU_DEBUG
	   assert(z<1.-p); 
#endif
	     etad = 0.5*p*p*(p*p+2.*z*z);
	     lambdae = p*p;
	     if(z==1.-p) // case 4
	       {
	       lambdad = ( 6.*acos(1.-2.*p)-4.*sqrt(p*(1.-p))*(3.+2.*p-8.*p*p) )/(9.*M_PI);
	       if(p>0.5)
	        lambdad -= 2./3.;

	       muo1 = 1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega;
	       mu0 = 1.-lambdae;
           }
	     else if(z==0.) // case 10
	       {
	       lambdad = -2./3.*(1.-p*p)*sqrt(1.-p*p);
	       muo1 = 1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega;
	       mu0 = 1.-lambdae;
	       }
	     else  // case 3, 9
	       {
#if !GROUP_FUNC_CALLS
	       double q = rsqrt((1.-x1)/(x2-x1));
	       thrust::tuple<double,double> EkKk = ellke(q);	   
	       double n = x2/x1-1.;
	       double ellpic_bulrisch_n_q = ellpic_bulirsch(n,q);
#endif
	       double x3 = p*p-z*z;
	       lambdad = 2./(9.*M_PI)*rsqrt(1.-x1)*
                       ( (1.-5.*z*z+p*p+x3*x3)*EkKk.get<1>()
		         +(1.-x1)*(z*z+7.*p*p-4.)*EkKk.get<0>()
			 -3.*x3/x1*ellpic_bulirsch(n,q) );
	       muo1 = 1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega;
	       mu0 = 1.-lambdae;
	       }
	     }  // end if (p<1.) case 3,4,9,10
	   } // end else case partially occulted

	double tmp = ((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega;
	muo1 = 1.+ (1.-2.*(p0>0.)) * tmp;
	mu0 = 1.+ (1.-2.*(p0>0.)) * lambdae;
	return thrust::make_tuple(muo1,mu0);
	}
};

struct occult_uniform_slow_functor  : public thrust::unary_function< thrust::tuple<double, double, double, double >,  double >

{
      typedef double return_type;
	__host__ __device__ inline return_type operator()( thrust::tuple< const double&, const double&, const double&, const double& >  val) const
	 {
	  occultquad_functor occultquad;
	  thrust::tuple<double,double> occultquad_result = occultquad(val);
	  return thrust::get<1>(occultquad_result);
	  }
};

struct occultquad_only_functor  : public thrust::unary_function< thrust::tuple<double, double, double, double >,  double >

{
      typedef double return_type;
	__host__ __device__ inline return_type operator()( thrust::tuple< const double&, const double&, const double&, const double& >  val) const
	 {
	  occultquad_functor occultquad;
	  thrust::tuple<double,double> occultquad_result = occultquad(val);
	  return thrust::get<0>(occultquad_result);
	  }
};

#endif

