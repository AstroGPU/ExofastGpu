#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <thrust/host_vector.h>
#include "flux_model_c_wrapper.cpp"

// ********************* BEGIN C DEMO PROGRAMS ****************************** 

// demo program for how to use
// 	flux_model_wrapper_c(...)
// command line arguments:
//      number of observation times
//      number of sets of model parameters
//      verbose (0, 1, 2)
// example:  ./flux_model_test.exe 2048 2048 0
//
int main(int argc, char **argv)
{
        srand(42u);
	// set size parameters from defaults or command line
	int num_obs = 2048;
	int num_param = 2048;
	int verbose = 0;
	double time_min = 0.;
	double time_max = 3.5*365.25;
	{
	std::istringstream iss;
	if(argc>1)
		{
		iss.str(std::string (argv[1]));
		iss >> num_obs;
		iss.clear();
		}
	if(argc>2)
		{
		iss.str(std::string (argv[2]));
		iss >> num_param;
		iss.clear();
		}
	if(argc>3)
		{
		iss.str(std::string (argv[3]));
		iss >> verbose;
		iss.clear();
		}
	}
	int num = num_obs*num_param;

	std::cerr << "# num_obs = " << num_obs << " num_param = " << num_param << " verbose = " << verbose << "\n";

	// allocate host memory
	thrust::host_vector<double> h_time(num_obs);
	thrust::host_vector<double> h_time_peri(num_param);
	thrust::host_vector<double> h_period(num_param);
	thrust::host_vector<double> h_a_over_rstar(num_param);
	thrust::host_vector<double> h_inc(num_param);
	thrust::host_vector<double> h_ecc(num_param);
	thrust::host_vector<double> h_omega(num_param);
	thrust::host_vector<double> h_rp_over_rstar(num_param);
	thrust::host_vector<double> h_u1(num_param);
	thrust::host_vector<double> h_u2(num_param);
	thrust::host_vector<double> h_model_flux(num);

	// initialize data on host 
	for(int i=0;i<num_param;++i)
		{
		h_time_peri[i] = 10.+1.*rand()/static_cast<double>(RAND_MAX);
		h_period[i] = 4.+0.01*rand()/static_cast<double>(RAND_MAX);
		h_a_over_rstar[i] = 10.+0.2*rand()/static_cast<double>(RAND_MAX);
		h_inc[i] = M_PI*(0.49+0.02*rand()/static_cast<double>(RAND_MAX));
		h_ecc[i] = 0.3*rand()/static_cast<double>(RAND_MAX);
		h_omega[i] = M_PI*rand()/static_cast<double>(RAND_MAX);
		h_rp_over_rstar[i] = 0.09+0.02*rand()/static_cast<double>(RAND_MAX);
		h_u1[i] = 0.3+0.01*rand()/static_cast<double>(RAND_MAX);
		h_u2[i] = 0.3+0.02*rand()/static_cast<double>(RAND_MAX);
		}
	for(int j=0;j<num_obs;++j)
		{	
		h_time[j]  = time_min+(time_max-time_min)*static_cast<double>(j)/static_cast<double>(num_obs);
		}

	// optional check up on input values
	if(verbose>1)
	   { // todo
	   }

	// extract raw pointers to host memory to simulate what you'd get from IDL or another library
	double *ph_time = &h_time[0]; 
	double *ph_time_peri = &h_time_peri[0]; 
	double *ph_period = &h_period[0]; 
	double *ph_a_over_rstar = &h_ecc[0]; 
	double *ph_inc = &h_inc[0]; 
	double *ph_ecc = &h_ecc[0]; 
	double *ph_omega = &h_omega[0]; 
	double *ph_rp_over_rstar = &h_rp_over_rstar[0]; 
	double *ph_u1 = &h_u1[0]; 
	double *ph_u2 = &h_u2[0]; 
	double *ph_model_flux = &h_model_flux[0];

	// wrapper function that could be called from IDL
	flux_model_wrapper_c(ph_time,ph_time_peri,ph_period,ph_a_over_rstar,ph_inc,ph_ecc,ph_omega,ph_rp_over_rstar,ph_u1,ph_u2,
	   num_obs,num_param,ph_model_flux);

	// print results to verify that this worked (optional)
	if(verbose>0)
	   {
	   for(int i = 0; i < h_model_flux.size(); i++)
	      {	     
	      std::cout << i << " t= " << h_time[modulo_stride_functor(num_obs)(i)] << " model_id= " << inverse_stride_functor(num_obs)(i) << ' ' << h_model_flux[i] << std::endl;	
	      }
	   }

	// report time spent on calculations and memory transfer
	// calc_model.print_profile_info();

	return 0;
	}

