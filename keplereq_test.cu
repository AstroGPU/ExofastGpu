#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <thrust/host_vector.h>

#include "keplereq_c_wrapper.cu"

// demo program for how to use
// 	keplereq_wrapper_c(ph_ma,ph_ecc,num_eval,ph_eccanom);
// command line arguments:
//      number of eccentricities
//      number of mean anomalies
//      verbose (0, 1, 2)
// example:  ./keplereq.exe 4096 8192 0
// performance note:  
//      For just solving Kepler's equation, CPU<->GPU memory transfer overhead 
// 	is several times more expensive than the actual calculations.
//	So you might as well calculate it many times.
//      Eventually move more calculations onto GPU to amortize memory transfer
//      On GF100, 32M evals take a total of 256ms, of which 213ms is memory
//
int main(int argc, char **argv)
{
	// set size parameters from defaults or command line
	int num_ecc = 4096;
	int num_ma = 4096;
	int verbose = 0;
	{
	std::istringstream iss;
	if(argc>1)
		{
		iss.str(std::string (argv[1]));
		iss >> num_ecc;
		iss.clear();
		}
	if(argc>2)
		{
		iss.str(std::string (argv[2]));
		iss >> num_ma;
		iss.clear();
		}
	if(argc>3)
		{
		iss.str(std::string (argv[3]));
		iss >> verbose;
		iss.clear();
		}
	}
	int num_eval = num_ecc*num_ma;

	std::cerr << "# num_ecc = " << num_ecc << " num_meannom = " << num_ma << " verbose = " << verbose << "\n";

	// allocate host memory
	thrust::host_vector<double> h_ecc(num_ecc);
	thrust::host_vector<double> h_ma(num_ma);

	thrust::host_vector<double> h_eccanom(num_eval);

	// initialize data on host 
	for(int i=0;i<num_ecc;++i)
		h_ecc[i] = 0.3+0.3*static_cast<double>(i)/static_cast<double>(num_ecc);

	for(int j=0;j<num_ma;++j)
		h_ma[j]  = 2.*M_PI*static_cast<double>(j)/static_cast<double>(num_ma);

	// optional check up on input values
	if(verbose>1)
	   {
		for(int i = 0; i < h_ecc.size(); i++)
        	   std::cout << "p[" << i << "] = " << h_ecc[i] << std::endl;

		for(int i = 0; i < h_ma.size(); i++)
		   std::cout << "z[" << i << "] = " << h_ma[i] << std::endl;
	   }

	// extract raw pointers to host memory to simulate what you'd get from IDL or another library
	double *ph_ecc = &h_ecc[0]; 
	double *ph_ma = &h_ma[0]; 
	double *ph_eccanom = &h_eccanom[0];

	// wrapper function that could be called from IDL
        keplereq_wrapper_c(ph_ma,ph_ecc,num_ma,num_ecc,ph_eccanom);

	// print results to verify that this worked (optional)
	if(verbose>0)
	   {
	   for(int i = 0; i < h_eccanom.size(); i++)
	      std::cout << i << ' ' << h_ecc[i] << ' ' << h_ma[i] << ' ' << ph_eccanom[i] << std::endl;	
	   }

	// report time spent on calculations and memory transfer
        //  kepler_solver.print_profile_info();
	
}

