#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <thrust/host_vector.h>
#include "occultuniform_c_wrapper.cu"


int main(int argc, char **argv)
{
	srand(42u);
	int num_zs = 8196;
	int num_param = 1024;
	double max_planet_size = 0.1;
	int verbose = 0;
	   { // read parameters from command line
	   std::istringstream iss;
	   if(argc>1)
	     {
	     iss.str(std::string (argv[1]));
	     iss >> num_zs;
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
	     iss >> max_planet_size;
	     iss.clear();
	     }
	   if(argc>4)
	     {
	     iss.str(std::string (argv[4]));
	     iss >> verbose;
	     iss.clear();
	     }
	   }
	int num_eval = num_param*num_zs;  
	std::cerr << "# " << argv[0] << " nzs= " << num_zs << " nparam= " << num_param << " max_planet_size= " << max_planet_size << " verbose= " << verbose << "\n";

	// allocate host memory
	thrust::host_vector<double> h_z(num_eval);
	thrust::host_vector<double> h_p(num_param);
	thrust::host_vector<double> h_mu0(num_eval);

	// initialize data on host 
	for(int i=0;i<num_param;++i)
	  {
	    h_p[i] = max_planet_size*static_cast<double>(i+1)/static_cast<double>(num_param);
	  for(int j=0;j<num_zs;++j)
	    {	
            int k = i*num_zs+j;
	    if(verbose>=256)
	       h_z[k] = 2.0*rand()/RAND_MAX;
	    else
	       h_z[k] = 2.0*static_cast<double>(j)/static_cast<double>(num_zs);
	    }
	  }
	// optional check up on input values
	if(verbose%128>4) 
	   {
	   for(int i = 0; i < h_z.size(); i++)
              std::cout << " i= " << i << " z= " << h_z[i] << std::endl;
	   for(int i = 0; i < h_p.size(); i++)
              std::cout << " i= " << i << " p= " << h_p[i] << std::endl;
	   }

	// extract raw pointers to host memory to simulate what you'd get from IDL or another library
	double *ph_z = &h_z[0];
	double *ph_p  = &h_p[0];
	double *ph_mu0 = &h_mu0[0];

	// wrapper function that could be called from IDL
	occult_uniform_wrapper_c(ph_z,ph_p,num_zs,num_param,ph_mu0);
	
	 // print results
	 if(verbose%128>0 ) 
	    {
	    for(int i = 0; i < num_eval; i++)
		   { 
		    std::cout << "i= " << i << " z= " << h_z[i] << " p= " << h_p[i/num_zs] <<  " mu1= " << ph_mu0[i];
		 std::cout << std::endl;
		   }
        }

	return 0;
}



