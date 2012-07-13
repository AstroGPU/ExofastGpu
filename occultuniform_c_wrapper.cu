#ifndef CU_OCCULT_UNIFORM_CWRAPPER
#define CU_OCCULT_UNIFORM_CWRAPPER

#include "exofast_cuda_util.cuh"
#include "occultuniform.cu"

// occult_uniform_wrapper_c:
//         C wrapper function
// inputs: 
//         z:        
//         p:        
//         num_data:    integer size of input z array
//         num_model:    integer size of input model parameter arrays
//         ph_mu1: pointer to beginning element of array of doubles 
// outputs:
//         ph_mu1: values overwritten with model flux for uniform limb darkening law
// assumptions:
//         z array has at least num_data elements 
//         ph_mu1 arrays have at least num_data*num_model elements 
//         other arrays have at least num_param elements

	__host__ void occult_uniform_wrapper_c(const double *ph_z, const double *ph_p, const int num_data, const int num_model, double *ph_mu1)
	  {
	    int gpuid = ebf::init_cuda();
	    // put vectors in thrust format from raw points
	    int num = num_data*num_model;
	    thrust::host_vector<double> h_z(ph_z,ph_z+num);
	    thrust::host_vector<double> h_p(ph_p,ph_p+num_model);

	    thrust::counting_iterator<int> index_begin(0);
	    thrust::counting_iterator<int> index_end(num);

	    if(gpuid>=0)
	    	{
		// allocate mem on GPU
		thrust::device_vector<double> d_z(num);
		thrust::device_vector<double> d_p(num_model);
		thrust::device_vector<double> d_mu1(num);

		
		// transfer input params to GPU
//		start_timer_upload();
		d_z = h_z;
		d_p = h_p;
//		stop_timer_upload();

		// distribute the computation to the GPU
	        cudaThreadSynchronize();
//	        start_timer_kernel();
		thrust::transform(
	   	   thrust::make_zip_iterator(thrust::make_tuple(
		      d_z.begin(), thrust::make_permutation_iterator(d_p.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))) )),
	   	   thrust::make_zip_iterator(thrust::make_tuple(
		      d_z.end(),  thrust::make_permutation_iterator(d_p.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))) )),
		   d_mu1.begin(),
		   occult_uniform_functor() );
		 cudaThreadSynchronize();
//                 stop_timer_kernel();

		 // transfer results back to host
//		 start_timer_download();
		 thrust::copy(d_mu1.begin(), d_mu1.end(), ph_mu1);
//		 stop_timer_download();
		 }
	       else
		 {
		 // distribute the computation to the CPU
//		 start_timer_kernel();
		thrust::transform(
	   	   thrust::make_zip_iterator(thrust::make_tuple(
		      h_z.begin(),		      thrust::make_permutation_iterator(h_p.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))) )),
	   	   thrust::make_zip_iterator(thrust::make_tuple(
		      h_z.end(),  thrust::make_permutation_iterator(h_p.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))) )), 
			ph_mu1,
		   occult_uniform_functor() );
//		  stop_timer_kernel();
		  }
	}

#endif

