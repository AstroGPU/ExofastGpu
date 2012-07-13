#ifndef CU_OCCULTNL_CWRAPPER
#define CU_OCCULTNL_CWRAPPER

#include "exofast_cuda_util.cuh"
#include "occultnl.cu"

// occultnl_wrapper_c:
//         C wrapper function
// inputs: 
//         z:        
//         c1:        
//         c2:        
//         c3:        
//         c4:        
//         p:        
//         num_data:    integer size of input z array
//         num_model:    integer size of input model parameter arrays
//         ph_muo1: pointer to beginning element of array of doubles 
// outputs:
//         ph_muo1: values overwritten with model flux for quadratic lim darkening law
//         z array has at least num_data elements 
//         ph_muo1 arrays have at least num_data*num_model elements 
//         other arrays have at least num_param elements

	__host__ void occultnl_wrapper_c(const double *ph_z, const double *ph_c1, const double *ph_c2, const double *ph_c3, const double *ph_c4, const double *ph_p, const int num_data, const int num_model, double *ph_muo1)
	  {
	    int gpuid = ebf::init_cuda();
	    // put vectors in thrust format from raw points
	    int num = num_data*num_model;
	    thrust::host_vector<double> h_z(ph_z,ph_z+num);
	    thrust::host_vector<double> h_c1(ph_c1,ph_c1+num_model);
	    thrust::host_vector<double> h_c2(ph_c2,ph_c2+num_model);
	    thrust::host_vector<double> h_c3(ph_c3,ph_c3+num_model);
	    thrust::host_vector<double> h_c4(ph_c4,ph_c4+num_model);
	    thrust::host_vector<double> h_p(ph_p,ph_p+num_model);

	    thrust::counting_iterator<int> index_begin(0);
	    thrust::counting_iterator<int> index_end(num);

	    if(gpuid>=0)
	    	{

		// allocate mem on GPU
		thrust::device_vector<double> d_z(num);
		thrust::device_vector<double> d_c1(num_model);
		thrust::device_vector<double> d_c2(num_model);
		thrust::device_vector<double> d_c3(num_model);
		thrust::device_vector<double> d_c4(num_model);
		thrust::device_vector<double> d_p(num_model);
		thrust::device_vector<double> d_muo1(num);

		
		// transfer input params to GPU
//		start_timer_upload();
		d_z = h_z;
		d_c1 = h_c1;
		d_c2 = h_c2;
		d_c3 = h_c3;
		d_c4 = h_c4;
		d_p = h_p;
//		stop_timer_upload();

		// distribute the computation to the GPU
	        cudaThreadSynchronize();
//	        start_timer_kernel();
		thrust::transform(
	   	   thrust::make_zip_iterator(thrust::make_tuple(
		      d_z.begin(),
		      thrust::make_permutation_iterator(d_c1.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(d_c2.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))),
		     thrust::make_permutation_iterator(d_c3.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))),
		     thrust::make_permutation_iterator(d_c4.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(d_p.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))) )),
	   	   thrust::make_zip_iterator(thrust::make_tuple(
		      d_z.end(),
		      thrust::make_permutation_iterator(d_c1.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(d_c2.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(d_c3.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(d_c4.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))),
			  thrust::make_permutation_iterator(d_p.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))) )),
		  d_muo1.begin(), 
		   occultnl_functor() );
		 cudaThreadSynchronize();
//                 stop_timer_kernel();

		 // transfer results back to host
//		 start_timer_download();
		 
		 thrust::copy(d_muo1.begin(), d_muo1.end(), ph_muo1);
//		 stop_timer_download();
		 }
	       else
		 {
		 // distribute the computation to the CPU
//		 start_timer_kernel();
		thrust::transform(
	   	   thrust::make_zip_iterator(thrust::make_tuple(
		      h_z.begin(),
		      thrust::make_permutation_iterator(h_c1.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(h_c2.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(h_c3.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(h_c4.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(h_p.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))) )),
			  
	   	   thrust::make_zip_iterator(thrust::make_tuple(
		      h_z.end(), 
              thrust::make_permutation_iterator(h_c1.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(h_c2.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(h_c3.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(h_c4.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))),
		      thrust::make_permutation_iterator(h_p.end(),thrust::make_transform_iterator(index_end,inverse_stride_functor(num_data))) )), 
			ph_muo1,
		   occultnl_functor() );
//		  stop_timer_kernel();
		  }
	}

#endif

