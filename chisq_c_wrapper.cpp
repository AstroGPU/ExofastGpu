#ifndef CU_CHISQ_C_WRAPPER
#define CU_CHISQ_C_WRAPPER

#include "exofast_cuda_util.cuh"
#include "chisq.cu"

// calc_chisq_wrapper_c:
//         C wrapper function
// inputs: 
//         model:        
//         obs:        
//         sigma:        
//         num_data:    integer size of input z array
//         num_model:   integer size of input model parameter arrays
//         ph_chisq:    pointer to beginning element of array of doubles 
// outputs:
//         ph_chisq: values overwritten with chisq
// assumptions:
//         obs, sigma arrays have at least num_data elements 
//         ph_chisq array has at least num_model elements 
//         other arrays have at least num_data*num_param elements

	__host__ void calc_chisq_wrapper_c(const double *ph_model, const double *ph_obs, const double *ph_sigma, const int num_data, const int num_model, double *ph_chisq )
	  {
	    int gpuid = ebf::init_cuda();
	    // put vectors in thrust format from raw points
	    int num = num_data*num_model;
	    thrust::host_vector<double> h_model(ph_model,ph_model+num);
	    thrust::host_vector<double> h_obs(ph_obs,ph_obs+num_data);
	    thrust::host_vector<double> h_sigma(ph_sigma,ph_sigma+num_data);
	    thrust::host_vector<double> h_chisq(ph_chisq,ph_chisq+num_model);

	    thrust::counting_iterator<int> index_begin(0);
	    thrust::counting_iterator<int> index_end(num);

	    if(gpuid>=0)
	    	{

		// allocate mem on GPU
		thrust::device_vector<double> d_model(num);
		thrust::device_vector<double> d_obs(num_data);
		thrust::device_vector<double> d_sigma(num_data);
		thrust::device_vector<double> d_chisq(num_model);

		// transfer input params to GPU
//		start_timer_upload();
		d_model = h_model;
		d_obs = h_obs;
		d_sigma = h_sigma;
//		stop_timer_upload();

		// distribute the computation to the GPU
		ebf::sync();
//	        start_timer_kernel();

        typedef thrust::counting_iterator<int> CountIntIter;
        typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
        typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ObsIdxIter>  ObsIter;
        typedef thrust::tuple<thrust::device_vector<double>::iterator, ObsIter, ObsIter> ModelObsSigmaIteratorTuple;
        typedef thrust::zip_iterator<ModelObsSigmaIteratorTuple> ModelObsSigmaZipIter;
 	ModelObsSigmaZipIter input_begin = thrust::make_zip_iterator(thrust::make_tuple(
                         d_model.begin(),
                         thrust::make_permutation_iterator(d_obs.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_data))),
                         thrust::make_permutation_iterator(d_sigma.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_data))) ));

		for(int m=0;m<num_model;++m)
		   {
		   d_chisq[m] = thrust::transform_reduce(
	   	      input_begin+m*num_data,
		      input_begin+((m+1)*num_data),
		      calc_ressq_functor(), 0., thrust::plus<double>() );	
		   }
		ebf::sync();
//                 stop_timer_kernel();

		 // transfer results back to host
//		 start_timer_download();
		 thrust::copy(d_chisq.begin(),d_chisq.end(),ph_chisq);
//		 stop_timer_download();
		 }
	       else
		 {
		 // distribute the computation to the CPU
        typedef thrust::counting_iterator<int> CountIntIter;
        typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
        typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ObsIdxIter>  ObsIter;
        typedef thrust::tuple<thrust::host_vector<double>::iterator, ObsIter, ObsIter> ModelObsSigmaIteratorTuple;
        typedef thrust::zip_iterator<ModelObsSigmaIteratorTuple> ModelObsSigmaZipIter;
 	ModelObsSigmaZipIter input_begin = thrust::make_zip_iterator(thrust::make_tuple(
                         h_model.begin(),
                         thrust::make_permutation_iterator(h_obs.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_data))),
                         thrust::make_permutation_iterator(h_sigma.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_data))) ));

//	        start_timer_kernel();
		for(int m=0;m<num_model;++m)
		   {
	           h_chisq[m] = thrust::transform_reduce(
                      input_begin+m*num_data,
                      input_begin+((m+1)*num_data),
                      calc_ressq_functor(), 0., thrust::plus<double>() );
		   }
//                 stop_timer_kernel();
		  }
	}



#endif

