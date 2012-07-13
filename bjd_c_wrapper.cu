#ifndef CU_BJD_CWRAPPER
#define CU_BJD_CWRAPPER

#include "exofast_cuda_util.cuh"

// bjd_target_wrapper_C:
//         C wrapper function
// inputs: 
//         time:        in barycentric dynamical time
//         time_peri:        
//         period:        
//         a_in_au:   I'm letting this a be the semi-major axis for the body of interest, not a of the system, so it assumes you've already multiplied by Eastman's "factor"
//         inc:        
//         ecc:        
//         omega:        
//         num_obs:    integer size of input time array
//         num_param:    integer size of input model parameter arrays
//         ph_bjd_target: pointer to beginning element of array of doubles 
//         Primary:  compile-time bool indicating if output should be based on primary, rather than secondary
// outputs:
//         ph_bjd_target: values overwritten with bjd in target barycenter frame
// assumptions:
//         time array has at least num_data elements 
//         ph_bjd_target arrays have at least num_data*num_param elements 
//         other arrays have at least num_param elements
//
template<bool Primary>
void bjd_target_wrapper_c(double *ph_time, double *ph_time_peri, double *ph_period, double *ph_a_in_au, double *ph_inc, double *ph_ecc, double *ph_omega, const int num_obs, const int num_param, double *ph_bjd_target )
{
	int gpuid = ebf::init_cuda();
	int num = num_obs*num_param;
	thrust::counting_iterator<int> index_begin(0);
	thrust::counting_iterator<int> index_end(num);

	// put vectors in thrust format from raw points
	thrust::host_vector<double> h_time(ph_time,ph_time+num_obs);
	thrust::host_vector<double> h_time_peri(ph_time_peri,ph_time_peri+num_param);
	thrust::host_vector<double> h_period(ph_period,ph_period+num_param);
	thrust::host_vector<double> h_a_in_au(ph_a_in_au,ph_a_in_au+num_param);
	thrust::host_vector<double> h_inc(ph_inc,ph_inc+num_param);
	thrust::host_vector<double> h_ecc(ph_ecc,ph_ecc+num_param);
	thrust::host_vector<double> h_omega(ph_omega,ph_omega+num_param);
	thrust::host_vector<double> h_bjd_target(ph_bjd_target,ph_bjd_target+num);

	if(gpuid>=0)
	{
	// transfer input params to GPU
	thrust::device_vector<double> d_time = h_time;
	thrust::device_vector<double> d_time_peri = h_time_peri;
	thrust::device_vector<double> d_period = h_period;
	thrust::device_vector<double> d_a_in_au = h_a_in_au;
	thrust::device_vector<double> d_inc = h_inc;
	thrust::device_vector<double> d_ecc = h_ecc;
	thrust::device_vector<double> d_omega = h_omega;

	// allocate mem on GPU
	thrust::device_vector<double> d_bjd_target(num);

	// prepare to distribute the computation to the GPU

	// Typedefs for efficiently fused GPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> BjdTargetInputIteratorTuple;
    	typedef thrust::zip_iterator<BjdTargetInputIteratorTuple> BjdTargetInputZipIter;
    	
    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	BjdTargetInputZipIter bjd_target_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(d_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_a_in_au.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_omega.begin(),param_idx_begin)
	));

    	BjdTargetInputZipIter bjd_target_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(d_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_a_in_au.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_omega.end(),param_idx_end)
	  ));

	// Compute model flux w/ limb darkening
	thrust::transform(bjd_target_input_begin,bjd_target_input_end,d_bjd_target.begin(),bjd_target_functor<Primary>() );

	// transfer results back to host
	thrust::copy(d_bjd_target.begin(),d_bjd_target.end(),ph_bjd_target);
	}
	else
	{
	// prepare to distribute the computation to the CPU
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> BjdTargetInputIteratorTuple;
    	typedef thrust::zip_iterator<BjdTargetInputIteratorTuple> BjdTargetInputZipIter;
    	
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	BjdTargetInputZipIter bjd_target_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(h_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_a_in_au.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_omega.begin(),param_idx_begin)
	));

    	BjdTargetInputZipIter bjd_target_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(h_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_a_in_au.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_omega.end(),param_idx_end)
	  ));

	// Do computation on CPU
	thrust::transform(bjd_target_input_begin,bjd_target_input_end,h_bjd_target.begin(),bjd_target_functor<Primary>() );

	}
}

#endif

