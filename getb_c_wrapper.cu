#ifndef CU_GETB_CWRAPPER
#define CU_GETB_CWRAPPER

#include "exofast_cuda_util.cuh"
#include "getb.cu"

// get_xyz_wrapper_C:
//         C wrapper function
// inputs: 
//         time:        
//         time_peri:        
//         period:        
//         a_over_rstar:        
//         inc:        
//         ecc:        
//         omega:        
//         num_obs:    integer size of input time array
//         num_param:    integer size of input model parameter arrays
//         ph_x: pointer to beginning element of array of doubles 
//         ph_y: pointer to beginning element of array of doubles 
//         ph_z: pointer to beginning element of array of doubles 
// outputs:
//         ph_x, ph_y, ph_z: values overwritten with x, y and z positions
// assumptions:
//         time array has at least num_data elements 
//         ph_x, ph_y, ph_z arrays have at least num_data*num_param elements 
//         other arrays have at least num_param elements
//
void get_xyz_wrapper_c(double *ph_time, double *ph_time_peri, double *ph_period, double *ph_a_over_rstar, double *ph_inc, double *ph_ecc, double *ph_omega, const int num_obs, const int num_param, double *ph_x, double *ph_y, double *ph_z )
{
	int gpuid = ebf::init_cuda();
	int num = num_obs*num_param;
	thrust::counting_iterator<int> index_begin(0);
	thrust::counting_iterator<int> index_end(num);

	// put vectors in thrust format from raw points
	thrust::host_vector<double> h_time(ph_time,ph_time+num_obs);
	thrust::host_vector<double> h_time_peri(ph_time_peri,ph_time_peri+num_param);
	thrust::host_vector<double> h_period(ph_period,ph_period+num_param);
	thrust::host_vector<double> h_a_over_rstar(ph_a_over_rstar,ph_a_over_rstar+num_param);
	thrust::host_vector<double> h_inc(ph_inc,ph_inc+num_param);
	thrust::host_vector<double> h_ecc(ph_ecc,ph_ecc+num_param);
	thrust::host_vector<double> h_omega(ph_omega,ph_omega+num_param);
	thrust::host_vector<double> h_x(ph_x,ph_x+num);
	thrust::host_vector<double> h_y(ph_y,ph_y+num);
	thrust::host_vector<double> h_z(ph_z,ph_z+num);

	if(gpuid>=0)
	{
	// transfer input params to GPU
	thrust::device_vector<double> d_time = h_time;
	thrust::device_vector<double> d_time_peri = h_time_peri;
	thrust::device_vector<double> d_period = h_period;
	thrust::device_vector<double> d_a_over_rstar = h_a_over_rstar;
	thrust::device_vector<double> d_inc = h_inc;
	thrust::device_vector<double> d_ecc = h_ecc;
	thrust::device_vector<double> d_omega = h_omega;

	// allocate mem on GPU
	thrust::device_vector<double> d_x(num), d_y(num), d_z(num);

	// prepare to distribute the computation to the GPU

	// Typedefs for efficiently fused GPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetXyzInputIteratorTuple;
    	typedef thrust::zip_iterator<GetXyzInputIteratorTuple> GetXyzInputZipIter;
    	typedef thrust::tuple<thrust::device_vector<double>::iterator,thrust::device_vector<double>::iterator,thrust::device_vector<double>::iterator> GetXyzOutputIteratorTuple;
    	typedef thrust::zip_iterator<GetXyzOutputIteratorTuple> GetXyzOutputZipIter;
    	
    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetXyzInputZipIter get_xyz_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(d_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_omega.begin(),param_idx_begin)
	));

    	GetXyzInputZipIter get_xyz_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(d_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_omega.end(),param_idx_end)
	  ));

    	GetXyzOutputZipIter get_xyz_output_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(d_x.begin(),d_y.begin(),d_z.begin() ));              
	// Compute model flux w/ limb darkening
	thrust::transform(get_xyz_input_begin,get_xyz_input_end,get_xyz_output_begin,get_xyz_functor() );

	// transfer results back to host
	thrust::copy(d_x.begin(),d_x.end(),ph_x);
	thrust::copy(d_y.begin(),d_y.end(),ph_y);
	thrust::copy(d_z.begin(),d_z.end(),ph_z);
	}
	else
	{
	// prepare to distribute the computation to the CPU

	// Typedefs for efficiently fused CPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetXyzInputIteratorTuple;
    	typedef thrust::zip_iterator<GetXyzInputIteratorTuple> GetXyzInputZipIter;
    	typedef thrust::tuple<thrust::host_vector<double>::iterator,thrust::host_vector<double>::iterator,thrust::host_vector<double>::iterator> GetXyzOutputIteratorTuple;
    	typedef thrust::zip_iterator<GetXyzOutputIteratorTuple> GetXyzOutputZipIter;
    	
    	// Fancy Iterators for efficiently fused CPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetXyzInputZipIter get_xyz_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(h_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_omega.begin(),param_idx_begin)
	));

    	GetXyzInputZipIter get_xyz_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(h_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_omega.end(),param_idx_end)
	  ));

    	GetXyzOutputZipIter get_xyz_output_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(h_x.begin(),h_y.begin(),h_z.begin() ));              
	// Compute model flux w/ limb darkening
	thrust::transform(get_xyz_input_begin,get_xyz_input_end,get_xyz_output_begin,get_xyz_functor() );

	}
}
// get_b_depth_wrapper_C:
//         C wrapper function
// inputs: 
//         time:        
//         time_peri:        
//         period:        
//         a_over_rstar:        
//         inc:        
//         ecc:        
//         omega:        
//         num_obs:    integer size of input time array
//         num_param:    integer size of input model parameter arrays
//         ph_b: pointer to beginning element of array of doubles 
//         ph_depth: pointer to beginning element of array of doubles 
// outputs:
//         ph_b, ph_depth: values overwritten
// assumptions:
//         time array has at least num_data elements 
//         ph_b, ph_depth arrays have at least num_data*num_param elements 
//         other arrays have at least num_param elements
//
void get_b_depth_wrapper_c(double *ph_time, double *ph_time_peri, double *ph_period, double *ph_a_over_rstar, double *ph_inc, double *ph_ecc, double *ph_omega, const int num_obs, const int num_param, double *ph_b, double *ph_depth )
{
	int gpuid = ebf::init_cuda();
	int num = num_obs*num_param;
	thrust::counting_iterator<int> index_begin(0);
	thrust::counting_iterator<int> index_end(num);

	// put vectors in thrust format from raw points
	thrust::host_vector<double> h_time(ph_time,ph_time+num_obs);
	thrust::host_vector<double> h_time_peri(ph_time_peri,ph_time_peri+num_param);
	thrust::host_vector<double> h_period(ph_period,ph_period+num_param);
	thrust::host_vector<double> h_a_over_rstar(ph_a_over_rstar,ph_a_over_rstar+num_param);
	thrust::host_vector<double> h_inc(ph_inc,ph_inc+num_param);
	thrust::host_vector<double> h_ecc(ph_ecc,ph_ecc+num_param);
	thrust::host_vector<double> h_omega(ph_omega,ph_omega+num_param);
	thrust::host_vector<double> h_b(ph_b,ph_b+num);
	thrust::host_vector<double> h_depth(ph_depth,ph_depth+num);

	if(gpuid>=0)
	{
	// transfer input params to GPU
	thrust::device_vector<double> d_time = h_time;
	thrust::device_vector<double> d_time_peri = h_time_peri;
	thrust::device_vector<double> d_period = h_period;
	thrust::device_vector<double> d_a_over_rstar = h_a_over_rstar;
	thrust::device_vector<double> d_inc = h_inc;
	thrust::device_vector<double> d_ecc = h_ecc;
	thrust::device_vector<double> d_omega = h_omega;

	// allocate mem on GPU
	thrust::device_vector<double> d_b(num), d_depth(num);

	// prepare to distribute the computation to the GPU

	// Typedefs for efficiently fused GPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBDepthInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBDepthInputIteratorTuple> GetBDepthInputZipIter;
    	typedef thrust::tuple<thrust::device_vector<double>::iterator,thrust::device_vector<double>::iterator> GetBDepthOutputIteratorTuple;
    	typedef thrust::zip_iterator<GetBDepthOutputIteratorTuple> GetBDepthOutputZipIter;
    	
    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetBDepthInputZipIter get_b_depth_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(d_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_omega.begin(),param_idx_begin)
	));

    	GetBDepthInputZipIter get_b_depth_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(d_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_omega.end(),param_idx_end)
	  ));

    	GetBDepthOutputZipIter get_b_depth_output_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(d_b.begin(),d_depth.begin() ));              
	// Compute model flux w/ limb darkening
	thrust::transform(get_b_depth_input_begin,get_b_depth_input_end,get_b_depth_output_begin,get_b_depth_functor() );

	// transfer results back to host
	thrust::copy(d_b.begin(),d_b.end(),ph_b);
	thrust::copy(d_depth.begin(),d_depth.end(),ph_depth);
	}
	else
	{
	// prepare to distribute the computation to the CPU
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBDepthInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBDepthInputIteratorTuple> GetBDepthInputZipIter;
    	typedef thrust::tuple<thrust::host_vector<double>::iterator,thrust::host_vector<double>::iterator> GetBDepthOutputIteratorTuple;
    	typedef thrust::zip_iterator<GetBDepthOutputIteratorTuple> GetBDepthOutputZipIter;
    	
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetBDepthInputZipIter get_b_depth_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(h_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_omega.begin(),param_idx_begin)
	));

    	GetBDepthInputZipIter get_b_depth_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(h_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_omega.end(),param_idx_end)
	  ));

    	GetBDepthOutputZipIter get_b_depth_output_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(h_b.begin(),h_depth.begin() ));              
	// Compute model flux w/ limb darkening
	thrust::transform(get_b_depth_input_begin,get_b_depth_input_end,get_b_depth_output_begin,get_b_depth_functor() );

	}
}


// get_b_wrapper_C:
//         C wrapper function
// inputs: 
//         time:        
//         time_peri:        
//         period:        
//         a_over_rstar:        
//         inc:        
//         ecc:        
//         omega:        
//         num_obs:    integer size of input time array
//         num_param:    integer size of input model parameter arrays
//         ph_b: pointer to beginning element of array of doubles 
// outputs:
//         ph_b: values overwritten
// assumptions:
//         time array has at least num_data elements 
//         ph_b arrays have at least num_data*num_param elements 
//         other arrays have at least num_param elements
//
void get_b_wrapper_c(double *ph_time, double *ph_time_peri, double *ph_period, double *ph_a_over_rstar, double *ph_inc, double *ph_ecc, double *ph_omega, const int num_obs, const int num_param, double *ph_b )
{
	int gpuid = ebf::init_cuda();
	int num = num_obs*num_param;
	thrust::counting_iterator<int> index_begin(0);
	thrust::counting_iterator<int> index_end(num);

	// put vectors in thrust format from raw points
	thrust::host_vector<double> h_time(ph_time,ph_time+num_obs);
	thrust::host_vector<double> h_time_peri(ph_time_peri,ph_time_peri+num_param);
	thrust::host_vector<double> h_period(ph_period,ph_period+num_param);
	thrust::host_vector<double> h_a_over_rstar(ph_a_over_rstar,ph_a_over_rstar+num_param);
	thrust::host_vector<double> h_inc(ph_inc,ph_inc+num_param);
	thrust::host_vector<double> h_ecc(ph_ecc,ph_ecc+num_param);
	thrust::host_vector<double> h_omega(ph_omega,ph_omega+num_param);
	thrust::host_vector<double> h_b(ph_b,ph_b+num);

	if(gpuid>=0)
	{
	// transfer input params to GPU
	thrust::device_vector<double> d_time = h_time;
	thrust::device_vector<double> d_time_peri = h_time_peri;
	thrust::device_vector<double> d_period = h_period;
	thrust::device_vector<double> d_a_over_rstar = h_a_over_rstar;
	thrust::device_vector<double> d_inc = h_inc;
	thrust::device_vector<double> d_ecc = h_ecc;
	thrust::device_vector<double> d_omega = h_omega;

	// allocate mem on GPU
	thrust::device_vector<double> d_b(num);

	// prepare to distribute the computation to the GPU

	// Typedefs for efficiently fused GPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBInputIteratorTuple> GetBInputZipIter;
    	
    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetBInputZipIter get_b_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(d_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_omega.begin(),param_idx_begin)
	));

    	GetBInputZipIter get_b_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(d_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_omega.end(),param_idx_end)
	  ));

	// Compute model flux w/ limb darkening
	thrust::transform(get_b_input_begin,get_b_input_end,d_b.begin(),get_b_functor() );

	// transfer results back to host
	thrust::copy(d_b.begin(),d_b.end(),ph_b);
	}
	else
	{
	// prepare to distribute the computation to the CPU
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBInputIteratorTuple> GetBInputZipIter;
    	
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetBInputZipIter get_b_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(h_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_omega.begin(),param_idx_begin)
	));

    	GetBInputZipIter get_b_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(h_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_omega.end(),param_idx_end)
	  ));

	// Compute model flux w/ limb darkening
	thrust::transform(get_b_input_begin,get_b_input_end,h_b.begin(),get_b_functor() );

	}
}

#endif 


