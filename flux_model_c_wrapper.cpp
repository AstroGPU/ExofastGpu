#ifndef CU_FLUX_MODEL
#define CU_FLUX_MODEL

#include<cassert>

#include "exofast_cuda_util.cuh"
#include "bjd.cu"
#include "getb.cu"
#include "occultquad.cu"
#include "chisq.cu"

// flux_model_wltt_wrapper_C:   this version is meant to worry about light travel time effects
//         C wrapper function
// inputs: 
//         time:        
//         time_peri:        
//         period:        
//         a_over_rstar:        
//         a_in_au:        // includes Eastman's factor
//         inc:        
//         ecc:        
//         omega:        
//         rp_over_rstar:        
//         u1: 
//         u2
//         num_obs:    integer size of input time array
//         num_param:    integer size of input model parameter arrays
//         ph_output: pointer to beginning element of array of doubles 
// outputs:
//         ph_output: values overwritten 
// assumptions:
//         time array has at least num_obs elements 
//         ph_output array has at least num_data*num_param elements  
//         other arrays have at least num_param elements
//
void flux_model_wltt_wrapper_c(double *ph_time, double *ph_time_peri, double *ph_period, double *ph_a_over_rstar, double *ph_a_in_au, double *ph_inc, double *ph_ecc, double *ph_omega, double *ph_rp_over_rstar, double *ph_u1, double *ph_u2, const int num_obs, const int num_param, double *ph_output)
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
	thrust::host_vector<double> h_a_in_au(ph_a_over_rstar,ph_a_over_rstar+num_param);
	thrust::host_vector<double> h_inc(ph_inc,ph_inc+num_param);
	thrust::host_vector<double> h_ecc(ph_ecc,ph_ecc+num_param);
	thrust::host_vector<double> h_omega(ph_omega,ph_omega+num_param);
	thrust::host_vector<double> h_rp_over_rstar(ph_rp_over_rstar,ph_rp_over_rstar+num_param);
	thrust::host_vector<double> h_u1(ph_u1,ph_u1+num_param);
	thrust::host_vector<double> h_u2(ph_u2,ph_u2+num_param);
	thrust::host_vector<double> h_model_flux(ph_output,ph_output+num);

	if(gpuid>=0)
	{
	// transfer input params to GPU
	thrust::device_vector<double> d_time = h_time;
	thrust::device_vector<double> d_time_peri = h_time_peri;
	thrust::device_vector<double> d_period = h_period;
	thrust::device_vector<double> d_a_over_rstar = h_a_over_rstar;
	thrust::device_vector<double> d_a_in_au = h_a_in_au;
	thrust::device_vector<double> d_inc = h_inc;
	thrust::device_vector<double> d_ecc = h_ecc;
	thrust::device_vector<double> d_omega = h_omega;
	thrust::device_vector<double> d_rp_over_rstar = h_rp_over_rstar;
	thrust::device_vector<double> d_u1 = h_u1;
	thrust::device_vector<double> d_u2 = h_u2;

	// allocate mem on GPU
	thrust::device_vector<double> d_model_flux(num);

	// prepare to distribute the computation to the GPU
	// Typedefs for efficiently fused GPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ParamIdxIter>  ParamIter;

    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> ObsTimeBjdTargetInputIteratorTuple;
    	typedef thrust::zip_iterator<ObsTimeBjdTargetInputIteratorTuple> ObsTimeBjdTargetInputZipIter;
	typedef thrust::transform_iterator< target2bjd_functor<false>, ObsTimeBjdTargetInputZipIter > ObsTimeBjdTargetIter;

    	typedef thrust::tuple<ObsTimeBjdTargetIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBInputIteratorTuple> GetBInputZipIter;
    	typedef thrust::transform_iterator<get_b_functor,GetBInputZipIter> BIter;
    	typedef thrust::tuple<BIter,ParamIter,ParamIter,ParamIter> OccultquadInputIteratorTuple;
    	typedef thrust::zip_iterator<OccultquadInputIteratorTuple> OccultquadInputZipIter;
    	typedef thrust::tuple<thrust::device_vector<double>::iterator > OccultquadOutputIteratorTuple;
    	typedef thrust::zip_iterator<OccultquadOutputIteratorTuple> OccultquadOutputZipIter;

    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

	ObsTimeBjdTargetInputZipIter obs_time_bjd_target_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(d_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_a_in_au.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_omega.begin(),param_idx_begin)
	));

        ObsTimeBjdTargetIter obs_time_bjd_target_begin = 
	   thrust::make_transform_iterator( obs_time_bjd_target_input_begin     , target2bjd_functor<false>() );
        ObsTimeBjdTargetIter obs_time_bjd_target_end = 
	   thrust::make_transform_iterator( obs_time_bjd_target_input_begin+num , target2bjd_functor<false>() );

	GetBInputZipIter getb_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    obs_time_bjd_target_begin, 
	    thrust::make_permutation_iterator(d_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_omega.begin(),param_idx_begin)
	));

    	GetBInputZipIter getb_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    obs_time_bjd_target_end,	
	    thrust::make_permutation_iterator(d_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_omega.end(),param_idx_end)
	  ));

    	  OccultquadInputZipIter occultquad_input_begin = 
	     thrust::make_zip_iterator(thrust::make_tuple(
	         thrust::make_transform_iterator(getb_input_begin,get_b_functor()),
		 thrust::make_permutation_iterator(d_u1.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(d_u2.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(d_rp_over_rstar.begin(),param_idx_begin)  ));

    	OccultquadInputZipIter occultquad_input_end = 
	   thrust::make_zip_iterator(thrust::make_tuple(
	     thrust::make_transform_iterator(getb_input_end,get_b_functor()),
	     thrust::make_permutation_iterator(d_u1.end(),param_idx_end),
	     thrust::make_permutation_iterator(d_u2.end(),param_idx_end),
	     thrust::make_permutation_iterator(d_rp_over_rstar.end(),param_idx_end)  ));

	// Compute model flux w/ limb darkening
	thrust::transform(occultquad_input_begin,occultquad_input_end,d_model_flux.begin(),occultquad_only_functor() );

	// transfer results back to host
	thrust::copy(d_model_flux.begin(),d_model_flux.end(),ph_output);
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
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBInputIteratorTuple> GetBInputZipIter;
    	typedef thrust::transform_iterator<get_b_functor,GetBInputZipIter> BIter;
    	typedef thrust::tuple<BIter,ParamIter,ParamIter,ParamIter> OccultquadInputIteratorTuple;
	typedef thrust::tuple<thrust::host_vector<double>::iterator> OccultquadOutputIteratorTuple;
    	typedef thrust::zip_iterator<OccultquadInputIteratorTuple> OccultquadInputZipIter;
    	typedef thrust::zip_iterator<OccultquadOutputIteratorTuple> OccultquadOutputZipIter;

    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetBInputZipIter getb_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(h_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_omega.begin(),param_idx_begin)
	));

    	GetBInputZipIter getb_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(h_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_omega.end(),param_idx_end)
	  ));

    	  OccultquadInputZipIter occultquad_input_begin = 
	     thrust::make_zip_iterator(thrust::make_tuple(
	         thrust::make_transform_iterator(getb_input_begin,get_b_functor()),
		 thrust::make_permutation_iterator(h_u1.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(h_u2.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(h_rp_over_rstar.begin(),param_idx_begin)  ));

    	OccultquadInputZipIter occultquad_input_end = 
	   thrust::make_zip_iterator(thrust::make_tuple(
	     thrust::make_transform_iterator(getb_input_end,get_b_functor()),
	     thrust::make_permutation_iterator(h_u1.end(),param_idx_end),
	     thrust::make_permutation_iterator(h_u2.end(),param_idx_end),
	     thrust::make_permutation_iterator(h_rp_over_rstar.end(),param_idx_end)  ));

	OccultquadOutputZipIter occultquad_output_begin = 
             thrust::make_zip_iterator(thrust::make_tuple(
             h_model_flux.begin() ));

	// Compute model flux w/ limb darkening
	thrust::transform(occultquad_input_begin,occultquad_input_end,h_model_flux.begin(),occultquad_only_functor() );

	}
}


// flux_model_wrapper_C:  this version does not worry about light travel time effects
//         C wrapper function
// inputs: 
//         time:        
//         time_peri:        
//         period:        
//         a_over_rstar:        
//         inc:        
//         ecc:        
//         omega:        
//         rp_over_rstar:        
//         u1: 
//         u2
//         num_obs:    integer size of input time array
//         num_param:    integer size of input model parameter arrays
//         ph_model_flux: pointer to beginning element of array of doubles 
// outputs:
//         ph_model_flux: values overwritten with model flux
// assumptions:
//         time array has at least num_data elements 
//         ph_model_flux array has at least num_data*num_param elements 
//         other arrays have at least num_param elements
//
void flux_model_wrapper_c(double *ph_time, double *ph_time_peri, double *ph_period, double *ph_a_over_rstar, double *ph_inc, double *ph_ecc, double *ph_omega, double *ph_rp_over_rstar, double *ph_u1, double *ph_u2, const int num_obs, const int num_param, double *ph_model_flux)
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
	thrust::host_vector<double> h_rp_over_rstar(ph_rp_over_rstar,ph_rp_over_rstar+num_param);
	thrust::host_vector<double> h_u1(ph_u1,ph_u1+num_param);
	thrust::host_vector<double> h_u2(ph_u2,ph_u2+num_param);
	thrust::host_vector<double> h_model_flux(ph_model_flux,ph_model_flux+num);

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
	thrust::device_vector<double> d_rp_over_rstar = h_rp_over_rstar;
	thrust::device_vector<double> d_u1 = h_u1;
	thrust::device_vector<double> d_u2 = h_u2;

	// allocate mem on GPU
	thrust::device_vector<double> d_model_flux(num);

	// prepare to distribute the computation to the GPU

	// Typedefs for efficiently fused GPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBInputIteratorTuple> GetBInputZipIter;
    	typedef thrust::transform_iterator<get_b_functor,GetBInputZipIter> BIter;
    	typedef thrust::tuple<BIter,ParamIter,ParamIter,ParamIter> OccultquadInputIteratorTuple;
    	typedef thrust::zip_iterator<OccultquadInputIteratorTuple> OccultquadInputZipIter;

    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetBInputZipIter getb_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(d_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_omega.begin(),param_idx_begin)
	));

    	GetBInputZipIter getb_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(d_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_omega.end(),param_idx_end)
	  ));

    	  OccultquadInputZipIter occultquad_input_begin = 
	     thrust::make_zip_iterator(thrust::make_tuple(
	         thrust::make_transform_iterator(getb_input_begin,get_b_functor()),
		 thrust::make_permutation_iterator(d_u1.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(d_u2.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(d_rp_over_rstar.begin(),param_idx_begin)  ));

    	OccultquadInputZipIter occultquad_input_end = 
	   thrust::make_zip_iterator(thrust::make_tuple(
	     thrust::make_transform_iterator(getb_input_end,get_b_functor()),
	     thrust::make_permutation_iterator(d_u1.end(),param_idx_end),
	     thrust::make_permutation_iterator(d_u2.end(),param_idx_end),
	     thrust::make_permutation_iterator(d_rp_over_rstar.end(),param_idx_end)  ));

	// Compute model flux w/ limb darkening
	thrust::transform(occultquad_input_begin,occultquad_input_end,d_model_flux.begin(),occultquad_only_functor() );

	// transfer results back to host
	thrust::copy(d_model_flux.begin(),d_model_flux.end(),ph_model_flux);
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
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBInputIteratorTuple> GetBInputZipIter;
    	typedef thrust::transform_iterator<get_b_functor,GetBInputZipIter> BIter;
    	typedef thrust::tuple<BIter,ParamIter,ParamIter,ParamIter> OccultquadInputIteratorTuple;
    	typedef thrust::zip_iterator<OccultquadInputIteratorTuple> OccultquadInputZipIter;

    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetBInputZipIter getb_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(h_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_omega.begin(),param_idx_begin)
	));

    	GetBInputZipIter getb_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(h_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_omega.end(),param_idx_end)
	  ));

    	  OccultquadInputZipIter occultquad_input_begin = 
	     thrust::make_zip_iterator(thrust::make_tuple(
	         thrust::make_transform_iterator(getb_input_begin,get_b_functor()),
		 thrust::make_permutation_iterator(h_u1.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(h_u2.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(h_rp_over_rstar.begin(),param_idx_begin)  ));

    	OccultquadInputZipIter occultquad_input_end = 
	   thrust::make_zip_iterator(thrust::make_tuple(
	     thrust::make_transform_iterator(getb_input_end,get_b_functor()),
	     thrust::make_permutation_iterator(h_u1.end(),param_idx_end),
	     thrust::make_permutation_iterator(h_u2.end(),param_idx_end),
	     thrust::make_permutation_iterator(h_rp_over_rstar.end(),param_idx_end)  ));

	// Compute model flux w/ limb darkening
	thrust::transform(occultquad_input_begin,occultquad_input_end,h_model_flux.begin(),occultquad_only_functor() );

	}
}


// chisq_flux_model_wltt_wrapper_C:   this version is meant to worry about light travel time effects
//         C wrapper function
// inputs: 
//         time:        
//         obs:        
//         sigma:        
//         time_peri:        
//         period:        
//         a_over_rstar:        
//         a_in_au:        // includes Eastman's factor
//         inc:        
//         ecc:        
//         omega:        
//         rp_over_rstar:        
//         u1: 
//         u2
//         num_obs:    integer size of input time array
//         num_param:    integer size of input model parameter arrays
//         ph_output: pointer to beginning element of array of doubles 
// outputs:
//         ph_output: values overwritten with chisq for each model
// assumptions:
//         time array has at least num_data elements 
//         ph_output array has at least num_param elelments
//         other arrays have at least num_param elements
//
void chisq_flux_model_wltt_wrapper_c(double *ph_time, double *ph_obs, double *ph_sigma, double *ph_time_peri, double *ph_period, double *ph_a_over_rstar, double *ph_a_in_au, double *ph_inc, double *ph_ecc, double *ph_omega, double *ph_rp_over_rstar, double *ph_u1, double *ph_u2, const int num_obs, const int num_param, double *ph_output)
{
	int gpuid = ebf::init_cuda();
	int num = num_obs*num_param;
	thrust::counting_iterator<int> index_begin(0);
	thrust::counting_iterator<int> index_end(num);

	// put vectors in thrust format from raw points
	thrust::host_vector<double> h_time(ph_time,ph_time+num_obs);
	thrust::host_vector<double> h_obs(ph_obs,ph_obs+num_obs);
	thrust::host_vector<double> h_sigma(ph_sigma,ph_sigma+num_obs);
	thrust::host_vector<double> h_time_peri(ph_time_peri,ph_time_peri+num_param);
	thrust::host_vector<double> h_period(ph_period,ph_period+num_param);
	thrust::host_vector<double> h_a_over_rstar(ph_a_over_rstar,ph_a_over_rstar+num_param);
	thrust::host_vector<double> h_a_in_au(ph_a_over_rstar,ph_a_over_rstar+num_param);
	thrust::host_vector<double> h_inc(ph_inc,ph_inc+num_param);
	thrust::host_vector<double> h_ecc(ph_ecc,ph_ecc+num_param);
	thrust::host_vector<double> h_omega(ph_omega,ph_omega+num_param);
	thrust::host_vector<double> h_rp_over_rstar(ph_rp_over_rstar,ph_rp_over_rstar+num_param);
	thrust::host_vector<double> h_u1(ph_u1,ph_u1+num_param);
	thrust::host_vector<double> h_u2(ph_u2,ph_u2+num_param);
	thrust::host_vector<double> h_chisq(ph_output,ph_output+num_param);

	if(gpuid>=0)
	{
	// transfer input params to GPU
	thrust::device_vector<double> d_time = h_time;
	thrust::device_vector<double> d_obs = h_obs;
	thrust::device_vector<double> d_sigma = h_sigma;
	thrust::device_vector<double> d_time_peri = h_time_peri;
	thrust::device_vector<double> d_period = h_period;
	thrust::device_vector<double> d_a_over_rstar = h_a_over_rstar;
	thrust::device_vector<double> d_a_in_au = h_a_in_au;
	thrust::device_vector<double> d_inc = h_inc;
	thrust::device_vector<double> d_ecc = h_ecc;
	thrust::device_vector<double> d_omega = h_omega;
	thrust::device_vector<double> d_rp_over_rstar = h_rp_over_rstar;
	thrust::device_vector<double> d_u1 = h_u1;
	thrust::device_vector<double> d_u2 = h_u2;

	// allocate mem on GPU
	thrust::device_vector<double> d_model_flux(num), d_chisq(num_param,0.);

	// prepare to distribute the computation to the GPU
	// Typedefs for efficiently fused GPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ParamIdxIter>  ParamIter;

    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> ObsTimeBjdTargetInputIteratorTuple;
    	typedef thrust::zip_iterator<ObsTimeBjdTargetInputIteratorTuple> ObsTimeBjdTargetInputZipIter;
	typedef thrust::transform_iterator< target2bjd_functor<false>, ObsTimeBjdTargetInputZipIter > ObsTimeBjdTargetIter;

    	typedef thrust::tuple<ObsTimeBjdTargetIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBInputIteratorTuple> GetBInputZipIter;
    	typedef thrust::transform_iterator<get_b_functor,GetBInputZipIter> BIter;
    	typedef thrust::tuple<BIter,ParamIter,ParamIter,ParamIter> OccultquadInputIteratorTuple;
    	typedef thrust::zip_iterator<OccultquadInputIteratorTuple> OccultquadInputZipIter;
    	typedef thrust::tuple<thrust::device_vector<double>::iterator > OccultquadOutputIteratorTuple;
    	typedef thrust::zip_iterator<OccultquadOutputIteratorTuple> OccultquadOutputZipIter;

    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

	ObsTimeBjdTargetInputZipIter obs_time_bjd_target_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(d_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_a_in_au.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_omega.begin(),param_idx_begin)
	));

        ObsTimeBjdTargetIter obs_time_bjd_target_begin = 
	   thrust::make_transform_iterator( obs_time_bjd_target_input_begin     , target2bjd_functor<false>() );
        ObsTimeBjdTargetIter obs_time_bjd_target_end = 
	   thrust::make_transform_iterator( obs_time_bjd_target_input_begin+num , target2bjd_functor<false>() );

	GetBInputZipIter getb_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    obs_time_bjd_target_begin, 
	    thrust::make_permutation_iterator(d_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_omega.begin(),param_idx_begin)
	));

    	GetBInputZipIter getb_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    obs_time_bjd_target_end,	
	    thrust::make_permutation_iterator(d_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_omega.end(),param_idx_end)
	  ));

    	  OccultquadInputZipIter occultquad_input_begin = 
	     thrust::make_zip_iterator(thrust::make_tuple(
	         thrust::make_transform_iterator(getb_input_begin,get_b_functor()),
		 thrust::make_permutation_iterator(d_u1.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(d_u2.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(d_rp_over_rstar.begin(),param_idx_begin)  ));

    	OccultquadInputZipIter occultquad_input_end = 
	   thrust::make_zip_iterator(thrust::make_tuple(
	     thrust::make_transform_iterator(getb_input_end,get_b_functor()),
	     thrust::make_permutation_iterator(d_u1.end(),param_idx_end),
	     thrust::make_permutation_iterator(d_u2.end(),param_idx_end),
	     thrust::make_permutation_iterator(d_rp_over_rstar.end(),param_idx_end)  ));

	OccultquadOutputZipIter occultquad_output_begin = 
             thrust::make_zip_iterator(thrust::make_tuple(
	     d_model_flux.begin() ));

	// Compute model flux w/ limb darkening
	thrust::transform(occultquad_input_begin,occultquad_input_end,d_model_flux.begin(),occultquad_only_functor() );

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cudaThreadSynchronize();
#endif 
	// Compute chisq

        typedef thrust::tuple<thrust::device_vector<double>::iterator, ObsIter, ObsIter> ModelObsSigmaIteratorTuple;
        typedef thrust::zip_iterator<ModelObsSigmaIteratorTuple> ModelObsSigmaZipIter;
        ModelObsSigmaZipIter input_begin = thrust::make_zip_iterator(thrust::make_tuple(
                         d_model_flux.begin(),
                         thrust::make_permutation_iterator(d_obs.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs))),
                         thrust::make_permutation_iterator(d_sigma.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs))) ));

         for(int m=0;m<num_param;++m)
                   {
                   // if performance matters, then check whether these are asynchronus and if not, whether writing directly to host helps
//                   ph_output[m] = thrust::transform_reduce(
                   d_chisq[m] = thrust::transform_reduce(
                      input_begin+m*num_obs,
                      input_begin+((m+1)*num_obs),
                      calc_ressq_functor(), 0., thrust::plus<double>() );
                   }
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
         cudaThreadSynchronize();
#endif

	// transfer results back to host
	thrust::copy(d_chisq.begin(),d_chisq.end(),ph_output);
	}
	else
	{
	// prepare to distribute the computation to the CPU
	thrust::host_vector<double> h_model_flux(num);

	// Typedefs for efficiently fused CPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBInputIteratorTuple> GetBInputZipIter;
    	typedef thrust::transform_iterator<get_b_functor,GetBInputZipIter> BIter;
    	typedef thrust::tuple<BIter,ParamIter,ParamIter,ParamIter> OccultquadInputIteratorTuple;
	typedef thrust::tuple<thrust::host_vector<double>::iterator> OccultquadOutputIteratorTuple;
    	typedef thrust::zip_iterator<OccultquadInputIteratorTuple> OccultquadInputZipIter;
    	typedef thrust::zip_iterator<OccultquadOutputIteratorTuple> OccultquadOutputZipIter;

    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetBInputZipIter getb_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(h_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_omega.begin(),param_idx_begin)
	));

    	GetBInputZipIter getb_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(h_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_omega.end(),param_idx_end)
	  ));

    	  OccultquadInputZipIter occultquad_input_begin = 
	     thrust::make_zip_iterator(thrust::make_tuple(
	         thrust::make_transform_iterator(getb_input_begin,get_b_functor()),
		 thrust::make_permutation_iterator(h_u1.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(h_u2.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(h_rp_over_rstar.begin(),param_idx_begin)  ));

    	OccultquadInputZipIter occultquad_input_end = 
	   thrust::make_zip_iterator(thrust::make_tuple(
	     thrust::make_transform_iterator(getb_input_end,get_b_functor()),
	     thrust::make_permutation_iterator(h_u1.end(),param_idx_end),
	     thrust::make_permutation_iterator(h_u2.end(),param_idx_end),
	     thrust::make_permutation_iterator(h_rp_over_rstar.end(),param_idx_end)  ));

	OccultquadOutputZipIter occultquad_output_begin = 
             thrust::make_zip_iterator(thrust::make_tuple(
             h_model_flux.begin() ));

	// Compute model flux w/ limb darkening
	thrust::transform(occultquad_input_begin,occultquad_input_end,h_model_flux.begin(),occultquad_only_functor() );
 
	// Compute chisq
        typedef thrust::tuple<thrust::host_vector<double>::iterator, ObsIter, ObsIter> ModelObsSigmaIteratorTuple;
        typedef thrust::zip_iterator<ModelObsSigmaIteratorTuple> ModelObsSigmaZipIter;
        ModelObsSigmaZipIter input_begin = thrust::make_zip_iterator(thrust::make_tuple(
                         h_model_flux.begin(),
                         thrust::make_permutation_iterator(h_obs.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs))),
                         thrust::make_permutation_iterator(h_sigma.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs))) ));

//              start_timer_kernel();
                for(int m=0;m<num_param;++m)
                   {
                   h_chisq[m] = thrust::transform_reduce(
                      input_begin+m*num_obs,
                      input_begin+((m+1)*num_obs),
                      calc_ressq_functor(), 0., thrust::plus<double>() );
                   }
	}
}



// chisq_flux_model_wrapper_C:  this version does not worry about light travel time effects
//         C wrapper function
// inputs: 
//         time:        
//         obs:        
//         sigma:        
//         time_peri:        
//         period:        
//         a_over_rstar:        
//         inc:        
//         ecc:        
//         omega:        
//         rp_over_rstar:        
//         u1: 
//         u2
//         num_obs:    integer size of input time array
//         num_param:    integer size of input model parameter arrays
//         ph_chisq: pointer to beginning element of array of doubles 
// outputs:
//         ph_chisq: values overwritten with model flux
// assumptions:
//         time array has at least num_obs elements 
//         ph_chisq array has at least num_param elements 
//         other arrays have at least num_param elements
//
void chisq_flux_model_wrapper_c(double *ph_time, double *ph_obs, double *ph_sigma, double *ph_time_peri, double *ph_period, double *ph_a_over_rstar, double *ph_inc, double *ph_ecc, double *ph_omega, double *ph_rp_over_rstar, double *ph_u1, double *ph_u2, const int num_obs, const int num_param, double *ph_chisq)
{
	int gpuid = ebf::init_cuda();
	int num = num_obs*num_param;
	thrust::counting_iterator<int> index_begin(0);
	thrust::counting_iterator<int> index_end(num);

	// put vectors in thrust format from raw points
	thrust::host_vector<double> h_time(ph_time,ph_time+num_obs);
	thrust::host_vector<double> h_obs(ph_obs,ph_obs+num_obs);
	thrust::host_vector<double> h_sigma(ph_sigma,ph_sigma+num_obs);
	thrust::host_vector<double> h_time_peri(ph_time_peri,ph_time_peri+num_param);
	thrust::host_vector<double> h_period(ph_period,ph_period+num_param);
	thrust::host_vector<double> h_a_over_rstar(ph_a_over_rstar,ph_a_over_rstar+num_param);
	thrust::host_vector<double> h_inc(ph_inc,ph_inc+num_param);
	thrust::host_vector<double> h_ecc(ph_ecc,ph_ecc+num_param);
	thrust::host_vector<double> h_omega(ph_omega,ph_omega+num_param);
	thrust::host_vector<double> h_rp_over_rstar(ph_rp_over_rstar,ph_rp_over_rstar+num_param);
	thrust::host_vector<double> h_u1(ph_u1,ph_u1+num_param);
	thrust::host_vector<double> h_u2(ph_u2,ph_u2+num_param);
	thrust::host_vector<double> h_chisq(ph_chisq,ph_chisq+num_param);

	if(gpuid>=0)
	{
	// transfer input params to GPU
	thrust::device_vector<double> d_time = h_time;
	thrust::device_vector<double> d_obs = h_obs;
	thrust::device_vector<double> d_sigma = h_sigma;
	thrust::device_vector<double> d_time_peri = h_time_peri;
	thrust::device_vector<double> d_period = h_period;
	thrust::device_vector<double> d_a_over_rstar = h_a_over_rstar;
	thrust::device_vector<double> d_inc = h_inc;
	thrust::device_vector<double> d_ecc = h_ecc;
	thrust::device_vector<double> d_omega = h_omega;
	thrust::device_vector<double> d_rp_over_rstar = h_rp_over_rstar;
	thrust::device_vector<double> d_u1 = h_u1;
	thrust::device_vector<double> d_u2 = h_u2;

	// allocate mem on GPU
	thrust::device_vector<double> d_chisq(num_param), d_model_flux(num);

	// prepare to distribute the computation to the GPU

	// Typedefs for efficiently fused GPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::device_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBInputIteratorTuple> GetBInputZipIter;
    	typedef thrust::transform_iterator<get_b_functor,GetBInputZipIter> BIter;
    	typedef thrust::tuple<BIter,ParamIter,ParamIter,ParamIter> OccultquadInputIteratorTuple;
    	typedef thrust::zip_iterator<OccultquadInputIteratorTuple> OccultquadInputZipIter;

    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetBInputZipIter getb_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(d_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(d_omega.begin(),param_idx_begin)
	));

    	GetBInputZipIter getb_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(d_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(d_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(d_omega.end(),param_idx_end)
	  ));

    	  OccultquadInputZipIter occultquad_input_begin = 
	     thrust::make_zip_iterator(thrust::make_tuple(
	         thrust::make_transform_iterator(getb_input_begin,get_b_functor()),
		 thrust::make_permutation_iterator(d_u1.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(d_u2.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(d_rp_over_rstar.begin(),param_idx_begin)  ));

    	OccultquadInputZipIter occultquad_input_end = 
	   thrust::make_zip_iterator(thrust::make_tuple(
	     thrust::make_transform_iterator(getb_input_end,get_b_functor()),
	     thrust::make_permutation_iterator(d_u1.end(),param_idx_end),
	     thrust::make_permutation_iterator(d_u2.end(),param_idx_end),
	     thrust::make_permutation_iterator(d_rp_over_rstar.end(),param_idx_end)  ));

	// Compute model flux w/ limb darkening
	thrust::transform(occultquad_input_begin,occultquad_input_end,d_model_flux.begin(),occultquad_only_functor() );
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cudaThreadSynchronize();
#endif
 
	// Compute chisq
        typedef thrust::tuple<thrust::device_vector<double>::iterator, ObsIter, ObsIter> ModelObsSigmaIteratorTuple;
        typedef thrust::zip_iterator<ModelObsSigmaIteratorTuple> ModelObsSigmaZipIter;
        ModelObsSigmaZipIter input_begin = thrust::make_zip_iterator(thrust::make_tuple(
                         d_model_flux.begin(),
                         thrust::make_permutation_iterator(d_obs.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs))),
                         thrust::make_permutation_iterator(d_sigma.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs))) ));

         for(int m=0;m<num_param;++m)
                   {
                   // check whether these are asynchronus and if not, whether writing directly to host solve problem
                   d_chisq[m] = thrust::transform_reduce(
                      input_begin+m*num_obs,
                      input_begin+((m+1)*num_obs),
                      calc_ressq_functor(), 0., thrust::plus<double>() );
                   }

	// transfer results back to host
	thrust::copy(d_chisq.begin(),d_chisq.end(),ph_chisq);
	}
	else
	{
	// prepare to distribute the computation to the CPU
	thrust::host_vector<double> h_model_flux(num);
	// Typedefs for efficiently fused CPU computation
    	typedef thrust::counting_iterator<int> CountIntIter;
    	typedef thrust::transform_iterator<modulo_stride_functor,CountIntIter> ObsIdxIter;
    	typedef thrust::transform_iterator<inverse_stride_functor,CountIntIter> ParamIdxIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ObsIdxIter>  ObsIter;
    	typedef thrust::permutation_iterator<thrust::host_vector<double>::iterator, ParamIdxIter>  ParamIter;
    	typedef thrust::tuple<ObsIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter,ParamIter> GetBInputIteratorTuple;
    	typedef thrust::zip_iterator<GetBInputIteratorTuple> GetBInputZipIter;
    	typedef thrust::transform_iterator<get_b_functor,GetBInputZipIter> BIter;
    	typedef thrust::tuple<BIter,ParamIter,ParamIter,ParamIter> OccultquadInputIteratorTuple;
    	typedef thrust::zip_iterator<OccultquadInputIteratorTuple> OccultquadInputZipIter;

    	// Fancy Iterators for efficiently fused GPU computation
    	ObsIdxIter obs_idx_begin = thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs));
    	ObsIdxIter obs_idx_end = thrust::make_transform_iterator(index_end,modulo_stride_functor(num_obs));
    	ParamIdxIter param_idx_begin = thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_obs));
    	ParamIdxIter param_idx_end = thrust::make_transform_iterator(index_end,inverse_stride_functor(num_obs));

    	GetBInputZipIter getb_input_begin = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.begin(),obs_idx_begin),
	    thrust::make_permutation_iterator(h_time_peri.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_period.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_a_over_rstar.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_inc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_ecc.begin(),param_idx_begin),
	    thrust::make_permutation_iterator(h_omega.begin(),param_idx_begin)
	));

    	GetBInputZipIter getb_input_end = 
	  thrust::make_zip_iterator(thrust::make_tuple(              
	    thrust::make_permutation_iterator(h_time.end(),obs_idx_end),
	    thrust::make_permutation_iterator(h_time_peri.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_period.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_a_over_rstar.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_inc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_ecc.end(),param_idx_end),
	    thrust::make_permutation_iterator(h_omega.end(),param_idx_end)
	  ));

    	  OccultquadInputZipIter occultquad_input_begin = 
	     thrust::make_zip_iterator(thrust::make_tuple(
	         thrust::make_transform_iterator(getb_input_begin,get_b_functor()),
		 thrust::make_permutation_iterator(h_u1.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(h_u2.begin(),param_idx_begin),
		 thrust::make_permutation_iterator(h_rp_over_rstar.begin(),param_idx_begin)  ));

    	OccultquadInputZipIter occultquad_input_end = 
	   thrust::make_zip_iterator(thrust::make_tuple(
	     thrust::make_transform_iterator(getb_input_end,get_b_functor()),
	     thrust::make_permutation_iterator(h_u1.end(),param_idx_end),
	     thrust::make_permutation_iterator(h_u2.end(),param_idx_end),
	     thrust::make_permutation_iterator(h_rp_over_rstar.end(),param_idx_end)  ));

	// Compute model flux w/ limb darkening
	thrust::transform(occultquad_input_begin,occultquad_input_end,h_model_flux.begin(),occultquad_only_functor() );
 
	// Compute chisq
        typedef thrust::tuple<thrust::host_vector<double>::iterator, ObsIter, ObsIter> ModelObsSigmaIteratorTuple;
        typedef thrust::zip_iterator<ModelObsSigmaIteratorTuple> ModelObsSigmaZipIter;
        ModelObsSigmaZipIter input_begin = thrust::make_zip_iterator(thrust::make_tuple(
                         h_model_flux.begin(),
                         thrust::make_permutation_iterator(h_obs.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs))),
                         thrust::make_permutation_iterator(h_sigma.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_obs))) ));

                for(int m=0;m<num_param;++m)
                   {
                   h_chisq[m] = thrust::transform_reduce(
                      input_begin+m*num_obs,
                      input_begin+((m+1)*num_obs),
                      calc_ressq_functor(), 0., thrust::plus<double>() );
                   }
	}
}


#endif 

