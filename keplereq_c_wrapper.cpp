#ifndef CU_KEPLEREQN_C_WRAPPER
#define CU_KEPLEREQN_C_WRAPPER

#include "exofast_cuda_util.cuh"
#include "keplereq.cu"

// ********************* BEGIN C WRAPPER FUNCTIONS ****************************** 

// keplereq_wrapper_C:
//         C wrapper function to solve's Kepler's equation num times.  
// inputs: 
//         ph_ma:  pointer to beginning element of array of doubles containing mean anomaly in radians 
//         ph_ecc: pointer to beginning element of array of doubles containing eccentricity 
//         num:    integer size of input arrays 
//         ph_eccanom: pointer to beginning element of array of doubles eccentric anomaly in radians 
// outputs:
//         ph_eccanom: values overwritten with eccentric anomaly
// assumptions:
//         input mean anomalies between 0 and 2pi
//         input eccentricities between 0 and 1
//         all three arrays have at least num elements 
//
void keplereq_wrapper_c(double *ph_ma, double *ph_ecc, int num_data, int num_model, double *ph_eccanom)
{
	int gpuid = ebf::init_cuda();
	int num = num_data*num_model;
	thrust::counting_iterator<int> index_begin(0);
	thrust::counting_iterator<int> index_end(num);

	// put vectors in thrust format from raw points
	thrust::host_vector<double> h_ecc(ph_ecc,ph_ecc+num_model);
	thrust::host_vector<double> h_ma(ph_ma,ph_ma+num_data);

	if(gpuid>=0)
	{
	// transfer input params to GPU
	thrust::device_vector<double> d_ecc = h_ecc;
	thrust::device_vector<double> d_ma = h_ma;
	// allocate mem on GPU
	thrust::device_vector<double> d_eccanom(num);

	// distribute the computation to the CPU
	ebf::sync();
	thrust::transform(
	   thrust::make_permutation_iterator(d_ma.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_data))),
	   thrust::make_permutation_iterator(d_ma.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_data)))+num,
	   thrust::make_permutation_iterator(d_ecc.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))),
	   d_eccanom.begin(),
	   keplereq_functor() );
	ebf::sync();

        // download results to host memory
	thrust::copy(d_eccanom.begin(),d_eccanom.end(),ph_eccanom);
	}
	else
	{
	// distribute the computation to the CPU
	thrust::transform(
	   thrust::make_permutation_iterator(h_ma.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_data))),
	   thrust::make_permutation_iterator(h_ma.begin(),thrust::make_transform_iterator(index_begin,modulo_stride_functor(num_data)))+num,
	   thrust::make_permutation_iterator(h_ecc.begin(),thrust::make_transform_iterator(index_begin,inverse_stride_functor(num_data))),
	   ph_eccanom,
	   keplereq_functor() );
	}
}

#endif
