# ExofastGPU
============
*Library for GPU-accelerated transit light curve calculations*

Features
========
* High-performance code runs on CPU, multi-core CPUs or nVidia GPUs
* Routines easily accesible from C, C++, CUDA, and IDL

*Warning*
=========
These routines have _not_ been validated (yet).
If you find bugs, validate a routine, or would like to contribute, please notify Eric Ford.

Algorithms
==========
* Solve Kepler's Equation
* Compute position of planet/star along Keplerian orbit
* Compute stellar limb darkening models (uniform, quadratic, non-linear)
* Convert time in target barycenter frame to barycentric dynamical time
* Compute light curve model for Keplerian orbital parameters
* Calculate \chi^2 between light curve model and data

Acknowledgements
================
It's not clear if/when there will be a publication about ExofastGpu.
In the mean time, please cite:
* Mandel, K. & E. Agol (2002) ApJ 580, L171-175. (doi:10.1086/345520)
* Ford, E.B. 2009, New Astronomy, 14, 406-412.  (doi:10.1016/j.newast.2008.12.001) 
* Eastman, J. et al. (2012), submitted to PASP (arXiv:1206.5798)


Related Links
=============
* http://www.astro.washington.edu/users/agol/transit.html "Mandel & Agol code website"
* http://astroutils.astronomy.ohio-state.edu/exofast/ "Exofast website"
* http://www.astro.ufl.edu/~eford/ "Eric Ford's website"
* http://thrust.github.com/ "Thrust website"
* http://en.wikipedia.org/wiki/Comparison_of_Nvidia_graphics_processing_units#Tesla "Comparison of nVidia GPU Specifications"

