# ExofastGpu
### Library for GPU-accelerated transit light curve calculations.

For more information, see the [ExofastGpu wiki](https://github.com/AstroGPU/ExofastGPU/wiki)

Features
========
* Common algorithms for transit light curve modeling
* High-performance code is one to two orders of magnitude faster on selected nVidia GPUs than a CPU core
* Code also runs on CPUs without a GPU (either single or multi-core via OpenMP or TBB)
* Routines easily accesible from C, C++, CUDA, and IDL

*Warning*
=========
These routines have _not_ been validated (yet).
If you find bugs, validate a routine, or would like to contribute, please notify Eric Ford.


Acknowledgments
===============
It's not clear if/when there will be a publication about ExofastGpu.
In the mean time, please cite:
* Mandel, K. & E. Agol (2002) ApJ 580, L171-175. (doi:10.1086/345520)
* Ford, E.B. 2009, New Astronomy, 14, 406-412.  (doi:10.1016/j.newast.2008.12.001) 
* Dindar, S. et al. (2013), New Astronomy, 23, 6-18. (arXiv:1208.1157)
* Eastman, J. et al. (2013), PASP, 125, 83 (arXiv:1206.5798)


Related Links
=============
* [Mandel & Agol code website](http://www.astro.washington.edu/users/agol/transit.html)
* [Exofast website](http://astroutils.astronomy.ohio-state.edu/exofast/)
* [Eric Ford's website](http://www.astro.ufl.edu/~eford/)
* [Thrust website] (http://thrust.github.com/)
* [Comparison of nVidia GPU Specifications] (http://en.wikipedia.org/wiki/Comparison_of_Nvidia_graphics_processing_units#Tesla)

