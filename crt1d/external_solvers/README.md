
Leaving these mostly untouched (except maybe radtran, which is somewhat unfinished)
placing necessary wrappers in module `solvers`.

External solvers currently incude:
* Myneni successive orders of scattering (SOSA) 1- and 2-angle, from jgomezdans/radtran
* 4SAIL: 4-stream + geometric optics, Python version from jgomezdans/radtran

Planned for future:
* Kuusk ACRM (if can easily add more layers than the 2 (similar to SAIL family))
* Gobron semi-discrete (discrete ordinates)
* SAIL++ (discrete ordinates)
