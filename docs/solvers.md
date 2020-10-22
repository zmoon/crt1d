
# Solvers

```{note}
Testing MyST with citations!
```

Dickinson--Sellers ({cite}`dickinson_land_1983,sellers_canopy_1985`) is quite common.

## Shortwave schemes

Scheme | Scheme ``name`` | Reference(s)
--- | :---: | ---
Beer--Lambert | ``bl`` | {cite}`campbell_introduction_2012`, ...
Dickinson--Sellers two-stream | ``2s`` | {cite}`dickinson_land_1983,sellers_canopy_1985,sellers_revised_1996`
Four-stream | ``4s`` | {cite}`tian_four-stream_2007`
Zhao & Qualls multi-scattering | ``zq`` | {cite}`zhao_multiple-layer_2005`
Bodin & Franklin 1.5 stream | ``bf`` | {cite}`bodin_efficient_2012`
Goudriaan one-stream | ``gd`` | {cite}`goudriaan_crop_1977,bodin_efficient_2012`
Zhao & Qualls from pyAPES | ``zq_pa`` |

## References

Bonan's 2019 book ({cite}`bonan_climate_2019`) is an excellent reference for many of the concepts,
and the implementation of the Norman scheme will be included here at some point.

```{bibliography} crt1d-refs.bib
:all:

```
