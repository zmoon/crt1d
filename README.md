# 1D-canopy-rad

Testbed for 1-D canopy radiative transfer schemes

## Schemes

### Current list

Scheme Name | Scheme ID in code | Reference(s)
--- | :---: | ---
Beer&ndash;Lambert | bl | [Cambell & Norman (1998)](https://www.springer.com/us/book/9780387949376), ...
Dickinson-Sellers two-stream | 2s | Dickinson-Sellers: [Dickinson (1983)](https://dx.doi.org/10.1016/S0065-2687(08)60176-4), [Sellers (1985)](https://dx.doi.org/10.1175/1520-0442(1996)009<0676:ARLSPF>2.0.CO;2)
Four-stream | 4s | [Tian et al. (2007)](https://dx.doi.org/10.1029/2006JD007545)
Zhao & Qualls multi-scattering | zq | [Zhao and Qualls (2005)](https://dx.doi.org/10.1029/2005WR004016)
Bodin & Franklin 1.5 stream | bf | [Bodin and Franklin (2012)](https://dx.doi.org/10.5194/gmd-5-535-2012)
Goudriaan one-stream | gd | [Goudriaan (1977)](http://library.wur.nl/WebQuery/wurpubs/70980)

### Schemes planning to add in the future

Scheme Name | Scheme ID in code | Reference(s)
--- | :---: | ---
Discrete ordinates one-angle | do1a | [Myneni et al. (1988)](https://doi-org.ezaccess.libraries.psu.edu/10.1016/0168-1923(88)90063-9); [jgomezdans/radtran](https://github.com/jgomezdans/radtran)
Discrete ordinates two-angle | do2a | [Myneni (1988)](https://dx.doi.org/); [jgomezdans/radtran](https://github.com/jgomezdans/radtran); 
4SAIL | 4s_sail | [Verhoef et al. (2007)](https://dx.doi.org/10.1109/TGRS.2007.895844); [jgomezdans/prosail](https://github.com/jgomezdans/prosail/blob/master/prosail/FourSAIL.py)
Semi-discrete (SOSA) | sd | [Gobron (1997)](https://dx.doi.org/10.1029/96JD04013)
Pinty two-stream | 2s_pinty | [Pinty et al. (2006)](https://dx.doi.org/10.1029/2005JD005952)
Yuan two-stream | 2s_yuan | [Yuan et al. (2017)](https://dx.doi.org/10.1002/2016MS000773)
Liang & Strahler four-stream | 4s_ls | [Liang & Strahler (1995)](https://dx.doi.org/10.1029/94JD03249)

<!-- comments
Apparently these are supposed to create an invisible comment (once rendered):

  [//]: # "Comment" and [//]: # (Comment

Url for doi links
  [](https://dx.doi.org/)
-->

## Goals

goals for v2:

- separate the plotting functionality
- general model class, classes or functions for each solution method
  - more of a testbed
  - incorporate time (multiple BCs and szas as inputs)
    - and diff leaf props
- more schemes !!
  - discrete ordinates, SOSA, N-stream, modified 2-stream, etc 
- optimize
  - use LUTs where would be most helpful
  - store default canopy descrip and such in a file and load from there
- consistent-ify coding style
  - stick to 100 cols max?
  - stick with py2 for now
- wrapper for SPCTRAL2 C or Fortran versions
  - need to find out SPCTRAL2 output vals are LHS or midpt of bands
  - need to consider this for leaf props as well!
- input:
  - add psi calculation here? or keep taking from SPCTRAL2
  - arbitrary number of maxima in canopy lai dist
    it should already do this, but need to change the way I input it in model
  - support different wl's in the different components?
- output:
  - write netCDF files (self-describing and easy time incorporation)
