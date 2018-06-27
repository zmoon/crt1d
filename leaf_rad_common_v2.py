#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:53:42 2018

common functions used by the canopy radiative transfer models

goals for v2:
- separate plotting function
- general model class, classes for each solution method
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

@author: zmoon
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as si
#import scipy.io as sio
#from scipy.interpolate import interp1d
import scipy.optimize as so
from scipy.sparse.linalg import spsolve  # for Z&Q model
#from scipy.stats import gamma


def G_ellipsoidal(psi, x):
    """
    leaf angle factor for the ellipsoidal leaf angle dist.

    ref: Campbell 1986 eqs. 5, 6

    the ROB@UVa code set x = 0.96 instead of using the calculated 'orient' value ???

    --- in ---
    psi: zenith angle in radians
    x: b/a, ratio of horizontal semixaxis length to vertical, s.t. x > 0 indicates oblate spheroid

    """

    phi = np.pi / 2 - psi  # elevation angle

    p1 = np.sqrt( x**2 + 1/(np.tan(phi)**2) )  # numerator
    if x > 1:
        eps1 = np.sqrt(1 - x**-2)
        p2 = x + 0.5 * eps1 * x * np.log( (1 + eps1)*(1 - eps1) )  # denom

    else:
        eps2 = np.sqrt(1 - x**2)
        p2 = x + np.arcsin(eps2) / eps2  # denom

    K = p1 / p2

    return K * np.cos(psi)  # K = G / cos(psi)




mi_pcd_keys = []  # pre canopy description (used to create lai dist)
mi_cd_keys1 = ['green', 'mean_leaf_angle', 'clump']  # canopy description
mi_cd_keys2 = ['lai', 'z']
mi_bc_keys = ['wl', 'dwl', 'I_dr0_all', 'I_df_all']  # top-of-canopy BC / incoming radiation
mi_rt_keys = ['leaf_r', 'leaf_t', 'soil_r']  # reflectivity and transmissivity
mi_keys = mi_pcd_keys + mi_cd_keys1 + mi_cd_keys2 + mi_bc_keys + mi_rt_keys

class model:
    """ general class for testing 1-D canopy radiative transfer schemes
    """

    def __init__(self, scheme_ID='bl', mi={}, psi=0., nlayers=60,
                 save_dir= './', save_ID='blah', save_figs=False, write_output=True):
        """
        ;scheme_id
        ;mi model input dict (canopy description, top-of-canopy BC, leaf and soil scattering props)
        ;psi solar zenith angle (radians)
        ^ trying to use that thing that pycharm uses
        """

        #> inputs related to model input (can have time dimension)
        self.psi = psi
        self.mu = np.cos(self.psi)  # use abs here??
        try:
            self.nt = psi.size
        except AttributeError:
            self.nt = 1
        self.nlayers = nlayers

        #> assign scheme
        self._schemes = [dict(ID='bl', solver=self.solve_bl, name='Beer--Lambert'),
                         ]
        self._scheme_IDs = [d['ID'] for d in self._schemes]
        self.scheme_ID = scheme_ID
        try:
            self.scheme = self._scheme_IDs[self.scheme_ID]
        except KeyError:
            print('{:s} is not a valid scheme ID'.format(scheme_ID))
            # self.terminate() or yield error

        #> if not enough canopy description, use default
        cdd_default = load_canopy_descrip('default_canopy_descrip.csv')
        
        try:
            self.lai, self.z = mi['lai'], mi['z']
        except KeyError:
            print('LAI vertical profile not provided. Using default (Borden 1995 late June).')
            lai, z = distribute_lai(cdd_default, nlayers)
            self.lai = lai  # LAI dist (cumulative)
            self.z   = z    # height above ground (layer bottoms; m)

        self.lai_tot = self.lai.max()  # total canopy LAI
        
        for varname in mi_cd_keys1:
            if not varname in mi:
                print('{:s} not provided. Using default'.format(varname))
                mi[varname] = cdd_default[varname]
        
        self.mean_leaf_angle = mi['mean_leaf_angle']
        self.clump = mi['clump']
        self.green = mi['green']
        
        #> spectral things
        #  includes: top-of-canopy BC, leaf reflectivity and trans., soil refl, 
        #  for now, assume wavelengths from the BC apply to leaf as well
        wl_default, dwl_default, \
        I_dr0_default, I_df0_default, \
        leaf_r_default, leaf_t_default, \
        soil_r_default = load_spectral_props(173, '1212')
        
        try:
            self.I_dr0, self.I_df0 = mi['I_dr0'], mi['I_df0']
            self.wl, self.dwl = mi['wl'], mi['dwl']
        except KeyError:
            print('Top-of-canopy irradiance BC not properly provided.\nUsing default.')
            self.I_dr0 = I_dr0_default
            self.I_df0 = I_df0_default
            self.wl    = wl_default
            self.dwl   = dwl_default
        
        try:
            self.leaf_r, self.leaf_t = mi['leaf_t'], mi['leaf_r']
            if self.leaf_r.size != self.wl.size:
                print('Leaf props size different from BC. Switching to default.')
                self.leaf_r, self.leaf_t = leaf_r_default, leaf_t_default
        except KeyError:
            print('Leaf r and t not provided. Using default.')
            self.leaf_r, self.leaf_t = leaf_r_default, leaf_t_default
            
        try:
            self.soil_r = mi['soil_r']
            if self.soil_r.size != self.wl.size:
                print('Soil props size different from BC. Switching to default.')
                self.soil_r = soil_r_default
        except KeyError:
            print('Soil r not provided. Using default.')
            self.soil_r = soil_r_default

        #> allocate arrays for the spectral profile solutions
        #  all in units W/m^2
        #  though could also save conversion factor to umol/m^2/s ?
        self.nwl = self.wl.size
        self.nlevs = self.lai.size
        self.I_dr = np.zeros((self.nt, self.nlevs, self.nwl))  # direct irradiance
        self.I_df_d = np.zeros_like(self.I_dr)  # downward diffuse irradiance
        self.I_df_u = np.zeros_like(self.I_dr)
        self.F = np.zeros_like(self.I_dr)  # actinic flux

        

        #> inputs related to model output
        self.save_dir = save_dir
        self.save_id = 'blah'
        self.save_figs = False
        self.write_output = True





    # def advance(self):
    #     """ go to next time step (if necessary) """


    def run(self):
        """ """

        for self.it in np.arange(0, self.nt, 1):
            print(self.it)
            
            #> pre calculations
            #  
            self.orient = (1.0 / self.mean_leaf_angle - 0.0107) / 0.0066  # leaf orientation dist. parameter for leaf angle > 57 deg.
            self.G = self.G_fn(self.psi)  # leaf angle dist factor
            self.K_b = self.G / self.mu  # black leaf extinction coeff for direct solar beam
            
            if self.save_figs:
                print('figs')
                
            if self.write_output:
                print('out')



    def G_fn(self, psi):
        """ 
        for ellipsoidal leaf dist. could include choice of others... 
        """
        return G_ellipsoidal(psi, self.orient)
    
    
    def K_b_fn(self, psi):
        """
        """
        return self.G_fn(psi) / np.cos(psi)




    def solve_bl(self):
        """ Beer--Lambert solution
        could just be for one band, multi-band loop could be outside this fn
        """
        
        #> grab model class attributes that we need
        #  to make solver code easier to read
#        mean_leaf_angle = self.mean_leaf_angle
#        orient = self.orient
#        G = self.G
        K_b = self.K_b
        K_b_fn = self.K_b_fn
        green = self.green
        lai = self.lai
#        z = self.z
#        psi = self.psi
        mu = self.mu
        wl = self.wl
        dwl = self.dwl
        I_dr0_all = self.I_dr0
        I_df0_all = self.I_df0
        leaf_r = self.leaf_r
        leaf_t = self.leaf_t
#        soil_r = self.soil_r
#        L = self.lai_tot
        # ------------------------------------------------
        

        # functions for calculating K_df (diffuse light extinction coeff)
        #   ref. Campbell & Norman p. eq. 15.5
        #tau = lambda psi, L: np.exp(-K_b_fn(psi) * k_prime * L)  # transmittance of beam, including transmission *through* leaves (downward scattering)
        tau_b = lambda psi, L: np.exp(-K_b_fn(psi) * L)
        tau_df = lambda L: 2 * si.quad(lambda psi: tau_b(psi, L) * np.sin(psi) * np.cos(psi), 0, np.pi/2, epsrel=1e-9)[0]
        tau_df_lut = {'{:.6f}'.format(lai_val): tau_df(lai_val) for lai_val in set(lai)}  # LUT so integration is not redone for each band
        K_df_fn = lambda L: -np.log(tau_df_lut['{:.6f}'.format(L)]) / L
        K_df_fn = np.vectorize(K_df_fn)

        I_dr_all = np.zeros((lai.size, wl.size))
        I_df_d_all = np.zeros_like(I_dr_all)
        F_all = np.zeros_like(I_dr_all)

        for i, band_width in enumerate(dwl):  # run for each band individually
    
            # calculate top-of-canopy irradiance present in the band
            I_dr0 = I_dr0_all[i] * band_width  # W / m^2
            I_df0 = I_df0_all[i] * band_width
    
            # relevant properties for the band (using values at waveband LHS)
            scat = green * (leaf_t[i] + leaf_r[i])  # + (1 - green) * (other wavelength tr and rf)  ???
            alpha = 1 - scat  # absorbed by leaf; ref Moneith & Unsworth p. 47
            k_prime = np.sqrt(alpha)  # bulk attenuation coeff for a leaf; Moneith & Unsworth eq. 4.16

            K = K_b * k_prime  # approx extinction coeff for non-black leaves; ref Moneith & Unsworth p. 120
        
            # calculate profiles
            #   here I_df is just downward diffuse
            I_dr = I_dr0 * np.exp(-K_b * lai)
            I_df = I_df0 * np.exp(-K_df_fn(lai[:-1]) * lai[:-1])  # doesn't work for L=0 since L in denom
            I_df = np.append(I_df, I_df0)
    
            # approximate the contribution of scattering of the direct beam to diffuse irradiance within canopy
            #   using the grey leaf K
            I_df_dr = I_dr0 * (np.exp(-K * lai) - np.exp(-K_b * lai))
    
            # approximate the contribution to diffuse from scattered direct.
            #   assume 1/2 downward, 1/2 upward for now
            #   though a more appropriate fraction (downward vs upward) could be computed, following the Z&Q methods
            I_df += 0.5 * I_df_dr
    
            # save
            I_dr_all[:,i] = I_dr
            I_df_d_all[:,i] = I_df
            F_all[:,i] = I_dr / mu + 2 * I_df  # actinic flux (upward + downward hemisphere components)

        #> update class attrs
        self.I_dr[self.it,:,:] = I_dr_all
        self.I_df_d[self.it,:,:] = I_df_d_all
        self.I_df_u[self.it,:,:] = I_df_d_all  # assume u and d equiv for now..
        self.F[self.it,:,:] = F_all
        
        
    def solve_2s(self):
        """ 2-stream model
        
        Following Sellers (1983) and Dickinson (1985)
          mainly Dickinson -- variable names chosen to match his
        and including correction in later Dickinson paper
        
        """

        #> grab model class attributes that we need
        #  to make solver code easier to read
        mean_leaf_angle = self.mean_leaf_angle
        orient = self.orient
#        G = self.G
        G_fn = self.G_fn
#        K_b = self.K_b
        K_b_fn = self.K_b_fn
        green = self.green
        lai = self.lai
#        z = self.z
        psi = self.psi
        mu = self.mu
        wl = self.wl
        dwl = self.dwl
        I_dr0_all = self.I_dr0
        I_df0_all = self.I_df0
        leaf_r = self.leaf_r
        leaf_t = self.leaf_t
        soil_r = self.soil_r
        # ------------------------------------------------

        theta_bar = np.deg2rad(mean_leaf_angle)  # mean leaf inclination angle; eq. 3

        # mu_bar := average inverse diffuse optical depth, per unit leaf area; p. 1336
        #   sa: angle of scattered flux
        mu_bar  = si.quad(lambda sa: np.cos(sa) / G_fn(sa) * -np.sin(sa), np.pi/2, 0)[0]  # p. 1336
        mu_bar2 = si.quad(lambda mu_prime: mu_prime / G_fn(np.arccos(mu_prime)), 0, 1)[0]
        assert( mu_bar == mu_bar2 )

        # where does this stuff with Pi come from ???
        # Dickinson 1983 ?
        if orient > 1:
            bigpi = (1 - orient**-2)**0.5  # the big pi
            theta = 1 + (np.log( (1+bigpi) / (1-bigpi) ) / (2 * bigpi * orient**2))  # theta
        else:
            bigpi = (1 - orient**2)**0.5;
            theta = (1 + np.arcsin(bigpi) ) / (orient * bigpi)  #
        muave = theta * orient / (orient + 1)  # mu_bar for spherical canopy? we don't seem to need this...

        L_T = lai[0]  # total LAI

        K = K_b_fn(psi)  # for black leaves; should grey leaf version, K_b * k_prime, be used ???

        I_dr_all = np.zeros((lai.size, wl.size))
        I_df_d_all = np.zeros_like(I_dr_all)
        I_df_u_all = np.zeros_like(I_dr_all)
        F_all = np.zeros_like(I_dr_all)

        for i, band_width in enumerate(dwl):  # run for each band individually

        #    if i > 20:
        #        break

            # calculate top-of-canopy irradiance present in the band
            I_dr0 = I_dr0_all[i] * band_width  # W / m^2
            I_df0 = I_df0_all[i] * band_width

            # load canopy optical properties
            alpha = green * leaf_r[i]  # leaf element reflectance
            tau   = green * leaf_t[i]  # leaf element transmittance
            rho_s = soil_r[i]         # soil reflectivity

            omega = alpha + tau  # scattering coefficient := omega = alpha + tau

            # diffuse beam upscatter param; eq. 3
            beta = ( 0.5 * ( alpha + tau + (alpha - tau) * np.cos(theta_bar)**2) ) / omega

            # single scattering albeo a_s; Table 2, p. 1339
            #   strictly should use the ellipsoidal version, but the orientation/eccentricity param is ~ 1,
            #   so spherical is good approx, and the form is much simpler
            a_s = omega/2 * ( 1 - mu * np.log( (mu + 1) / mu) )

            # direct beam upscatter param; eq. 4
            beta_0 = (1 + mu_bar * K ) / ( omega * mu_bar * K ) * a_s


            # ----------------------------------------------------------------------------
            # intermediate params used in calculation of solns

            b = (1 - (1 - beta)*omega)
            c = omega * beta
            d = omega * mu_bar * K * beta_0
            f = omega * mu_bar * K * (1 - beta_0)
            h = (b**2 - c**2)**0.5 / mu_bar
            sigma = (mu_bar * K)**2 + c**2 - b**2

            u1 = b - c / rho_s
            u2 = b - c * rho_s
            u3 = f + c * rho_s
            S1 = np.exp(-h * L_T)
            S2 = np.exp(-K * L_T)
            p1 = b + mu_bar * h
            p2 = b - mu_bar * h
            p3 = b + mu_bar * K
            p4 = b - mu_bar * K
            D1 = p1 * (u1 - mu_bar * h) * S1**-1 - p2 * (u1 + mu_bar * h) * S1
            D2 = (u2 + mu_bar * h) * S1**-1 - (u2 - mu_bar * h) * S1

            h1 = - d * p4 - c * f
            h2 = D1**-1 * ( (d - h1 / sigma * p3) * (u1 - mu_bar * h) * S1**-1          \
                            - p2 * (d - c - h1 / sigma * (u1 + mu_bar * K) ) * S2       \
                        )
            h3 = -D1**-1  * ( (d - h1 / sigma * p3) * (u1 + mu_bar * h) * S1           \
                            - p1 * (d - c - h1 / sigma * (u1 + mu_bar * K) ) * S2    \
                            )
            h4 = -f * p3 - c * d
            h5 = -D2**-1 * ( h4 / sigma * (u2 + mu_bar * h) / S1 \
                            + (u3 - h4 / sigma * (u2 - mu_bar * K) ) * S2 \
                        )
            h6 = D2**-1  * ( h4 / sigma * (u2 - mu_bar * h) * S1 \
                            + (u3 - h4 / sigma * (u2 - mu_bar * K) ) * S2 \
                        )
            h7 =  c / D1 * (u1 - mu_bar * h) / S1
            h8 = -c / D1 * (u1 + mu_bar * h) * S1
            h9 =   D2**-1  * (u2 + mu_bar * h) / S1
            h10 = -D2**-1  * (u2 - mu_bar * h) * S1


            # ----------------------------------------------------------------------------
            # direct beam soln ???
            #   contributions to downward and upward diffuse by scattering of direct radiation by leaves

            # I feel like this should work vector-ly without needing to use vectorize
            # maybe it is the dividing by sigma that is the issue
            # I know I wouldn't have done this if there wasn't one
            I_df_u_dr_fn = np.vectorize( lambda L: h1 * np.exp(-K * L) / sigma + h2 * np.exp(-h * L) + h3 * np.exp(h * L) )
            I_df_d_dr_fn = np.vectorize( lambda L: h4 * np.exp(-K * L) / sigma + h5 * np.exp(-h * L) + h6 * np.exp(h * L) )


            # ----------------------------------------------------------------------------
            # diffuse soln ???
            #   contributions to downward and upward diffuse by attenuation/scattering of top-of-canopy diffuse

            I_df_u_df_fn = np.vectorize( lambda L: h7 * np.exp(-h * L) +  h8 * np.exp(h * L) )
            I_df_d_df_fn = np.vectorize( lambda L: h9 * np.exp(-h * L) + h10 * np.exp(h * L) )


            # apply them fns
            I_df_u_dr = I_df_u_dr_fn(lai) * I_dr0
            I_df_d_dr = I_df_d_dr_fn(lai) * I_dr0
            I_df_u_df = I_df_u_df_fn(lai) * I_df0
            I_df_d_df = I_df_d_df_fn(lai) * I_df0

            # combine the contributions
            I_df_u = I_df_u_dr + I_df_u_df
            I_df_d = I_df_d_dr + I_df_d_df


            # Beer--Lambert direct beam attenuation
            I_dr = I_dr0 * np.exp(-K * lai)


            # save
            I_dr_all[:,i] = I_dr
            I_df_d_all[:,i] = I_df_d
            I_df_u_all[:,i] = I_df_u
            F_all[:,i] = I_dr / mu + 2 * I_df_u + 2 * I_df_d

        #> update class attrs
        self.I_dr[self.it,:,:] = I_dr_all
        self.I_df_d[self.it,:,:] = I_df_d_all
        self.I_df_u[self.it,:,:] = I_df_u_all
        self.F[self.it,:,:] = F_all



    def solve_4s(self):
        """ 4-stream model
    
        all refs are to Tian et al. 2007 unless otherwise noted
    
        note that the authors use I for radiance, and F for irradiance (or actinic flux?)
        to be consistent with the other model codes I am using
          I for irradiance
          R for radiance
          F for actinic flux
    
        mi: model input dictionary
    
        """

        #> grab model class attributes that we need
        #  to make solver code easier to read
#        mean_leaf_angle = self.mean_leaf_angle
#        orient = self.orient
#        G = self.G
        G_fn = self.G_fn
#        K_b = self.K_b
        K_b_fn = self.K_b_fn
        green = self.green
        lai = self.lai
#        z = self.z
        psi = self.psi
        mu = self.mu
        wl = self.wl
        dwl = self.dwl
        I_dr0_all = self.I_dr0
        I_df0_all = self.I_df0
        leaf_r = self.leaf_r
        leaf_t = self.leaf_t
        soil_r = self.soil_r
        # ------------------------------------------------

    
        def eqns(x, y, d, direct=1):
            """ contribution to diffuse streams by scattering of direct beam
    
            x: LAI
            y: [I2d, I1d, I1u, I2u]
            d: dictionary with various info needed for computation of the params
            direct: 1 means direct beam, 0 for diffuse beam
              for diffuse beam, we want the last term of the equations eliminated
    
            """
    
            G       = d['G']  # note that these could be class attrs instead of using dict
    
            mu_0    = d['mu_0']
            mu_1    = d['mu_1']
            mu_2    = d['mu_2']
    
            alpha_p = d['alpha_p']
            alpha_m = d['alpha_m']
            beta_p  = d['beta_p']
            beta_m  = d['beta_m']
    
            gamma_p = d['gamma_p']
            gamma_m = d['gamma_m']
            eps_p1  = d['eps_p1']
            eps_m1  = d['eps_m1']
            eps_p2  = d['eps_p2']
            eps_m2  = d['eps_m2']
            k_p1    = d['k_p1']
            k_m1    = d['k_m1']
            k_p2    = d['k_p2']
            k_m2    = d['k_m2']
    
            rhs1 = 1/mu_2 * ( (alpha_p - k_m2)*y[0] + beta_p*y[1] + beta_m*y[2] + alpha_m*y[3] ) + \
                    direct * G * eps_m2 / mu_2 * np.exp(-G*x/mu_0)
    
            rhs2 = 1/mu_1 * ( beta_p*y[0] + (gamma_p - k_m1)*y[1] + gamma_m*y[2] + beta_m*y[3] ) + \
                    direct * G * eps_m1 / mu_1 * np.exp(-G*x/mu_0)
    
            rhs3 = 1/mu_1 * ( -beta_m*y[0] - gamma_m*y[1] - (gamma_p - k_p1)*y[2] - beta_p*y[3] ) - \
                    direct * G * eps_p1 / mu_1 * np.exp(-G*x/mu_0)
    
            rhs4 = 1/mu_2 * ( -alpha_m*y[0] - beta_m*y[1] - beta_p*y[2] - (alpha_p - k_p2)*y[3] ) - \
                    direct * G * eps_p2 / mu_2 * np.exp(-G*x/mu_0)
    
            return np.vstack((rhs1, rhs2, rhs3, rhs4))
    
    
    
        def dfdr_bcs(ya, yb, d, direct=1, R0=0):
            """
            eq. 8a ...
    
            direct: 1 for direct, 0 for diffuse version
              though that could be moved to be part of this fn instead
    
            """
    
            R_at_top = R0  # incident radiance (direct or diffuse)
    
            if direct == 1:
                R2d_top = 0  # note: I used for radiance instead of R in Tian et al. 2007
                R1d_top = 0
    
            elif direct == 0:
                R2d_top = R_at_top  # these should be radiance values, and we are assuming isotropic, so same values. y/n?
                R1d_top = R_at_top
    
    
            rho = d['rho']  # soil reflectance
    
            G       = d['G']
            LAI     = d['LAI']
    
            mu_0    = d['mu_0']
            mu_1    = d['mu_1']
            mu_2    = d['mu_2']
    
            # soil albedo conditions; ref eq. 8b
            I_down = 2 * np.pi * (mu_1*yb[1] + mu_2*yb[0])  # irradiance at bottom of canopy
            R_reflect = rho/np.pi * (I_down + direct * mu_0 * np.pi * R_at_top * np.exp(-G*LAI/mu_0) )
    
            # bcs
        #    return np.vstack((ya[0] - I2d_top,    # down beams at top of canopy
        #                      ya[1] - I1d_top,
        #                      yb[2] - I_reflect,  # up beams at soil surface (x = LAI)
        #                      yb[3] - I_reflect))
            return np.array([ya[0] - R2d_top,   ya[1] - R1d_top,
                             yb[2] - R_reflect, yb[3] - R_reflect])
    
    
    
        # ----------------------------------------------------------------------------------------------
    
        K = K_b_fn(psi)  # for black leaves; should grey leaf version, K_b * k_prime, be used ???
    
        mu_s = 0.501  # dividing angle for the two beams (~ 60 deg.), used by the authors
        #mu_s = 0.34  # another value used
        # really the two beams should be diff angles from the vertical !!!
    
        LAI = lai[0]  # total LAI
        P = 1  # robably phase function ???, 1 indicates isotropic scattering <-- note: not exactly satisfied in the leaf optical data (refl doesn't always = transmit)
        G = G_fn(psi)
        G_int_1 = si.quad(lambda mu_prime: G_fn(np.arccos(mu_prime)), 0, mu_s)[0]  # Barr code did not do these integrals, assumed a constant G fn
        G_int_2 = si.quad(lambda mu_prime: G_fn(np.arccos(mu_prime)), mu_s, 1)[0]
    
    
    
        I_dr_all = np.zeros((lai.size, wl.size))
        I_df_d_all = np.zeros_like(I_dr_all)
        I_df_u_all = np.zeros_like(I_dr_all)
        F_all = np.zeros_like(I_dr_all)
    
        for i, band_width in enumerate(dwl):  # run for each band individually
    
        #    if i > 20:
        #        break
    
            # calculate top-of-canopy irradiance present in the band
            I_dr0 = I_dr0_all[i] * band_width  # W / m^2
            I_df0 = I_df0_all[i] * band_width
    
            # convert to radiance, called "intensity" in the paper (Tian et al. 2007)
            #   ref for the conversion: Madronich 1987
            #   L often used for radiance, but in canopy stuff L is used for cumulative LAI so I use R
            R_dr0 = I_dr0 / (np.pi * mu)
            R_df0 = I_df0 / np.pi
    
            # soil albedo/reflectance
            rho = soil_r[i]
    
            # leaf optical props
            #   should the fact that omega isn't always 2*t = 2*r be accounted for ?
            #   could this information be used to form a more accurate phase function,
            #   since we know that the one direction (reflection or transmission) is often preffered (reflection!) ?
            r     = green * leaf_r[i]  # leaf element reflectance
            t     = green * leaf_t[i]  # leaf element transmittance
            omega = r + t               # single-scattering albedo
    
    
            # ------------------------------------------------------
            # calculate params; ref eqs. 4
            # note: assuming constant P for now, so many params have the same value
            #   should move the integration outside loop and store the value since it is the same
    
            mu_1    = 0.5 * mu_s**2
            mu_2    = 0.5 * (1 - mu_s**2)
            alpha_p = 0.5 * omega * P * (1 - mu_s) * G_int_2
            alpha_m = alpha_p
            beta_p  = 0.5 * omega * P * (1 - mu_s) * G_int_1
            beta_m  = beta_p
            gamma_p = 0.5 * omega * P * (mu_s - 0) * G_int_1
            gamma_m = gamma_p
            eps_p1  = 0.25 * omega * R_dr0 * P * (mu_s - 0)
            eps_m1  = eps_p1
            eps_p2  = 0.25 * omega * R_dr0 * P * (1 - mu_s)
            eps_m2  = eps_p2
            k_p1    = G_int_1
            k_m1    = G_int_1  # note Barr code did not have negative here. This seems to be right thing to do. at least needs abs()
            k_p2    = G_int_2
            k_m2    = G_int_2
    
    
    
            # create dict for input into the fns for the BVP solultion process
            #   probably not currently necessary since these vars are global within the model fn
            #   but would be helpful if redesign
            d = {'G': G, 'LAI': LAI,
                 'rho': rho,
                 'mu_0': mu, 'mu_1': mu_1, 'mu_2': mu_2, 'mu_s': mu_s,
                 'alpha_p': alpha_p, 'alpha_m': alpha_m,
                 'beta_p': beta_p, 'beta_m': beta_m,
                 'gamma_p': gamma_p, 'gamma_m': gamma_m,
                 'eps_p1': eps_p1, 'eps_m1': eps_m1, 'eps_p2': eps_p2, 'eps_m2': eps_m2,
                 'k_p1': k_p1, 'k_m1': k_m1, 'k_p2': k_p2, 'k_m2': k_m2}
    
    
            # ------------------------------------------------------
            # contribution to diffuse due to scattering of direct beam
    
            x = np.linspace(0, LAI, 50)
            y0 = np.ones((4, x.size))  # initial guess
    
            fun = lambda x, y: eqns(x, y, d, direct=1)
            bcs = lambda ya, yb: dfdr_bcs(ya, yb, d, direct=1, R0=R_dr0)
    
            res = si.solve_bvp(fun, bcs, x, y0, tol=1e-6)
    
            y = res.sol(lai)  # solution splined over LAI vals
    
            # compute irradiances for the 4 streams from the radiance ("intensity") solution
            I2d_dr = 2 * np.pi * mu_2 * y[0,:]
            I1d_dr = 2 * np.pi * mu_1 * y[1,:]
            I1u_dr = 2 * np.pi * mu_1 * y[2,:]
            I2u_dr = 2 * np.pi * mu_2 * y[3,:]
    
    
            # ------------------------------------------------------
            # contribution to diffuse due to attenuation of canopy-incident diffuse
    
        #    x = np.linspace(0, LAI, 50)
        #    y0 = np.ones((4, x.size))  # initial guess
    
            fun = lambda x, y: eqns(x, y, d, direct=0)
            bcs = lambda ya, yb: dfdr_bcs(ya, yb, d, direct=0,  R0=R_df0)
    
            res = si.solve_bvp(fun, bcs, x, y0, tol=1e-6)
    
            y = res.sol(lai)  # solution splined over LAI vals
    
            # compute irradiances for the 4 streams from the radiance ("intensity") solution
            I2d_df = 2 * np.pi * mu_2 * y[0,:]
            I1d_df = 2 * np.pi * mu_1 * y[1,:]
            I1u_df = 2 * np.pi * mu_1 * y[2,:]
            I2u_df = 2 * np.pi * mu_2 * y[3,:]
    
    
            # ------------------------------------------------------
            # combine results and save
    
            I2d = I2d_dr + I2d_df
            I1d = I1d_dr + I1d_df
            I1u = I1u_dr + I1u_df
            I2u = I2u_dr + I2u_df
    
            # downward and upward diffuse
            I_df_u = I1u + I2u
            I_df_d = I1d + I2d
    
            # Beer--Lambert direct beam attenuation
            I_dr = I_dr0 * np.exp(-K * lai)
    
            # save
            I_dr_all[:,i] = I_dr
            I_df_d_all[:,i] = I_df_d
            I_df_u_all[:,i] = I_df_u
            F_all[:,i] = I_dr / mu + 2 * I_df_u + 2 * I_df_d
    

        #> update class attrs
        self.I_dr[self.it,:,:] = I_dr_all
        self.I_df_d[self.it,:,:] = I_df_d_all
        self.I_df_u[self.it,:,:] = I_df_u_all
        self.F[self.it,:,:] = F_all



    def solve_zq(self):
        """ Zhao & Qualls model
    
        all refs are to Zhao & Qualls 2005 unless otherwise noted
    
        mi: model input dictionary
    
        """
    
        #> grab model class attributes that we need
        #  to make solver code easier to read
#        mean_leaf_angle = self.mean_leaf_angle
#        orient = self.orient
        G = self.G
        G_fn = self.G_fn
#        K_b = self.K_b
        K_b_fn = self.K_b_fn
        green = self.green
        lai = self.lai
#        z = self.z
        psi = self.psi
        mu = self.mu
        wl = self.wl
        dwl = self.dwl
        I_dr0_all = self.I_dr0
        I_df0_all = self.I_df0
        leaf_r = self.leaf_r
        leaf_t = self.leaf_t
        soil_r = self.soil_r
        # ------------------------------------------------

    
    
        dlai = np.diff(lai)

    
    
        # backward scattering functions
        def r_psi_fn(bl, tl, psi=psi):
            """ backscatter coeff for directional radiation; eq. 22
            """
            return 0.5 + 0.3334 * ((bl - tl)/(bl + tl)) * np.cos(psi)
    
        def r_fn(bl, tl):
            """ backscatter coeff for hemispherical (isotropic) rad.; eq. 23
            """
            return 2.0/3 * (bl / (bl + tl)) + 1.0/3 * (tl / (bl + tl))
    
    
        K = K_b_fn(psi)  # for black leaves
    
        # to find transmission of diffuse light; ref. Cambell & Norman eq. 15.5
        tau_b = lambda psi, L: np.exp(-K_b_fn(psi) * L)
        tau_df = lambda L: 2 * si.quad(lambda psi: tau_b(psi, L) * np.sin(psi) * np.cos(psi), 0, np.pi/2, epsrel=1e-9)[0]
    
        # most dlai vals are the same, so just calculate one tau_i and tau_psi value for now
        dlai_mean = np.abs(np.mean(dlai[dlai != 0]))
        tau_i_mean = tau_df(dlai_mean)
        tau_b_mean = tau_b(psi, dlai_mean)
    
    
#        LAI = lai[0]  # total LAI
#        G = G_fn(psi)
    
    
        #> lai subset??
        lai_plot = np.copy(lai)
        #lai = lai[:-1]  # upper boundary included in model
    
    
        I_dr_all = np.zeros((lai_plot.size, wl.size))
        I_df_d_all = np.zeros_like(I_dr_all)
        I_df_u_all = np.zeros_like(I_dr_all)
        I_df_d_ss_all = np.zeros_like(I_dr_all)
        I_df_u_ss_all = np.zeros_like(I_dr_all)
        F_all = np.zeros_like(I_dr_all)
        F_ss_all = np.zeros_like(I_dr_all)
    
        for i, band_width in enumerate(dwl):  # run for each band individually
    
        #    if i > 20:
        #        break
    
            # calculate top-of-canopy irradiance present in the band
            I_dr0 = I_dr0_all[i] * band_width  # W / m^2
            I_df0 = I_df0_all[i] * band_width
    
    
            # soil albedo/reflectance
            rho = soil_r[i]
            alpha0 = 1 - rho
    
            # leaf optical props; ref. p. 6, eq. 22
            beta_L  = green * leaf_r[i]  # leaf element reflectance
            tau_L   = green * leaf_t[i]  # leaf element transmittance
            alpha_L = 1 - (beta_L + tau_L)  # leaf element absorbance
    
    
            # ------------------------------------------
            # to form the matrix A we need:
            #   r_i: backward scattering fraction for layer
            #   tau_i: fraction of hemispherical shortwave radiation that totally penetrates layer (without encountering leaf)
            #     this should be equal to tau_d from Cambell & Norman (eq. 15.5)
            #   alpha_i: fraction of total intercepted radiation that is absorbed, within a layer
            #     this should be equal to the leaf element absorbance
            #
            # r, t, a varnames used instead to make it easier to read/type
    
            m = lai.size  # number of layers in the model
    
            r = r_fn(beta_L, tau_L) * np.ones((m+2, ))
            t = tau_i_mean * np.ones_like(r)  # really should calculate the value for each layer, but almost all are the same
            a = alpha_L * np.ones_like(r)
    
            # boundary values for imaginary bottom, top layers
            r[0], r[-1] = 1, 0  # at ground (z=0), no forward scattering, only back
            t[0], t[-1] = 0, 1  # no penetration into ground; 100% penetration through top ghost layer
            a[0], a[-1] = alpha0, 0  # 1 - rho absorbed at ground sfc; no absorption (by leaves) in top ghost layer
    
            A = np.zeros((2*m + 2, 2*m + 2))
    
            li = np.arange(m) + 1  # layer indices (plus 1) <- inds corresponding to the non-boundary values
    
            # now following p. 8
            A[0,0] = 1
            A[2*li-1,2*li-1-1] = -(t[li] + (1 - t[li])*(1 - a[li])*(1 - r[li]))
            A[2*li-1,2*li+0-1] = -r[li-1] * (t[li] + (1 - t[li])*(1 - a[li])*(1 - r[li])) * (1 - a[li-1]) * (1 - t[li-1])
            A[2*li-1,2*li+1-1] = 1 - r[li-1] * r[li] * (1 - a[li-1]) * (1 - t[li-1]) * (1 - a[li]) * (1 - t[li])
            A[2*li+1-1,2*li+0-1] = 1 - r[li] * r[li+1] * (1 - a[li]) * (1 - t[li]) * (1 - a[li+1]) * (1 - t[li+1])
            A[2*li+1-1,2*li+1-1] = -r[li+1] * (t[li] + (1 - t[li])*(1 - a[li])*(1 - r[li])) * (1 - a[li+1]) * (1 - t[li+1])
            A[2*li+1-1,2*li+2-1] = -(t[li] + (1 - t[li])*(1 - a[li])*(1 - r[li]))
            A[-1,-1] = 1
    
    
            # ------------------------------------------
            # to form C we need: (following p. 8 still)
            #   S: direct beam extinction
            #   r_psi
    
        #    S = I_dr0 * np.exp(-K * lai[::-1])
            S = I_dr0 * np.exp(-K * lai)
            r_psi = r_psi_fn(beta_L, tau_L)
            t_psi = tau_b_mean
    
            C = np.zeros((2*m+2, ))
    
            C[0] = rho * S[0] #rho * S.min()#rho * I_dr0
            C[2*li-1] =   (1 - r[li-1]*r[li] * (1 - a[li-1]) * (1 - t[li-1]) * (1 - a[li]) * (1 - t[li])) * \
                       r_psi * (1 - t_psi) * (1 - a[li]) * S
            C[2*li] = (1 - r[li]*r[li+1] * (1 - a[li]) * (1 - t[li]) * (1 - a[li+1]) * (1 - t[li+1])) * \
                       (1 - t_psi) * (1 - a[li]) * (1 - r_psi) * S
            C[-1] = I_df0  # top-of-canopy diffuse
    
    
            # ------------------------------------------
            # find soln to A x = C, where x is the radiation flux density (irradiance, hopefully)
            #   maybe it is supposed to actinic flux, since the term "flux density" is used
            #
            # note: could also use a Thomas algorithm solver method (write a fn for it)
            #
    
            x = spsolve(A, C)
            SWu0 = x[::2]  # "original downward and upward hemispherical shortwave radiation flux densities"
            SWd0 = x[1::2] # i.e., before multiple scattering within layers is accounted for
    
    
            # ------------------------------------------
            # multiple scattering correction
            # p. 6, eqs. 24, 25
    
            SWd = np.zeros((m+1, ))
            SWu = np.zeros_like(SWd)
    
            # note that li = 1:m
    
            # eq. 24; i+1 -> li, i -> li - 1
            SWd[li] = SWd0[li] / (1 - r[li-1]*r[li]*(1 - a[li-1])*(1 - t[li-1])*(1 - a[li])*(1 - t[li])) + \
                      r[li] * (1 - a[li]) * (1 - t[li]) * SWu0[li-1] / \
                        (1 - r[li-1]*r[li]*(1 - a[li-1])*(1 - t[li-1])*(1 - a[li])*(1 - t[li]))
    
            # eq. 25
            SWu[li-1] = SWu0[li-1] / (1 - r[li-1]*r[li]*(1 - a[li-1])*(1 - t[li-1])*(1 - a[li])*(1 - t[li])) + \
                        r[li-1] * (1 - a[li-1]) * (1 - t[li-1]) * SWd0[li] / \
                          (1 - r[li-1]*r[li]*(1 - a[li-1])*(1 - t[li-1])*(1 - a[li])*(1 - t[li]))
    
    
            # -----------------------------------------
            # save
    
        #    I_df_d_ss_all[:,i] = SWd0[:-1]  # single-scattering results
        #    I_df_d_all[:,i] =     SWd[:-1]  # after multiple-scattering corrections
        #    F_ss_all[:,i] = S / mu + 2 * SWu0[1:] + 2 * SWd0[:-1]
        #    F_all[:,i] =    S / mu +  2 * SWu[1:] +  2 * SWd[:-1]
    
            I_df_d_ss_all[:,i] = SWd0[1:]  # single-scattering results
            I_df_d_all[:,i] =     SWd[1:]  # after multiple-scattering corrections
            I_df_u_ss_all[:,i] = SWu0[:-1]  # single-scattering results
            I_df_u_all[:,i] =     SWu[:-1]  # after multiple-scattering corrections
            F_ss_all[:,i] = S / mu + 2 * SWu0[:-1] + 2 * SWd0[1:]
            F_all[:,i] =    S / mu +  2 * SWu[:-1] +  2 * SWd[1:]
    
        #    S = np.append(S[::-1], I_dr0)
        #    I_df_d_ss_all[:,i] = SWd0  # single-scattering results
        #    I_df_d_all[:,i] =     SWd  # after multiple-scattering corrections
        #    I_df_u_ss_all[:,i] = SWu0  # single-scattering results
        #    I_df_u_all[:,i] =     SWu  # after multiple-scattering corrections
        #    F_ss_all[:,i] = S / mu + 2 * SWu0 + 2 * SWd0
        #    F_all[:,i] =    S / mu +  2 * SWu +  2 * SWd
    
            I_dr_all[:,i] = I_dr0 * np.exp(-K * lai)

    
        #> update class attrs
        self.I_dr[self.it,:,:] = I_dr_all
        self.I_df_d[self.it,:,:] = I_df_d_all
        self.I_df_u[self.it,:,:] = I_df_u_all
        self.F[self.it,:,:] = F_all




    def figs_make_save(self):
        """ """
        
        #> grab model class attrs
        ID = self.scheme_ID
        I_dr_all = self.I_dr
        I_df_d_all = self.I_df_d
        lai = self.lai
        z = self.z

        plt.close('all')

        fig1, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(9.5, 6), num='diffuse-vs-direct_{:s}'.format(ID),
              gridspec_kw={'wspace':0.05, 'hspace':0})

        ax1.plot(lai, z, '.-', ms=8)
        ax1.set_xlabel(r'cum. LAI (m$^2$ m$^{-2}$)')
        ax1.set_ylabel('h (m)')

        ax2.plot(I_dr_all.sum(axis=1), z, '.-', ms=8, label='direct')
        ax2.plot(I_df_d_all.sum(axis=1), z, '.-', ms=8, label='downward diffuse')

        ax2.set_xlabel(r'irradiance (W m$^{-2}$)')
        ax2.yaxis.set_major_formatter(plt.NullFormatter())
        ax2.legend()

        ax3.plot(I_df_d_all.sum(axis=1) / (I_dr_all.sum(axis=1) + I_df_d_all.sum(axis=1)), z, '.-', ms=8)
        ax3.set_xlabel('diffuse fraction')
        ax3.yaxis.set_major_formatter(plt.NullFormatter())

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(xmin=0)
            ax.set_ylim(ymin=0)
            ax.grid('on')


        for num in plt.get_fignums():
            fig = plt.figure(num)
            figname = fig.canvas.get_window_title()
            fig.savefig('{savedir:s}/{figname:s}.pdf'.format(savedir=self.save_dir, figname=figname),
                        transparent=True,
                        bbox_inches='tight', pad_inches=0.05,
                        )










 

def distribute_lai(cdd, n):
    """
    ;cdd  canopy description dict. contains many things
    ;n    number of layers we want

    """
    LAI = cdd['lai_tot']  # total canopy LAI
    h_bottom = cdd['h_bot'][-1]  # canopy bottom is the bottom of the bottom-most layer
    
    #> form inputs to the layer class
    nlayers = len(cdd['lai_frac'])
    layers = []
    for i in range(nlayers):
        layer = dict(h_max=cdd['h_max_lad'][i],
                     h_top=cdd['h_top'][i], 
                     lad_h_top=cdd['lad_h_top'][i],
                     fLAI=cdd['lai_frac'][i])
        layers.append(layer)
    
    
    cld = canopy_lai_dist(h_bottom, layers, LAI)  # form dist of leaf area

    h_canopy = cld.lds[-1]['h_top']  # canopy height is the top of the upper-most layer

    dlai = float(LAI) / (n - 1)  # desired const lai increment
#    print dlai

    lai = np.zeros((n, ))
    z = h_canopy * np.ones_like(lai)  # bottoms of layers?

    ub = h_canopy
    LAIcum = 0
    for i in range(n-2, -1, -1):  # build from top down

        if LAI - LAIcum < dlai:
            assert( i == 0 )
            z[0] = h_bottom
            lai[0] = LAI

        else:
            lb = cld.inv_cdf(ub, dlai)
            z[i] = lb
            lai[i] = lai[i+1] + dlai

            ub = lb
            LAIcum += dlai

    return lai, z


def load_canopy_descrip(fname):
    """
    """
    varnames, vals = np.genfromtxt(fname, 
                                   usecols=(0, 1), skip_header=1, delimiter=',',  
                                   dtype=str, unpack=True)
    
    n = varnames.size
    d_raw = {varnames[i]: vals[i] for i in range(n)}
#    print d_raw
    
    d = d_raw
    for k, v in d.items():
        if ';' in v:            
            d[k] = [float(x) for x in v.split(';')]
        else:
            d[k] = float(v)
#    print d
    
    #> checks
    assert( np.array(d['lai_frac']).sum() == 1.0 )
 
    return d
    
    


def load_spectral_props(DOY, hhmm):
    """
    returns: wl, dwl, I_dr0, I_df0, leaf_r, leaf_t, soil_r

    radiation data for top-of-canopy must already exist

    """

    # ----------------------------------------------------------------------------------------------
    # radiation data from SPCTRAL2

    spectral_data_file = '../SPCTRAL2_xls/Borden_DOY{DOY:d}/EDT{hhmm:s}.csv'.format(DOY=DOY, hhmm=hhmm)
    spectral_data = np.loadtxt(spectral_data_file,
                               delimiter=',', skiprows=1)
    wl = spectral_data[:,0]    # wavelength (um); can't use 'lambda' because it is reserved
    dwl = np.append(np.diff(wl), np.nan)  # um; UV region more important so add nan to end instead of beginning, though doesn't really matter since outside leaf data bounds
    # note: could also do wl band midpoints instead..
    I_dr0 = spectral_data[:,1]  # direct (spectral) irradiance impinging on canopy (W/m^2/um)
    I_df0 = spectral_data[:,2]  # diffuse

#    wl_a = 0.35  # lower limit of leaf data; um
    wl_a = 0.30  # lower limit of leaf data, extended; um
    wl_b = 2.6   # upper limit of leaf data; um
    leaf_data_wls = (wl >= wl_a) & (wl <= wl_b)

    # use only the region where we have leaf data
    #  initially, everything should be at the SPCTRAL2 wls
    wl    = wl[leaf_data_wls]
    dwl   = dwl[leaf_data_wls]
    I_dr0 = I_dr0[leaf_data_wls]
    I_df0 = I_df0[leaf_data_wls]

    # also eliminate < 0 values ??
    #   this was added to the leaf optical props generation


    # ----------------------------------------------------------------------------------------------
    # soil reflectivity

    soil_r = np.ones_like(wl)
    soil_r[wl <= 0.7] = 0.1100  # this is the PAR value
    soil_r[wl > 0.7]  = 0.2250  # near-IR value


    # ----------------------------------------------------------------------------------------------
    # green leaf properties

    leaf_data_file = '../ideal-leaf-optical-props/leaf-idealized-rad_SPCTRAL2_wavelengths_extended.csv'
    leaf_data = np.loadtxt(leaf_data_file,
                           delimiter=',', skiprows=1)
    leaf_t = leaf_data[:,1][leaf_data_wls]
    leaf_r = leaf_data[:,2][leaf_data_wls]


    return wl, dwl, I_dr0, I_df0, leaf_r, leaf_t, soil_r









# --------------------------------------------------------------------------------------------------
#
class layer:
    """
    a layer with pdf (normalized to fLAI) and corresponding cdf methods
    """

    def __init__(self, h1, lad_h1, hmax, LAI, h2, lad_h2):
        self.h1 = h1
        self.lad_h1 = lad_h1
        self.hmax = hmax
        self.LAI = LAI
        self.h2 = h2
        self.lad_h2 = lad_h2

#        print si.quad(self.pdf0, h1, h2)[0]  # initial LAI

        fun = lambda X: si.quad(lambda h: self.pdf0(h, lai_mult=X), h1, h2)[0] - LAI
        self.lai_mult = so.fsolve(fun, 0.1)
        if self.lai_mult <= 0:
            print('desired LAI too small')
#        self.lai_mult = 0.04
#        print self.lai_mult

#        print si.quad(self.pdf, h1, h2)[0]  # new LAI


    def f_lw(self, h, lai_mult=1):
        """ """
#        linear = lambda h: (h - self.h1) * 0.07 + self.lad_h1
#        curve = lambda h: np.sqrt((h - self.h1)**2 - 1)
        curve = lambda h: np.sin(1.3 * (h - self.h1) / (self.hmax - self.h1))

        mult = 1 / (curve(self.hmax) + self.lad_h1) * lai_mult
        return curve(h) * mult + self.lad_h1




    def F_lw(self, h):
        """ cum. LAI (from bottom)
        http://www.wolframalpha.com/input/?i=integral+of+sin(1.3+*+(x+-+a)+%2F+(b+-+a))+*+c+%2B+d
        """
        curve = lambda h: np.sin(1.3 * (h - self.h1) / (self.hmax - self.h1))
        a = self.h1
        b = self.hmax
        c = 1 / (curve(self.hmax) + self.lad_h1) * self.lai_mult
        d = self.lad_h1
        return 1 / 1.3 * c * (a - b) * np.cos(1.3 * (a - h) / (a - b)) + d * h + 0



    def f_up(self, h, lai_mult=1):
        """ """
        mult = self.f_lw(self.hmax, lai_mult) - self.lad_h2
        x = (h - self.hmax) / (self.h2 - self.hmax)
        return ( -(x + 0.5)**2 + (x + 0.5) + 0.75 ) * mult + self.lad_h2


    def find_ub_up(self, lb, lai):
        """
        analytically find upper bound for integration that gives the desired lai
        with lower bound lb

        http://www.wolframalpha.com/input/?i=integrate+from+a+to+b++(-2*(x%2B0.5)%5E2+%2B+2*(x%2B0.5)+%2B+1.5)%2F2+%3D+c+solve+for+b
        """
        return None


    def F_up(self, h):
        """ cum. LAI (from bottom)
        http://www.wolframalpha.com/input/?i=integral+of+(-(x+%2B+0.5)**2+%2B+(x+%2B+0.5)+%2B+0.75)+*+a+%2B+b
        """
        x = (h - self.hmax) / (self.h2 - self.hmax)
        dhdx = self.h2 - self.hmax
        a = self.f_lw(self.hmax, self.lai_mult) - self.lad_h2
        b = self.lad_h2
        return x * (-1.0/3 * a * x**2 + a + b) * dhdx


    def pdf0(self, h, lai_mult=1):
        """
        pdf before normalization to desired LAI
        LAD at hmax = 1

        """

        if h < self.hmax:  # lower portion
            return self.f_lw(h, lai_mult)

        else:  # upper portion
            return self.f_up(h, lai_mult)


    def pdf(self, h):
        """
        using the lai multipler, integration now gives the desired LAI

        """
        if h >= self.h2:
            return np.nan

        if h >= self.hmax:  # lower portion
            return self.f_up(h, self.lai_mult)

        elif h >= self.h1:  # upper portion
            return self.f_lw(h, self.lai_mult)

        else:
            return np.nan


    def cdf(self, h):
        """ """
        lai_top = self.F_up(self.h2) - self.F_up(self.hmax)
#        print 'lai_top', lai_top

        if h >= self.h2:
            return 0

        elif h >= self.hmax:  # if gets here, is also < self.h_top
            return (self.F_up(self.h2) - self.F_up(h))

        elif h >= self.h1: # if gets here, is also < self.h_max
            return (self.F_lw(self.hmax) - self.F_lw(h)) + lai_top

        else:
            return self.LAI

## --------------------------------------------------------------------------------------------------
## tests of the layer class
#plt.close('all')
#
#layer_test = layer(2, 0, 4.5, 1.3, 8, 0.1)
#
#h = np.linspace(1, 9, 200)
#dh = np.insert(np.diff(h), 0, 0)
#dh_step = dh[-1]
#midpts = h[:-1] + dh_step/2
#
#lad0 = np.array([layer_test.pdf0(x) for x in h])
#lad = np.array([layer_test.pdf(x) for x in h])
#lai = np.array([layer_test.cdf(x) for x in h])
#dlai = np.insert(np.diff(lai), 0, 0)
#lai2 = dh_step * np.nancumsum(lad[::-1])[::-1]
#lai3 = np.array([-si.quad(layer_test.pdf, 8, x)[0] for x in midpts])
#
#plt.plot(lad0, h, label='before LAI mult')
#plt.plot(lad, h, label='after')
#plt.plot(-dlai / dh, h, 'm:')
#plt.xlim(xmin=0)
#plt.legend()
#
#plt.figure()
#
#plt.plot(lai, h, c='0.75', lw=4)
#plt.plot(lai2, h, 'm:')
#plt.plot(lai3, midpts, 'b--')  # we see that lai3 gives same as lai
#plt.xlim(xmin=0)
## --------------------------------------------------------------------------------------------------

class canopy_lai_dist:
    """
    arbitrary number of nodes, for including emergent layer, lower story, upper story, etc.
    uses the layer() class to create layers

    """

    def __init__(self, h_bottom, layers, LAI):
        """
        h_bottom: point above ground where LAD starts increasing from 0
        LAI: total LAI of the canopy

        layers: list of dicts with the following keys (for if more than 1 layer)
            h_max: height of max LAD in the section
            h_top: height of top of layer
            lad_h_top: LAD at the point where the layer ends and the layer above starts
            fLAI: fraction of total LAI that is in this layer
        """
        self.h_bottom = h_bottom
#        self.h_canopy = h_canopy
        self.LAItot = LAI

        self.h_tops = np.array([ld['h_top'] for ld in layers])

        h1 = h_bottom
        lad_h1 = 0
        for ld in layers:  # loop through the layer dicts (lds)

            hmax = ld['h_max']
            LAI = ld['fLAI'] * self.LAItot
            h2 = ld['h_top']
            lad_h2 = ld['lad_h_top']

            l = layer(h1, lad_h1, hmax, LAI, h2, lad_h2)  # create instance of layer class

            ld['pdf'] = l.pdf
            ld['cdf'] = l.cdf
            ld['LAI'] = LAI

            h1 = h2
            lad_h1 = lad_h2

        self.lds = layers
        self.LAIlayers = np.array([ld['LAI'] for ld in layers])

        # vectorize fns ?
#        self.pdf = np.vectorize(self.pdf)  # this one doesn't work for some reason
#        self.cdf = np.vectorize(self.cdf)


    def pdf(self, h):
        """

        note: could put code in here to support h being a list or np.ndarray,
        since np.vectorize seems to fail if an h value < h_bottom is included
        """

        # find relevant layer
        if h > self.h_tops.max() or h < self.h_bottom:
            return 0

        else:
            lnum = np.where(self.h_tops >= h)[0].min()
#            print lnum

            return self.lds[lnum]['pdf'](h)


    def cdf(self, h):

        if h > self.h_tops.max():
            return 0

        else:
            lnum = np.where(self.h_tops >= h)[0].min()

            if lnum == len(self.lds) - 1:
                above = 0
            else:
                above = np.sum([self.LAIlayers[lnum+1:]])

            return self.lds[lnum]['cdf'](h) + above


    def inv_cdf(self, ub, lai):
        """
        find the lb that gives that desired lai from numerical integration of the pdf

        using scipy.optimize.fsolve
        """

        fun = lambda lb: si.quad(self.pdf, lb, ub)[0] - lai
        lb = so.fsolve(fun, ub - 1e-6)

        return lb

## -------------------------------------------------------------------------------------------------
## tests of canopy_lai_dist2
#plt.close('all')
#
#l1 = {'h_max': 2.5, 'h_top': 5, 'lad_h_top': 0.1, 'fLAI': 0.2}
#l2 = {'h_max': 8, 'h_top': 13, 'lad_h_top': 0.1, 'fLAI': 0.35}
#l3 = {'h_max': 16, 'h_top': 20, 'lad_h_top': 0.05, 'fLAI': 0.35}
#l4 = {'h_max': 21.5, 'h_top': 23, 'lad_h_top': 0, 'fLAI': 0.1}
#
#cld1 = canopy_lai_dist2(0.5, [l1, l2, l3, l4], 5)
#
#h = np.linspace(0, 24, 200)
#
#lad = np.array([cld1.pdf(x) for x in h])
#lai = np.array([cld1.cdf(x) for x in h])
#
#plt.figure()
#plt.plot(lad, h)
#plt.xlim(xmin=0)
#
#plt.figure()
#plt.plot(lai, h)
#plt.xlim(xmin=0)
## -------------------------------------------------------------------------------------------------







# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    plt.close('all')

