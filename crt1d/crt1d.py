"""


"""

from __future__ import print_function
# import os
# this_dir = os.path.dirname(os.path.realpath(__file__))
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as si
#import scipy.io as sio
#from scipy.interpolate import interp1d
import scipy.optimize as so
from scipy.sparse.linalg import spsolve  # for Z&Q model
#from scipy.stats import gamma
import xarray as xr

from . import input_data_dir
from .leaf_angle_dist import G_ellipsoidal
from .leaf_area_alloc import canopy_lai_dist
from .solvers import available_schemes  # also loaded in __init__.py
from .utils import distribute_lai, load_default_leaf_soil_props, load_canopy_descrip





mi_pcd_keys = []  # pre canopy description (used to create lai dist)
mi_cd_keys1 = ['green', 'mean_leaf_angle', 'clump']  # canopy description
mi_cd_keys2 = ['lai', 'z']
mi_bc_keys = ['wl', 'dwl', 'I_dr0_all', 'I_df_all']  # top-of-canopy BC / incoming radiation
mi_rt_keys = ['leaf_r', 'leaf_t', 'soil_r']  # reflectivity and transmissivity
mi_keys = mi_pcd_keys + mi_cd_keys1 + mi_cd_keys2 + mi_bc_keys + mi_rt_keys



class _cnpy_descrip(dict):
    """Container for canopy description parameters and data"""
    pass

class _cnpy_rad_state(dict):
    """Container for in-canopy radiation parameters and solution at a certain time"""
    pass



class model:
    """A general class for testing 1-D canopy radiative transfer schemes.

    Optional keyword arguments are used to create the cnpy_descrip and initial cnpy_rad_state
    dicts that will be passed to the solvers. If not provided, defaults are used. 

    Initialization arguments
    ------------------------
    scheme_ID : str
        identifier for the radiation transfer model to use
    mi : dict
        model input dict (canopy description, top-of-canopy BC, leaf and soil scattering props)
    dts : np.array or list
        datetimes for each time step
    nlayers : int
        number of layers to use in the solver
    
    


    """

    def __init__(self, scheme_ID='bl', nlayers=60,
                 psi=0, dts=[], 
                 mi={},  # could use **kwargs instead of mi dict?
                 save_dir= './', save_ID='blah', 
                 save_figs=False, write_output=True):
        """
        
        scheme_id
        ;mi model input dict (canopy description, top-of-canopy BC, leaf and soil scattering props)
        ;psi solar zenith angle (radians)
        """

        #> inputs related to model input (can have time dimension)
        #  also want to have datetime string input for inputs with multiple time steps
        #  need to np.newaxis or np.array()? if the input is only one number
        self.psi_all = psi  # in radians
        self.mu_all = np.cos(self.psi_all)  # use abs here??
        try:
            self.nt = psi.size
        except AttributeError:
            self.nt = 1
        self.dts = dts  # array of datetime objects. pref np.datetime64
        self.nlayers = nlayers  # only used if no lai profile is input

        #> assign scheme
        #  could use obj here instead of dict?
        self._schemes = available_schemes
        self._scheme_IDs = {d['ID']: d for d in self._schemes}
        self.scheme_ID = scheme_ID
        try:
            self.scheme = self._scheme_IDs[self.scheme_ID]
            self.solve = self.scheme['solver']
            print('\n\n')
            print('='*40)
            print('scheme:', self.scheme_ID)
            print('-'*40)
        except KeyError:
            print('{:s} is not a valid scheme ID!'.format(scheme_ID))
            print('Defaulting to Dickinson-Sellers two-stream')
            self.scheme = self._scheme_IDs['2s']
            # also could self.terminate() or yield error or set flag

        #> if not enough canopy description, use default
        cdd_default = load_canopy_descrip(input_data_dir+'/'+'default_canopy_descrip.csv')
        
        # assuming lai profile is const for now
        # could have a check that `z[0]` is the lowest not highest level
        # and that cumulative LAI `lai` corresponds accordingly
        try:
            self.lai, self.z = mi['lai'], mi['z']
        except KeyError:
            print('\nLAI vertical profile not provided. Using default (Borden 1995 late June).')
            lai, z = distribute_lai(cdd_default, nlayers)
            self.lai = lai  # LAI dist (cumulative)
            self.z   = z    # height above ground (layer bottoms; m)

        self.lai_tot = self.lai.max()  # total canopy LAI
        
        for varname in mi_cd_keys1:
            if not varname in mi:
                print("\n`{:s}' not provided. Using default.".format(varname))
                mi[varname] = cdd_default[varname]
        
        self.mean_leaf_angle = mi['mean_leaf_angle']
        self.clump = mi['clump'][0]
        self.green = mi['green']
        
        #> spectral things
        #  includes: top-of-canopy BC, leaf reflectivity and trans., soil refl, 
        #  for now, assume wavelengths from the BC apply to leaf as well
        wl_default, dwl_default, \
        leaf_r_default, leaf_t_default, \
        soil_r_default = load_default_leaf_soil_props()
        
        try:
            self.wl, self.dwl = mi['wl'], mi['dwl']
        except KeyError:
            print('\nWavelengths and bandwidths not properly provided.\nUsing default.')
            self.wl    = wl_default
            self.dwl   = dwl_default
        
        try:
            self.I_dr0_all, self.I_df0_all = mi['I_dr0'], mi['I_df0']
        except KeyError:
            # raise()
            print('\nTop-of-canopy irradiance BC not provided.\nNoooo...')
            # sys.exit()



        try:
            self.leaf_r, self.leaf_t = mi['leaf_t'], mi['leaf_r']
            if self.leaf_r.size != self.wl.size:
                print('\nLeaf props size different from BC. Switching to default.')
                self.leaf_r, self.leaf_t = leaf_r_default, leaf_t_default
        except KeyError:
            print('\nLeaf r and t not provided. Using default.')
            self.leaf_r, self.leaf_t = leaf_r_default, leaf_t_default
            
        try:
            self.soil_r = mi['soil_r']
            if self.soil_r.size != self.wl.size:
                print('\nSoil props size different from BC. Switching to default.')
                self.soil_r = soil_r_default
        except KeyError:
            print('\nSoil r not provided. Using default.')
            self.soil_r = soil_r_default

        #> build the dicts for input into schemes
        self.cnpy_descrip = dict(\
            lai=self.lai*self.clump,  # use effective LAI (*clump) for CRT model
            mean_leaf_angle=self.mean_leaf_angle,
            green=self.green,
            leaf_t=self.leaf_t, 
            leaf_r=self.leaf_r,
            soil_r=self.soil_r,
            )


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
        self.save_id = save_ID
        self.save_figs = save_figs
        self.write_output = write_output





    # def advance(self):
    #     """ go to next time step (if necessary) """


    def run(self):
        """ 
        TODO: could add verbose option
        """

        print('\n')
        print('-'*40)
        print('now running the model...\n')
        
#        if self.nt == 1:
#            self.I_dr0 = self.I_dr0_all
#            self.I_df0 = self.I_df0_all
        
        for self.it in np.arange(0, self.nt, 1):
            print('it:', self.it)
            
            self.psi = self.psi_all[self.it]
            self.mu = self.mu_all[self.it]
            self.I_dr0 = self.I_dr0_all[self.it,:]
            self.I_df0 = self.I_df0_all[self.it,:]
            
            #> pre calculations
            #  
            self.orient = (1.0 / self.mean_leaf_angle - 0.0107) / 0.0066  # leaf orientation dist. parameter for leaf angle > 57 deg.
            self.G = self.G_fn(self.psi)  # leaf angle dist factor
            self.K_b = self.G / self.mu  # black leaf extinction coeff for direct solar beam
            # ^ should/could also incorporate clumping index for K_b
            #   could also use the class method

            #> add to canopy descrip
            self.cnpy_descrip.update(dict(\
                orient=self.orient,  # both should be moved to init
                G_fn=self.G_fn))
            
            #> build dict for input at time t_i
            self.cnpy_rad_state = dict(\
                I_dr0_all=self.I_dr0,  # at all wavelengths for one time
                I_df0_all=self.I_df0,
                wl=self.wl, dwl=self.dwl,
                psi=self.psi,
                mu=self.mu,
                K_b_fn=self.K_b_fn,
                K_b=self.K_b,
                I_dr_all=np.zeros_like(self.I_dr[self.it,:,:]),  # create empty arrays for solutions
                I_df_d_all=np.zeros_like(self.I_dr[self.it,:,:]),
                I_df_u_all=np.zeros_like(self.I_dr[self.it,:,:]),
                F_all=np.zeros_like(self.I_dr[self.it,:,:]), 
                )

            #> run selected solver for all wavelengths
            #  currently cnpy_rad_state dict is modified by the solver
            #self.cnpy_rad_state = self.solve(self.cnpy_rad_state, self.cnpy_descrip)
            self.solve(self.cnpy_rad_state, self.cnpy_descrip)

            #> update history class attrs from values in cnpy_rad_state
            self.I_dr[self.it,:,:] = self.cnpy_rad_state['I_dr_all']
            self.I_df_d[self.it,:,:] = self.cnpy_rad_state['I_df_d_all']
            self.I_df_u[self.it,:,:] = self.cnpy_rad_state['I_df_u_all']
            self.F[self.it,:,:] = self.cnpy_rad_state['F_all']
            
            if self.save_figs:  # figs for each time step
                print('making them figs...')
                self.figs_make_save()
                
            if self.write_output:
                print('writing output...')
                self.write_output_nc()

        # end iteration through time



    def G_fn(self, psi):
        """ 
        for ellipsoidal leaf dist. could include choice of others... 
        """
        return G_ellipsoidal(psi, self.orient)
    
    
    def K_b_fn(self, psi):
        """
        """
        return self.G_fn(psi) / np.cos(psi)





    def figs_make_save(self):
        """ """
        
        #> grab model class attrs
        ID = self.scheme_ID
        I_dr_all = self.I_dr[self.it,:,:]
        I_df_d_all = self.I_df_d[self.it,:,:]
        I_df_u_all = self.I_df_u[self.it,:,:]
        lai = self.lai
        z = self.z

        plt.close('all')

        fig1, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(9.5, 6), num='diffuse-vs-direct_{:s}'.format(ID),
              gridspec_kw={'wspace':0.05, 'hspace':0})

        ax1.plot(lai, z, '.-', ms=8)
        ax1.set_xlabel(r'cum. LAI (m$^2$ m$^{-2}$)')
        ax1.set_ylabel('h (m)')

        ax2.plot(I_dr_all.sum(axis=1), z, '.-', ms=8, label='direct')
        ax2.plot(I_df_u_all.sum(axis=1), z, '.-', ms=8, label='upward diffuse')
        ax2.plot(I_df_d_all.sum(axis=1), z, '.-', ms=8, label='downward diffuse')

        ax2.set_xlabel(r'irradiance (W m$^{-2}$)')
        ax2.yaxis.set_major_formatter(plt.NullFormatter())
        ax2.legend()

        ax3.plot(I_df_d_all.sum(axis=1) / (I_dr_all.sum(axis=1) + I_df_d_all.sum(axis=1)), z, '.-', ms=8)
        ax3.set_xlabel('diffuse fraction')
        ax3.yaxis.set_major_formatter(plt.NullFormatter())

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(True)

        ax1.set_title('scheme: {:s}'.format(self.scheme_ID))

        ax2.set_xlim(right=1200)
        ax2.set_title(str(pd.Timestamp(self.dts[self.it])))

        ax3.set_xlim(right=1.03)

        #> save all created figs and close them
        for num in plt.get_fignums():
            fig = plt.figure(num)
            figname = '{:s}_{:s}_it{:d}'.format(fig.canvas.get_window_title(), self.save_id, self.it)
            fig.savefig('{savedir:s}/{figname:s}.pdf'.format(savedir=self.save_dir, figname=figname),
                        transparent=True,
                        bbox_inches='tight', pad_inches=0.05,
                        )

            plt.close(fig)


    def write_output_nc(self):
        """
        """
        
        inds = np.arange(0, self.nt, 1)
        z = self.z
        wl = self.wl
        
        dwl = self.dwl

        #sdt = self.dts #['' for _ in inds]  # temporary..
        try:
            sdt = [pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S') for x in self.dts]  # datetime strings from array/list of datetimes
            time = self.dts
        except:  # e.g., of dts=[] or not convertable
            sdt = ['' for _ in inds]
            time = inds
        
        psi = self.psi_all
        lai = self.lai
        
        info = ''  # info passed in at model init?
        scheme_lname = self.scheme['longname']
        scheme_sname = self.scheme['shortname']
        scheme_id = self.scheme_ID
        
        Idr = self.I_dr
        Idfd = self.I_df_d
        Idfu = self.I_df_u
        F = self.F
        crds = ['time', 'z', 'wl']
        
        #> create dataset
        dset = xr.Dataset(\
            coords={'time': time,
                    'z': ('z', z, {'units': 'm', 'longname': 'height above ground'}),
                    'wl': ('wl', wl, {'units': 'um', 'longname': 'wavelength'}),
                    },
            data_vars={\
                'Idr':  (crds, Idr,  {'units': 'W m^-2', 'longname': 'direct beam solar irradiance'}),
                'Idfd': (crds, Idfd, {'units': 'W m^-2', 'longname': 'downward diffuse solar irradiance'}),
                'Idfu': (crds, Idfu, {'units': 'W m^-2', 'longname': 'upward diffuse solar irradiance'}),
                'F':    (crds, F,    {'units': 'W m^-2', 'longname': 'actinic flux'}),
                'dwl':  ('wl', dwl, {'units': 'um', 'longname': 'wavelength band width'}),
                'sdt':  ('time', sdt, {'longname': 'datetime string'}),
                'sza':  ('time', np.rad2deg(psi), {'units': 'deg.', 'longname': 'solar zenith angle'}),
                'lai':  ('z', lai, {'units': 'm^2 m^-2', 'longname': 'leaf area index (cumulative)'})
                },
            attrs={'save_id': self.save_id, 'info': info, 
                   'scheme_id': scheme_id, 'scheme_longname': scheme_lname, 'scheme_shortname': scheme_sname,
                   },
        )
        
        #> save dataset
        ofname = '{:s}/{:s}.nc'.format(self.save_dir, self.save_id)
        dset.to_netcdf(ofname)
        
        






# --------------------------------------------------------------------------------------------------
#




# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    plt.close('all')

