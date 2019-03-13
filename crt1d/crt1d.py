"""


"""

from __future__ import print_function
# import os
# this_dir = os.path.dirname(os.path.realpath(__file__))

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as si
#import scipy.io as sio
#from scipy.interpolate import interp1d
import scipy.optimize as so
from scipy.sparse.linalg import spsolve  # for Z&Q model
#from scipy.stats import gamma
import xarray as xr

from .leaf_angle_dist import G_ellipsoidal
from .leaf_area_alloc import canopy_lai_dist
from .solvers import available_schemes  # also loaded in __init__.py
from .utils import distribute_lai, load_canopy_descrip





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
    


    """

    def __init__(self, scheme_ID='bl', mi={}, psi=0., dts=[], nlayers=60,  # could use **kwargs instead of mi dict?
                 save_dir= './', save_ID='blah', 
                 save_figs=False, write_output=True):
        """
        ;scheme_id
        ;mi model input dict (canopy description, top-of-canopy BC, leaf and soil scattering props)
        ;psi solar zenith angle (radians)
        ^ trying to use that thing that pycharm uses
        """

        #> inputs related to model input (can have time dimension)
        #  also want to have datetime string input for inputs with multiple time steps
        #  need to np.newaxis if the input is only one number
        self.psi_all = psi
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
            self.I_dr0_all, self.I_df0_all = mi['I_dr0'], mi['I_df0']
            self.wl, self.dwl = mi['wl'], mi['dwl']
        except KeyError:
            print('\nTop-of-canopy irradiance BC not properly provided.\nUsing default.')
            self.I_dr0 = I_dr0_default
            self.I_df0 = I_df0_default
            self.wl    = wl_default
            self.dwl   = dwl_default
        
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
        """ """

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
            
            #> run selected solver for all wavelengths
            self.solve()

            #> update history class attrs from values in cnpy_rad_state
            self.I_dr[self.it,:,:] = I_dr_all
            self.I_df_d[self.it,:,:] = I_df_d_all
            self.I_df_u[self.it,:,:] = I_df_u_all
            self.F[self.it,:,:] = F_all
            
            if self.save_figs:
                print('making them figs...')
                
            if self.write_output:
                print('writing output...')
                self.write_output_nc()



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


    def write_output_nc(self):
        """
        """
        
        inds = np.arange(0, self.nt, 1)
        z = self.z
        wl = self.wl
        
        dwl = self.dwl
        sdt = self.dts #['' for _ in inds]  # temporary..
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
        crds = ['i', 'z', 'wl']
        
        #> create dataset
        dset = xr.Dataset(\
            coords={'wl': ('wl', wl, {'units': 'um', 'longname': 'wavelength'}),
                    'z': ('z', z, {'units': 'm', 'longname': 'height above ground'}),
                    'i': inds,
                    },
            data_vars={\
                'Idr':  (crds, Idr,  {'units': 'W m^-2', 'longname': 'direct beam solar irradiance'}),
                'Idfd': (crds, Idfd, {'units': 'W m^-2', 'longname': 'downward diffuse solar irradiance'}),
                'Idfu': (crds, Idfu, {'units': 'W m^-2', 'longname': 'upward diffuse solar irradiance'}),
                'F':    (crds, F,    {'units': 'W m^-2', 'longname': 'actinic flux'}),
                'dwl':  ('wl', dwl, {'units': 'um', 'longname': 'wavelength band width'}),
                'sdt':  ('i', sdt, {'longname': 'datetime string'}),
                'sza':  ('sza', psi, {'units': 'deg.', 'longname': 'solar zenith angle'}),
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

