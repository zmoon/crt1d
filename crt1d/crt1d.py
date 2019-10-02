"""


"""

from __future__ import print_function, division
from copy import deepcopy
# import os
# this_dir = os.path.dirname(os.path.realpath(__file__))
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import scipy.integrate as si
#import scipy.io as sio
#from scipy.interpolate import interp1d
# import scipy.optimize as so
# from scipy.sparse.linalg import spsolve  # for Z&Q model
#from scipy.stats import gamma
import xarray as xr

from . import input_data_dir
from .cases import load_default_case
from .leaf_angle_dist import G_ellipsoidal, G_ellipsoidal_approx
from .leaf_area_alloc import (\
    distribute_lai_gamma, distribute_lai_beta, distribute_lai_from_cdd )
from .solvers import ( available_schemes,  # also loaded in __init__.py
    calc_leaf_absorption )
# from .utils import load_default_leaf_soil_props, load_default_toc_spectra, load_canopy_descrip





# mi_pcd_keys = []  # pre canopy description (used to create lai dist)
# mi_cd_keys1 = ['green', 'mean_leaf_angle', 'clump']  # canopy description
# mi_cd_keys2 = ['lai', 'z']
# mi_bc_keys = ['wl', 'dwl', 'I_dr0_all', 'I_df_all']  # top-of-canopy BC / incoming radiation
# mi_rt_keys = ['leaf_r', 'leaf_t', 'soil_r']  # reflectivity and transmissivity
# mi_keys = mi_pcd_keys + mi_cd_keys1 + mi_cd_keys2 + mi_bc_keys + mi_rt_keys



# class _cnpy_descrip(dict):
#     """Container for canopy description parameters and data"""
#     pass

# class _cnpy_rad_state(dict):
#     """Container for in-canopy radiation parameters and solution at a certain time"""
#     pass





class model:
    """A general class for testing 1-D canopy radiative transfer schemes.

    Optional keyword arguments are used to create the cnpy_descrip and initial cnpy_rad_state
    dicts that will be passed to the solvers. If not provided, defaults are used. 

    Initialization arguments
    ------------------------
    scheme_ID : str
        identifier for the radiation transfer model to use
    psi : float
        solar zenith angle (radians)
    nlayers : int
        number of in-canopy layers to use in the solver
    
    mi : dict
        model input dict (canopy description, top-of-canopy BC, leaf and soil scattering props)
    dts : np.array or list
        datetimes for each time step
    nlayers : int
        number of layers to use in the solver
    
    save_dir : str
        where to save the output nc files
    save_ID : str
        basename for the output nc filename
    save_figs : bool
    write_output : bool
    


    """

    required_input_keys = (\
        'lai', 'z',
        'mean_leaf_angle', 
        'green', 
        'leaf_t', 
        'leaf_r',
        'soil_r',
        'wl_leafsoil', 
        'orient',  # don't really need both this and mla as input
        'G_fn',
        'I_dr0_all', 'I_df0_all',  # spectral (W/m^2/um)
        'wl', 'dwl',  # for the toc spectra
        'psi', 
        )

    def __init__(self, scheme_ID='2s', nlayers=60,
                 save_dir= './', save_ID='blah', 
                 save_figs=False, write_output=False):


        #> load default case, for given nlayers
        self.nlayers = nlayers 
        self.cnpy_descrip_default, self.cnpy_rad_state_default = \
            load_default_case(nlayers=self.nlayers)

        #> create the dicts from the defaults
        #  these can be modified by the user
        self.cnpy_descrip   = deepcopy(self.cnpy_descrip_default)
        self.cnpy_rad_state = deepcopy(self.cnpy_rad_state_default)

        self.cd = self.cnpy_descrip
        self.crs = self.cnpy_rad_state
        # ^ aliases for easy checking

        # TODO: store last_state and runcount?
        self.run_count = 0

        #> check
        self.check_inputs()

        #> assign scheme
        self.assign_scheme(scheme_ID)

        #> allocate arrays for the spectral profile solutions
        #  all in units W/m^2
        #  though could also save conversion factor to umol/m^2/s ?

        # self.I_dr = np.zeros((self.nt, self.nlevs, self.nwl))  # < please pylint
        # self.I_df_d = np.zeros((self.nt, self.nlevs, self.nwl))
        # self.I_df_u = np.zeros((self.nt, self.nlevs, self.nwl))
        # self.F = np.zeros((self.nt, self.nlevs, self.nwl))

        #> inputs related to model output
        self.save_dir = save_dir
        self.save_id = save_ID
        self.save_figs = save_figs
        self.write_output = write_output



    def assign_scheme(self, scheme_ID, verbose=True):
        """Using the available_schemes dict, 
        assign scheme and necessary scheme attrs
        """
        self._schemes = available_schemes
        # self._scheme_IDs = {d['ID']: d for d in self._schemes}
        self._scheme_IDs = list(self._schemes.keys())
        self.scheme_ID = scheme_ID
        try:
            self.scheme = self._schemes[self.scheme_ID]
            self.solve = self.scheme['solver']
            self.solver_args = self.scheme['args']
            if verbose:
                print('\n\n')
                print('='*40)
                print('scheme:', self.scheme_ID)
                print('-'*40)
        except KeyError:
            print('{:s} is not a valid scheme ID!'.format(scheme_ID))
            print('Defaulting to Dickinson-Sellers two-stream')
            self.scheme = self._schemes['2s']
            # also could self.terminate() or yield error or set flag


    def check_inputs(self):
        """
        Check input LAI profile and compute additional vars from it...,
        Check wls match optical props and toc spectra...

        update some class var (e.g. nlayers)
        """
        crs = self.cnpy_rad_state
        cd  = self.cnpy_descrip

        # check for required input all at once variables
        cdcrs = {**cd, **crs}
        for key in model.required_input_keys:
            if key not in cdcrs:
                print(f'required key {key} is not present in the cnpy_* dicts')


        lai = cd['lai']
        z   = cd['z']
        assert( z.size == lai.size )
        self.nlev = lai.size
        assert( z[-1] > z[0] )  # z increasing
        assert( lai[0] > lai[-1] ) # LAI decreasing
        assert( lai[-1] == 0 )
        cd['lai_tot'] = lai[0]
        cd['lai_eff'] = lai*cd['clump']
        cd['dlai'] = lai[:-1] - lai[1:]

        # check mla and orient/x similar to psi/mu

        psi = crs['psi']  # precendence to psi
        try:
            mu = crs['mu']
            if mu != np.cos(psi):
                print('provided mu not consistent with provided psi')
        except KeyError:  # no mu
            crs['mu'] = np.cos(psi)

        # check the two wl sources
        wl_toc = crs['wl']          # for the toc spectra
        wl_op  = cd['wl_leafsoil']  # for optical properties
        if not np.allclose( wl_toc, wl_op ):
            print('wl for optical props and toc BC appear to be incompatible')
            # or could convert, but which to choose as base?
        self.nwl = wl_toc.size

        # G_fn and K_b_fn ?
        # crs['G_fn'] = cd['G_fn']
        cd['K_b_fn'] = lambda psi_: cd['G_fn'](psi_)/np.cos(psi_)
        crs['G']   = cd['G_fn'](psi)
        crs['K_b'] = cd['K_b_fn'](psi)
        # ^ should clumping index be included somewhere here?
        

        
        
        


    # def advance(self):
    #     """ go to next time step (if necessary) """


    def run(self):
        """ 
        TODO: could add verbose option
        """

        self.run_count += 1

        # check wavelengths are compatbile, etc.

        #> check inputs and fill out (other pre-calculations)
        self.check_inputs()

        # print('\n')
        # print('-'*40)
        # print('now running the model...\n')

        #
        #> run selected solver for all wavelengths
        #  since the solvers are for now written to be fully modular,
        #  the dict must be updated manually  
        #
        d = {**self.cnpy_rad_state, **self.cnpy_descrip}
        args = {k:d[k] for k in self.solver_args}

        I_dr, I_df_d, I_df_u, F = self.solve(**args)

        self.cnpy_rad_state.update(dict(\
            I_dr=I_dr, I_df_d=I_df_d, I_df_u=I_df_u, F=F,
            ))


        #> calculate absorption
        self.cnpy_rad_state.update(\
            calc_leaf_absorption(self.cnpy_descrip, self.cnpy_rad_state)
            )


        # if self.save_figs:  # figs for each time step
        #     print('making them figs...')
        #     self.figs_make_save()
        
        # #> write combined output file
        # if self.write_output:
        #     print('writing output...')
        #     self.write_output_nc()

        # end iteration through time



    # def G_fn(self, psi):
    #     """ 
    #     for ellipsoidal leaf dist. could include choice of others... 
    #     """
    #     #return G_ellipsoidal(psi, self.orient)
    #     return G_ellipsoidal_approx(psi, self.orient)
    
    
    # def K_b_fn(self, psi):
    #     """
    #     """
    #     return self.G_fn(psi) / np.cos(psi)





    # def figs_make_save(self):
    #     """ """
        
    #     #> grab model class attrs
    #     ID = self.scheme_ID
    #     I_dr_all = self.I_dr[self.it,:,:]
    #     I_df_d_all = self.I_df_d[self.it,:,:]
    #     I_df_u_all = self.I_df_u[self.it,:,:]
    #     lai = self.lai
    #     z = self.z

    #     plt.close('all')

    #     _, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(9.5, 6), num='diffuse-vs-direct_{:s}'.format(ID),
    #         gridspec_kw={'wspace':0.05, 'hspace':0})

    #     ax1.plot(lai, z, '.-', ms=8)
    #     ax1.set_xlabel(r'cum. LAI (m$^2$ m$^{-2}$)')
    #     ax1.set_ylabel('h (m)')

    #     ax2.plot(I_dr_all.sum(axis=1), z, '.-', ms=8, label='direct')
    #     ax2.plot(I_df_u_all.sum(axis=1), z, '.-', ms=8, label='upward diffuse')
    #     ax2.plot(I_df_d_all.sum(axis=1), z, '.-', ms=8, label='downward diffuse')

    #     ax2.set_xlabel(r'irradiance (W m$^{-2}$)')
    #     ax2.yaxis.set_major_formatter(plt.NullFormatter())
    #     ax2.legend()

    #     ax3.plot(I_df_d_all.sum(axis=1) / (I_dr_all.sum(axis=1) + I_df_d_all.sum(axis=1)), z, '.-', ms=8)
    #     ax3.set_xlabel('diffuse fraction')
    #     ax3.yaxis.set_major_formatter(plt.NullFormatter())

    #     for ax in [ax1, ax2, ax3]:
    #         ax.set_xlim(left=0)
    #         ax.set_ylim(bottom=0)
    #         ax.grid(True)

    #     ax1.set_title('scheme: {:s}'.format(self.scheme_ID))

    #     ax2.set_xlim(right=1200)
    #     ax2.set_title(str(pd.Timestamp(self.dts[self.it])))

    #     ax3.set_xlim(right=1.03)

    #     #> save all created figs and close them
    #     for num in plt.get_fignums():
    #         fig = plt.figure(num)
    #         figname = '{:s}_{:s}_it{:d}'.format(fig.canvas.get_window_title(), self.save_id, self.it)
    #         fig.savefig('{savedir:s}/{figname:s}.pdf'.format(savedir=self.save_dir, figname=figname),
    #                     transparent=True,
    #                     bbox_inches='tight', pad_inches=0.05,
    #                     )

    #         plt.close(fig)


    def to_xr(self):
        """
        """
        if self.run_count == 0:
            print('must run first')
            return
        
        lai = self.cnpy_descrip['lai']
        dlai = self.cnpy_descrip['dlai']
        z = self.cnpy_descrip['z']  # at lai values (layer interfaces)
        dz = np.diff(z)
        zm = z[:-1] + 0.5*dz  # midpts

        wl = self.cnpy_rad_state['wl']        
        dwl = self.cnpy_rad_state['dwl']

        psi = self.cnpy_rad_state['psi']
        mu  = self.cnpy_rad_state['mu']
        sza = np.rad2deg(psi)
        G = self.cnpy_rad_state['G']
        K_b = self.cnpy_rad_state['K_b']
        
        info = ''  # info passed in at model init?
        scheme_lname = self.scheme['long_name']
        scheme_sname = self.scheme['short_name']
        scheme_id = self.scheme_ID
        
        Idr  = self.cnpy_rad_state['I_dr']
        Idfd = self.cnpy_rad_state['I_df_d']
        Idfu = self.cnpy_rad_state['I_df_u']
        F    = self.cnpy_rad_state['F']
        crds = ['z', 'wl']
        crds2 = ['zm', 'wl']
        
        #> create dataset
        ln = 'long_name'
        z_units = dict(units='m')
        wl_units = dict(units='um')
        E_units = dict(units='W m^-2')  # E (energy flux) units
        SE_units = dict(units='W m^-2 um^-1')  # spectral E units
        pf_units = dict(units='umol photons m^-2 s^-1')  # photon flux units
        lai_units = dict(units='m^2 leaf m^-2')

        #> data vars for absorption (many)
        aI_suffs = ['', '_sl', '_sh', '_PAR']
        aI_lns   = ['absorbed irradiance', 'absorbed irradiance by sunlit leaves',
            'absorbed irradiance by shaded leaves', 
            'absorbed PAR irradiance',
            ]
        aI_data_vars = {}
        for s in aI_suffs:
            k = f'aI{s}'
            if 'sl' in k:
                by = ' by sunlit leaves'
            elif 'sh' in k:
                by = ' by shaded leaves'
            else:
                by = ''
            if 'PAR' in k:  # need to fix
                band = 'PAR '
                crds_ = ['zm']
            else:
                band = ''
                crds_ = crds2
            lni = f'absorbed {band}irradiance{by}'
            v = self.cnpy_rad_state[k]
            aI_data_vars[k] = (crds_, v, 
                {**E_units, ln: lni,}
                )
        print(aI_data_vars)
        

        dset = xr.Dataset(\
            coords={\
                    'z':  ('z',  z,  {**z_units, ln: 'height above ground'}),
                    'wl': ('wl', wl, {**wl_units, ln: 'wavelength'}),
                    'zm': ('zm', zm, {**z_units, ln: 'layer midpoint height'})
                    },
            data_vars={\
                'Idr':  (crds, Idr,  {**E_units, ln: 'direct beam solar irradiance'}),
                'Idfd': (crds, Idfd, {**E_units, ln: 'downward diffuse solar irradiance'}),
                'Idfu': (crds, Idfu, {**E_units, ln: 'upward diffuse solar irradiance'}),
                'F':    (crds, F,    {**E_units, ln: 'actinic flux'}),
                'dwl':  ('wl', dwl,  {**wl_units, ln: 'wavelength band width'}),
                # 'sdt':  ('time', sdt, {'long_name': 'datetime string'}),
                # 'sza':  ('time', sza, {'units': 'deg.', 'long_name': 'solar zenith angle'}),
                'lai':  ('z',  lai,  {**lai_units, ln: 'leaf area index (cumulative)'}),
                'dlai': ('zm', dlai, {**lai_units, ln: 'leaf area index in layer'}),
                **aI_data_vars
                },
            attrs={'save_id': self.save_id, 'info': info, 
                   'scheme_id': scheme_id, 
                   'scheme_long_name': scheme_lname, 
                   'scheme_short_name': scheme_sname,
                   'sza': sza, 
                   'psi': psi,
                   'mu' : mu,
                   'G' : G,
                   'K_b' : K_b
                   },
        )
        # TODO: add crt1d version?
        
        #> save dataset
        # ofname = '{:s}/{:s}.nc'.format(self.save_dir, self.save_id)
        # dset.to_netcdf(ofname)
        return dset
        
        


def run_cases(m0, cases):
    """For model m, run multiple cases.
    Vary one or more attributes and create combined dataset.

    m0 : model object
        base case to branch off of

    cases : list of dicts ?
        one dict for each case
        keys to change and values to change them to
    """
    case0 = cases[0]
    nparam = len(case0)  # number of things we are varying



        # #sdt = self.dts #['' for _ in inds]  # temporary..
        # try:
        #     sdt = [pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S') for x in self.dts]  # datetime strings from array/list of datetimes
        #     time = self.dts
        # except:  # e.g., of dts=[] or not convertable
        #     sdt = ['' for _ in inds]
        #     time = inds
        # if sdt == []:
        #     sdt = ['' for _ in inds]  # temporary fix since dts=[] not caught
        #     time = inds


# --------------------------------------------------------------------------------------------------
#




# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    plt.close('all')

    # m = model()
    # m.run()

