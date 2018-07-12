#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:03:42 2018


@author: zmoon
"""

from __future__ import print_function

#from tqdm import tqdm

import leaf_rad_common_v2 as lrc
reload(lrc)


#%% test load default canopy descrip

#fname = './default_canopy_descrip.csv'

#print(lrc.load_canopy_descrip(fname))


#%% test each scheme

#scheme_ids = ['bl', '2s', '4s', 'zq']
scheme_ids = ['bf']

for sid in scheme_ids:

    m = lrc.model(scheme_ID=sid, mi={}, psi=0., nlayers=60,
                  save_dir= './out/data', save_ID=sid+'_test', 
                  save_figs=False, write_output=True)
    
    m.run()  # could put the save args in here instead?











