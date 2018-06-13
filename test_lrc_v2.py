#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:03:42 2018


@author: zmoon
"""

import leaf_rad_common_v2 as lrc


#%% test load default canopy descrip

fname = './default_canopy_descrip.csv'

print lrc.load_canopy_descrip(fname)
