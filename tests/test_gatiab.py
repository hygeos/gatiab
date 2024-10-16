#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import xarray as xr
import glob
import os
ROOTPATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.insert(0, ROOTPATH)

from gatiab import ckdmip2od, Gatiab
from pathlib import Path
import logging

# *********************************** logging ***********************************
# Create log file
Path(os.path.join(ROOTPATH, "tests/logs/")).mkdir(parents=True, exist_ok=True)

# Create a named logger
logger = logging.getLogger('test_gatiab')
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# Set the formatter for the console handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
datefmt='%m/%d/%Y %I:%M:%S%p')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Create a file handler
file_handler = logging.FileHandler(ROOTPATH + "/tests/logs/gatiab.log", mode='w')
file_handler.setLevel(logging.INFO)

# Set the formatter for the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
# *******************************************************************************

GAS_LIST = ['O3']

O3_RES = {'B00':1.000,  # 400.3
          'B01':1.000,  # 411.8
          'B02':0.996,  # 443.0
          'B03':0.976,  # 490.5
          'B04':0.952,  # 510.5
          'B05':0.881,  # 560.5
          'B06':0.879,  # 620.4
          'B07':0.942,  # 665.3
          'B08':0.953,  # 674.0
          'B09':0.959,  # 681.6
          'B10':0.978,  # 709.1
          'B11':0.990,  # 754.2
          'B12':0.992,  # 761.7
          'B13':0.992,  # 764.8
          'B14':0.992,  # 767.9
          'B15':0.991,  # 779.3
          'B16':0.998,  # 865.4
          'B17':0.999,  # 884.3
          'B18':0.998,  # 899.3
          'B19':0.999,  # 939.0
          'B20':1.000,  # 1015.8
              }

H2O_RES = {'B00':0.999,  # 400.3
           'B01':1.000,  # 411.8
           'B02':0.995,  # 443.0
           'B03':0.998,  # 490.5
           'B04':0.986,  # 510.5
           'B05':0.999,  # 560.5
           'B06':0.999,  # 620.4
           'B07':0.993,  # 665.3
           'B08':1.000,  # 674.0
           'B09':0.996,  # 681.6
           'B10':0.896,  # 709.1
           'B11':0.999,  # 754.2
           'B12':1.000,  # 761.7
           'B13':1.000,  # 764.8
           'B14':1.000,  # 767.9
           'B15':0.993,  # 779.3
           'B16':0.993,  # 865.4
           'B17':0.970,  # 884.3
           'B18':0.495,  # 899.3
           'B19':0.050,  # 939.0
           'B20':0.964,  # 1015.8
              }

@pytest.fixture(scope='module')
def get_rsrf_data():
    '''
    DGet the S3A OLCI spectral response data
    '''

    rsrf_files = sorted(glob.glob(ROOTPATH + "/tests/S3A_OLCI_rsrf/*.nc"))
    nbands = len(rsrf_files)
    rsrf = []
    srf_wvl = []
    for i in range (0, nbands):
        with xr.open_dataset(rsrf_files[i]) as ds:
            rsrf.append(ds['rsrf'].values)
            srf_wvl.append(ds['wvl'].values)

    return rsrf, srf_wvl

@pytest.mark.parametrize('gas', GAS_LIST)
def test_gas(request, gas, get_rsrf_data):
    # === Get the S3A OLCI spectral response data
 
    rsrf, srf_wvl = get_rsrf_data
    
    ds = ckdmip2od(gas=gas,
                   dir_ckdmip=request.config.getoption("--dir-ckdmip"),
                   dir_atm=request.config.getoption("--dir-atm"),
                   wvn_min=9550., wvn_max=26000.)
    
    gt = Gatiab(ds)
    gt.print_gas_content()

    if gas == 'H2O': gas_content = np.array([3., 3.5, 4.])
    else: gas_content = np.array([250., 300., 350.])
    air_mass = np.array([3., 4., 5.])
    p0 = gt.od['P_hl'][-2:].data*1e-2
    p0 = p0[::-1]
    id0 = 0
    id1 = 21
    srf_wvl_r = srf_wvl[id0:id1]
    rsrf_r = rsrf[id0:id1]
    trans = gt.calc(gas_content,air_mass, p0, srf_wvl_r, rsrf_r)
    wvl_central_r = trans['lambda'].values

    if gas == 'H2O' : U = xr.DataArray(np.array([3.5]), dims='U')
    else: U = xr.DataArray(np.array([300.]), dims='U')
    M = xr.DataArray(np.array([4.]), dims='M')
    ip = 0
    nbands = len(wvl_central_r)
    print(nbands)
    trans_reduced = trans.interp(**{'U':U,'M':M})
    for iband in range (0, nbands):
        bandname = 'B' + str(iband).zfill(2)
        t_gas_ib = np.round(trans_reduced['trans'][iband, 0, 0, ip].data,3)
        iwvlc = np.round(wvl_central_r[iband], 2)

        if gas == 'H2O':
            logger.info(f"{gas} - {iwvlc} nm - T_ref={H2O_RES[bandname]:.3f} - T_calc={t_gas_ib  :.3f}")
            assert (t_gas_ib == H2O_RES[bandname]), f"Problem with {bandname}({iwvlc}) " + \
                f"get {t_gas_ib} instead of {H2O_RES[bandname]}"
        else:
            logger.info(f"{gas} - {iwvlc} nm - T_ref={O3_RES[bandname]:.3f} - T_calc={t_gas_ib  :.3f}")
            assert (t_gas_ib == O3_RES[bandname]), f"Problem with {bandname}({iwvlc}) " + \
                f"get {t_gas_ib} instead of {O3_RES[bandname]}"
            
        print("T(", np.round(wvl_central_r[iband], 2), " nm) =",
            np.round(trans_reduced['trans'][iband, 0, 0, ip].data,3))



    