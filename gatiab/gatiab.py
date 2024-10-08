#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import constants
from luts.luts import read_mlut, MLUT, LUT, Idx
import glob
import xarray as xr
from tqdm import tqdm
from scipy.interpolate import make_interp_spline, interp1d
from scipy.integrate import simpson
from datetime import datetime
from sys import exit

AccelDueToGravity  = 9.80665 # m s-2
MolarMassAir       = 28.970  # g mol-1 dry air

molar_mass={ 'h2o':18.0152833,
             'co2': 44.011,
             'o3': 47.9982,
             'n2o': 44.013,
             'co': 28.0101,
             'ch4':16.043,
             'o2': 31.9988,
             'cfc11': 137.3686,
             'cfc12': 120.914,
             'hcfc22': 86.469,
             'ccl4':153.823,
             'no2': 46.0055}

def get_zatm(P_hl, T_fl, M_air, g, h2o_mole_frac_fl=None, P_fl = None, M_h2o=None, method='barometric'):
    """
    Description : Compute the zatm knowing the pressure and temperature variability
    half level -> at layer altitude
    full level -> mean between 2 layers

    P_hl : half level pressure
    T_fl : full level temperature
    M_air : dry air molar mass in Kg mol-1
    g : gravity constant in m s-2
    h2o_mole_frac_fl : full level h2o mole fraction (needed if hypsometric method is chosen)
    P_fl : full level pressure (needed if hypsometric method is chosen)
    M_h2o : h2o molar mass in Kg mol-1
    method : 'barometric' or 'hypsometric'

    return the z_atm grid profil in Km 
    """

    if method != 'barometric' and method != 'hypsometric':
        raise NameError("Unknown method! Please choose between: 'barometric' or 'hypsometric")

    if method == 'hypsometric':
        epsilon = M_h2o/M_air
        e = P_fl * h2o_mole_frac_fl

    z = 0.
    zgrid = [0.]
    nlvl = len(T_fl)
    Rs = constants.R / M_air

    for itemp in range (0, nlvl):
        id1= -(itemp+1)
        id2 = -(itemp+2)
        T_mean = T_fl[id1]
        if method == 'barometric':
            dz = ((Rs*T_mean)/g) * (np.log(P_hl[id1]/P_hl[id2]))
            # dz = (-1/(M_air*g)) * (constants.R*T_mean) * (np.log(P_hl[id2]/P_hl[id1]))
        if method == 'hypsometric':
            Tv = T_mean / (1 - (e[id1]/P_fl[id1])*(1-epsilon)) #virtual temperature
            dz = ((Rs*Tv)/g) * (np.log(P_hl[id1]/P_hl[id2]))
        z += dz
        zgrid.append(z)
    zgrid=np.array(zgrid)[::-1]*1e-3
    return zgrid # return z_atm in km

def find_layer_index(dz, z_final):
    ilayer = 0
    nz_idea = len(dz)
    while( np.sum(dz[nz_idea-ilayer:]) < z_final ):
        ilayer+=int(1)
        if (ilayer >= nz_idea):
            ilayer = None
            return ilayer
    return ilayer-1


def diff1(A, axis=0, samesize=True):
    if samesize:
        B = np.zeros_like(A)
        key = [slice(None)]*A.ndim
        key[axis] = slice(1, None, None)
        B[tuple(key)] = np.diff(A, axis=axis)[:]
        return B
    else:
        return np.diff(A, axis=axis)
    

def check_input_ckdmip2od(gas, wvn_min, wvn_max, ckdmip_files):

    gas_ok = ['H2O', 'CO2', 'O2', 'O3', 'N2O', 'N2', 'CH4']

    if gas not in gas_ok:
        raise NameError("the gas '", gas, "' is not accepted! Choose between: 'H2O', 'O3', 'CO2', 'O2'")

    if (wvn_min < 250 or wvn_max >50000):
        raise NameError("Wavenumber range is from 250 to 50000 cm-1. Choose wvn_min and/or wvn_max between this interval!")

    if (wvn_min > wvn_max):
        raise NameError("wvn_min must be smaller than wvn_max!")
    
    nfiles = len(ckdmip_files)
    if gas == 'H2O' and nfiles != 12 :
        raise NameError(f"The number of {gas} files must be equal to 12!")
    
    if gas != 'H2O' and nfiles != 1 :
        raise NameError(f"The number of {gas} files must be equal to 1!")
    

def ckdmip2od(gas, dir_ckdmip, atm='afglus', wvn_min = 2499.99, wvn_max=50000, chunk=500,
              save=False, dir_save='./'):
    """
    Description : Use ckdmip shortwave idealized look-up tables to generate optical depth for a given atm

    gas              : choose between -> 'H2O', 'CO2', 'O2', 'O3', 'N2O', 'N2', 'CH4'
    dir_ckdmip       : directory where are located ckdmip look-up tables
    atm              : choose between -> 'afglus', 'afglt', 'afglms', 'afglmw', 'afglss', 'afglsw'
    wvn_min, wvn_max : wavenumber min and max interval values
    chunk            : number of wavenumber considered at each iteration (wavenumber dim is splited during interpolation)
    save             : if not False, output directory where to save the optical depth generated lut
    """

    files_name = "ckdmip_idealized_sw_spectra_" + gas.lower() + "_const*.h5"
    ckdmip_files = sorted(glob.glob(dir_ckdmip + files_name))

    check_input_ckdmip2od(gas, wvn_min, wvn_max, ckdmip_files)

    afgl_pro = read_mlut('./atmospheres/'+atm+'.nc')

    # Declaration of variables
    ds_gas_imf = xr.open_dataset(ckdmip_files[0])
    wavn = ds_gas_imf.wavenumber[np.logical_and(ds_gas_imf.wavenumber>=wvn_min, ds_gas_imf.wavenumber<=wvn_max)].values
    nwavn = len(wavn)
    P_fl = ds_gas_imf['pressure_fl'].values[0,:]
    nP = len(P_fl)
    T_fl = ds_gas_imf['temperature_fl'].values[:,:]
    T_fl_unique = np.unique(T_fl)
    nT_unique = len(T_fl_unique)
    nc = len(ds_gas_imf['pressure_fl'].values[:,0]) 
    nlvl = len(ds_gas_imf['level'].values)
    P_hl = ds_gas_imf['pressure_hl'].values[0,:]
    nmf = len(ckdmip_files)
    M_air = MolarMassAir*1e-3

    z_afgl = afgl_pro.axes['z_atm'][:] # in Km
    nzafgl = len(z_afgl)
    P_afgl = afgl_pro['P'][:] * 1e2 # in Pa

    # First step: Load optical depths and save it in numpy array
    mole_fraction = []
    OD_gas = np.zeros((nc,nlvl,nwavn,nmf), dtype=np.float32)
    with tqdm(total=nmf) as bar_mf:
        for imf in range (0,nmf):
            bar_mf.set_description("Load ckdmip optical depth...")
            if imf > 0 : ds_gas_imf = xr.open_dataset(ckdmip_files[imf])
            ds_gas_imf = ds_gas_imf.sel(wavenumber = wavn)
            OD_gas[:,:,:,imf] = ds_gas_imf['optical_depth'].values
            mole_fraction.append(ds_gas_imf['mole_fraction_hl'].values[0,0])
            bar_mf.update(1)
    mole_fraction = np.array(mole_fraction)

    # Second step: Conversion and interpolations
    # Here we split wavenumber into several pieces (chunk) to save memory and optimize calculations
    C_ext_gas = np.zeros((nzafgl-1,nwavn), dtype=np.float64)

    T_int = interp1d(afgl_pro['P'][:], afgl_pro['T'][:], bounds_error=False, fill_value='extrapolate')
    P_afgl_fl = P_afgl[1:] - 0.5*np.diff(P_afgl)
    T_afgl_fl = T_int(P_afgl_fl*1e-2)
    mole_fraction_afgl = LUT( ((afgl_pro[gas][:]*1e6* constants.Boltzmann*afgl_pro['T'][:])/(afgl_pro['P'][:]*1e2)), axes=[z_afgl], names=["z_atm"])
    z_afgl_fl = z_afgl[1:] + 0.5*np.abs(np.diff(z_afgl))
    mole_fraction_afgl_fl = mole_fraction_afgl[Idx(z_afgl_fl)]

    nwc = chunk
    nwc_ini = 0
    nwc_end = nwc_ini
    reste = nwavn

    with tqdm(total=int(reste/nwc) +1) as bar_wavn:
        while(reste > 0):
            bar_wavn.set_description("Convert to extinction coeff in m2 Kg-1 and interpolate...")
            reste = reste - nwc
            if reste < 0: nwc_end += nwc - np.abs(reste)
            else: nwc_end += nwc

            # print(nwc_ini, nwc_end)

            C_ext_iw = np.zeros((nP, nT_unique, nwc_end-nwc_ini,nmf), dtype=np.float64)
            for imf in range (0, nmf):
                OD_gas_imf_iw = OD_gas[:,:,nwc_ini:nwc_end,imf]
                # extinction coefficient in m2 kg-1
                C_ext_iw_imf_bis = np.zeros((nc, nlvl,nwc_end-nwc_ini), dtype=np.float64)
                for ilvl in range (0, nlvl):
                    for ic in range (0, nc):
                        C_ext_iw_imf_bis[ic,ilvl,:] = (AccelDueToGravity * M_air * OD_gas_imf_iw[ic,ilvl,:].astype(np.float64)) / \
                            ( mole_fraction[imf] * (P_hl[ilvl+1] - P_hl[ilvl] ) )

                for ip in range (0, nP):
                        C_ext_iw[ip,:,:,imf] = make_interp_spline(T_fl[:,ip], C_ext_iw_imf_bis[:,ip,:], k=1, axis=0)(T_fl_unique)

            lut_C_ext = LUT(C_ext_iw, axes=[P_fl,T_fl_unique,wavn[nwc_ini:nwc_end],mole_fraction],
                            names=['P', 'T', 'wavenumber','mole_fraction'])

            for iz in range (0,nzafgl-1):
                if (P_afgl_fl[iz] < np.max(P_fl)) and (P_afgl_fl[iz] > np.min(P_fl)):
                    if (gas == 'H2O'): C_ext_gas[iz,nwc_ini:nwc_end] = lut_C_ext[Idx(P_afgl_fl[iz]), Idx(T_afgl_fl[iz]), :, Idx(mole_fraction_afgl_fl[iz])]
                    else : C_ext_gas[iz,nwc_ini:nwc_end] = lut_C_ext[Idx(P_afgl_fl[iz]), Idx(T_afgl_fl[iz]), :, 0]
            nwc_ini = nwc_end
            bar_wavn.update(1)

    # Third step: reconversion to optical depth
    tau_gas = np.zeros((nzafgl-1,nwavn), dtype=np.float64)
    P_afgl_hl = (afgl_pro['P'][:]*1e2).astype(np.float64)

    print("reconvert to optical depth...")
    with tqdm(total=int(nzafgl-1)) as bar_lvl:
        for ilvl in range (0, nzafgl-1):
            tau_gas[ilvl] = (C_ext_gas[ilvl,:] * mole_fraction_afgl_fl[ilvl].astype(np.float64) * (P_afgl_hl[ilvl+1] - P_afgl_hl[ilvl])) / (AccelDueToGravity * M_air)
            bar_lvl.update(1)
    print("reconverted to optical depth.")

    # Fourth step: Create final LUT and optionnaly save
    mlut_idea = MLUT()
    mlut_idea.add_axis('level', (np.arange(nzafgl-1)+1).astype(np.int32))
    mlut_idea.add_axis('half_level', (np.arange(nzafgl)+1).astype(np.int32))
    mlut_idea.add_axis('wavenumber', wavn)
    mlut_idea.add_dataset('P_fl', P_afgl_fl.astype(np.float32), axnames=['level'], attrs={"Description":"Full level pressure in Pa."})
    mlut_idea.add_dataset('T_fl', T_afgl_fl.astype(np.float32), axnames=['level'], attrs={"Description":"Full level temperature in Pa."})
    mlut_idea.add_dataset('P_hl', (afgl_pro['P'][:]*1e2).astype(np.float32), axnames=['half_level'], attrs={"Description":"half level pressure in K."})
    mlut_idea.add_dataset('T_hl', (afgl_pro['T'][:]).astype(np.float32), axnames=['half_level'], attrs={"Description":"half level temperature in K."})
    mlut_idea.add_dataset('mole_fraction_fl', mole_fraction_afgl_fl, axnames=['level'])
    mlut_idea.add_dataset('mole_fraction_hl', mole_fraction_afgl[:], axnames=['half_level'])
    mlut_idea.add_dataset('optical_depth', tau_gas.astype(np.float32), axnames=['level', 'wavenumber'])
    mlut_idea.add_dataset('z_atm', z_afgl, axnames=['half_level'])
    date = datetime.now().strftime("%Y-%m-%d")
    mlut_idea.set_attr('name', 'Spectral optical depth profiles of ' + gas)
    mlut_idea.set_attr('experiment', 'AFGLUS U.S. standard based on Idealized CKDMIP interpolation')
    mlut_idea.set_attr('date', date)
    mlut_idea.set_attr('source', 'Created by HYGEOS, using CKDMIP data')
    if save :
        save_filename = f"od_{gas}_{atm}_ckdmip_idealized_solar_spectra.nc"
        mlut_idea.to_xarray().to_netcdf(dir_save + save_filename)

    return mlut_idea


def od2trans(od, gas_content, air_mass, p0, srf_wvl, rsrf, save=False):
    """
    od          : path to optical depth LUT (created using ckdmip2od function) or MLUT object
    gas_content : numpy array with gas content, in dopson for O3 and in g m-2 for other gas
    air_mass    : numpy array with ratio of slant path optical depth and vertical optical depth
    p0          : numpy array with several ground pressure (p0) values in hPa
    srf_wvl     : SRF wavelengths numpy arrays into an iband list
    rsrf        : SRF values numpy arrays into an iband list
    """


    gas = od.attrs['name'].split(' ')[-1]

    n_U = len(gas_content)
    nbands = len(srf_wvl)
    n_M = len(air_mass)
    n_p0 = len(p0)

    p_gas = od['P_hl'][:] * od['mole_fraction_hl'][:] 
    dens_gas = (p_gas) / (constants.Boltzmann*od['T_hl'][:]) * 1e-6
    dens_gas_lut = LUT(dens_gas, axes=[od['P_hl'][:]], names=['P_hl'])
    half_level_lut = LUT(od.axes['half_level'][:], axes=[od['P_hl'][:]], names=['P_hl'])

    trans_gas = np.zeros((nbands,n_U,n_M,n_p0), dtype=np.float64)

    for iband in range (0, nbands):
        wvn_bi = 1/srf_wvl[iband]*1e7
        #nwl_bi = len(wvn_bi)
        lut_gas_bi = od.sub({'wavenumber':Idx(lambda x: np.logical_and(x >=np.min(wvn_bi)-1, x <=np.max(wvn_bi)+1))})
        isrf_int =interp1d(wvn_bi, rsrf[iband], bounds_error=False, fill_value=0.)(lut_gas_bi.axes['wavenumber'])
        for iU in range (0, n_U):
            for ip in range (0, n_p0):
                p_ip = np.sort(np.concatenate((od['P_hl'][(od['P_hl'][:]<p0[ip]*1e2)], np.array([p0[ip]*1e2], dtype='float32') )))
                half_level_ip = half_level_lut[Idx(p_ip)]
                dens_gas_ip = dens_gas_lut[Idx(p_ip)].astype(np.float64)
                z_atm_ib = od['z_atm'][Idx(half_level_ip)]
                
                if gas == 'O3':
                    dens_gas_iUp =  dens_gas_ip * (2.6867e16 * gas_content[iU] / (simpson(y=dens_gas_ip, x=-od['z_atm'][Idx(half_level_ip)]) * 1e5))
                else:
                    dens_gas_iUp =  dens_gas_ip * (gas_content[iU]/ molar_mass[gas.lower()] * constants.Avogadro / (simpson(y=dens_gas_ip, x=-z_atm_ib) * 1e5))
                
                # convert to abs coeff
                ot = lut_gas_bi['optical_depth'].data.astype(np.float64)
                ot = np.append(np.zeros((1,len(lut_gas_bi.axes['wavenumber']))), ot, axis=0).astype(np.float64)
                ot = np.swapaxes(ot, 0,1)
                dz = diff1(od['z_atm'][:]).astype(np.float64)
                k = abs(ot/dz)
                k[np.isnan(k)] = 0
                sl = slice(None,None,1)
                k = k[:,sl]
                C_abs = np.swapaxes(k, 0,1)
                C_abs_lut = LUT(C_abs,axes=[od['z_atm'][:], lut_gas_bi.axes['wavenumber'][:]], names=['z_atm', 'wavenumber'])
                # reconvert to optical depth
                C_abs_ib = C_abs_lut[Idx(z_atm_ib),:]
                dz_ib = np.abs(diff1(z_atm_ib))
                od_ib = dz_ib[:,None] * C_abs_ib

                fac_iUp = dens_gas_iUp[:]/dens_gas_ip[:]
                tau_zw = fac_iUp[:,None] * od_ib[:]
                tau_zw[tau_zw<0] = 0.
                tau_w = np.sum(tau_zw,axis=0)
                
                # Consider air_mass
                for iM in range (0, n_M):
                    tau_wM = tau_w * air_mass[iM]
                    trans_wM = np.exp(-tau_wM)
                    # simpson to consider the case where dw is varying, this is the case at 625 and 1941.75 nm
                    num = simpson(y=trans_wM[isrf_int>0]*isrf_int[isrf_int>0], x=lut_gas_bi.axes['wavenumber'][isrf_int>0])
                    den = simpson(y=isrf_int[isrf_int>0], x=lut_gas_bi.axes['wavenumber'][isrf_int>0])
                    trans = num/den
                    trans_gas[iband,iU,iM,ip] = trans

    bands = []
    for iband in range (0, nbands):
        bands.append(round(simpson(srf_wvl[iband]*rsrf[iband], x=srf_wvl[iband])/simpson(rsrf[iband], x=srf_wvl[iband]), 1))
    bands = np.array(bands, dtype=np.float64)

    mlut = MLUT()
    mlut.add_axis('bands', bands)
    mlut.add_axis('gas_content', gas_content)
    mlut.add_axis('air_mass', air_mass)
    mlut.add_axis('p0', p0)
    mlut.add_dataset('transmission', trans_gas.astype(np.float32), axnames=['bands', 'gas_content', 'air_mass', 'p0'])

    return mlut