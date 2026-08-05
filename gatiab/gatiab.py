"""
Core module of the gatiab package.

This module generates gas optical depth look-up tables from the
CKDMIP shortwave idealized spectra for a given AFGL atmosphere,
and computes the gaseous transmissions at instrument averaged
bands using the spectral response functions of the instrument.

Key Functions and Classes
-------------------------
ckdmip2od
    Generate the optical depth look-up table of a gas for a given
    AFGL atmosphere, from the CKDMIP idealized look-up tables.
Gatiab
    Compute the gaseous transmissions at instrument averaged
    bands from an optical depth look-up table, as function of the
    gas content, the airmass and the ground pressure. The
    columnar gas content of the look-up table can also be
    rescaled (update_gas_content).
vec_float_indexing
    Vectorized multilinear interpolation indexing with float
    scalars/1d arrays, based on the Idx method of the luts
    package.
get_zatm
    Altitude profile from the pressure and temperature
    variability, with the barometric or hypsometric method.
get_bands
    SRF-weighted central wavelengths of the instrument bands.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import constants
from scipy.integrate import simpson
from scipy.interpolate import make_interp_spline, interp1d
from tqdm import tqdm

from gatiab.constants import (
    ACCEL_DUE_TO_GRAVITY,
    MOLAR_MASS,
    MOLAR_MASS_AIR,
    VERSION,
)


def get_binary_mat(ndim: int) -> NDArray[np.int32]:
    """Compute matrix filled with binary values.

    Row ``i`` holds the little-endian binary digits of ``i``; the
    returned array has shape ``(ndim**2, ndim)``.
    """
    indices = np.arange(ndim**2)
    return ((indices[:, np.newaxis] >> np.arange(ndim)) & 1).astype(np.int32)


def vec_float_indexing(
    data: NDArray, keys: Sequence[float | NDArray | slice]
) -> NDArray:
    """Perform vectorized indexing with float scalars/1d arrays.

    - Method based on Idx of luts module, see:
      https://github.com/hygeos/luts
    - Non slice objects on keys list must be all scalars or all 1d
      arrays with same size

    Parameters
    ----------
    data : np.ndarray
        The array to be indexed
    keys : list
        List with the scalars or 1darrays indices

    Returns
    -------
    R : np.ndarray
        The indexed array

    Examples
    --------
    >>> m1 = np.arange(6).reshape(3,2)
    >>> m1
    array([[0, 1],
          [2, 3],
          [4, 5]])
    >>> m2 = vec_float_indexing(m1, [np.array([0.8, 1.1, 1.5]), slice(None)])
    >>> m2
    array([[1.6, 2.6],
           [2.2, 3.2],
           [3. , 4. ]])
    >>> m3 = vec_float_indexing(m1, [0.8, 0.])
    >>> m3
    1.6
    """

    nkeys = len(keys)

    # Find interpolation shape
    dims_array = None
    index0 = []
    for i in range(nkeys):
        k = keys[i]
        if isinstance(k, np.ndarray) and (k.ndim > 0):
            index0.append(np.zeros_like(k, dtype='int'))
            if dims_array is None:
                dims_array = k.shape
        elif isinstance(k, slice):
            index0.append(k)
        else:  # scalar
            index0.append(0)
    shp_int = np.zeros(1).reshape([1]*nkeys)[tuple(index0)].shape

    isinterp = [not isinstance(k, slice) for k in keys]
    interp_keys_idx = [i for i, e in enumerate(isinterp) if e]
    ninterp = len(interp_keys_idx)

    res = 0
    keys_bis = list(keys)
    for it in range(0, 2**ninterp):
        fac = 1
        for ik in range(0, ninterp):
            dim_idx = interp_keys_idx[ik]
            # Bit ik of it (position among the interpolated keys,
            # like LUT), not the raw dimension position, so that
            # slices interleaved with interpolated keys do not
            # shift the corner selection
            bit = (it >> ik) & 1
            key = cast('float | NDArray', keys[dim_idx])
            # Clamp inf to N-2 max (like LUT), so inf+1 never
            # exceeds N-1
            inf = np.clip(
                np.floor(key).astype(np.int32),
                0, data.shape[dim_idx] - 2
            )
            idx = inf + bit
            x = key - inf  # weight
            if bit:
                fac *= x
            else:
                fac *= (1 - x)
            keys_bis[dim_idx] = idx
        if isinstance(fac, np.ndarray):
            res = res + fac.reshape(shp_int) * data[tuple(keys_bis)]
        else:
            res = res + fac * data[tuple(keys_bis)]

    return cast('NDArray', res)


def get_zatm(
    P_hl: NDArray,
    T_fl: NDArray,
    M_air: float,
    g: float,
    h2o_mole_frac_fl: NDArray | None = None,
    P_fl: NDArray | None = None,
    M_h2o: float | None = None,
    method: str = 'barometric',
) -> NDArray:
    """Get z profil knowing the pressure and temperature
    variability.

    - half level -> at layer altitude
    - full level -> mean between 2 layers

    Parameters
    ----------
    P_hl : np.ndarray
        Half level pressure
    T_fl : np.ndarray
        Full level temperature
    M_air : float
        Dry air molar mass in Kg mol-1
    g : float
        Gravity constant in m s-2
    h2o_mole_frac_fl : np.ndarray, optional
        Full level h2o mole fraction (needed if hypsometric method
        is chosen)
    P_fl : np.ndarray, optional
        Full level pressure (needed if hypsometric method is
        chosen)
    M_h2o : float, optional
        H2O molar mass in Kg mol-1
    method : str
        Choose between -> 'barometric' or 'hypsometric'

    Returns
    -------
    Z : np.ndarray
        z_atm grid profil in Km
    """

    if method != 'barometric' and method != 'hypsometric':
        raise NameError(
            "Unknown method! Please choose between: 'barometric' or "
            "'hypsometric"
        )

    if method == 'hypsometric':
        epsilon = M_h2o/M_air  # pyright: ignore[reportOptionalOperand]
        e = P_fl * h2o_mole_frac_fl  # pyright: ignore[reportOptionalOperand]

    z = 0.
    zgrid = [0.]
    nlvl = len(T_fl)
    Rs = constants.R / M_air

    for itemp in range(0, nlvl):
        id1 = -(itemp+1)
        id2 = -(itemp+2)
        T_mean = T_fl[id1]
        if method == 'barometric':
            dz = ((Rs*T_mean)/g) * (np.log(P_hl[id1]/P_hl[id2]))
            # dz = (-1/(M_air*g)) * (constants.R*T_mean) \
            #     * (np.log(P_hl[id2]/P_hl[id1]))
        else:  # hypsometric
            # virtual temperature
            Tv = T_mean / (
                1 - (e[id1]/P_fl[id1])*(1-epsilon)  # pyright: ignore
            )
            dz = ((Rs*Tv)/g) * (np.log(P_hl[id1]/P_hl[id2]))
        z += dz
        zgrid.append(z)
    zgrid = np.array(zgrid)[::-1]*1e-3
    return zgrid  # return z_atm in km


def find_layer_index(dz: NDArray, z_final: float) -> int | None:
    """Find the layer index reaching a cumulated thickness.

    Return the index such that the cumulated thickness of the last
    layers of ``dz`` reaches ``z_final``, or None if the total
    thickness is insufficient.
    """
    ilayer = 0
    nz_idea = len(dz)
    while np.sum(dz[nz_idea-ilayer:]) < z_final:
        ilayer += int(1)
        if ilayer >= nz_idea:
            ilayer = None
            return ilayer
    return ilayer-1


def diff1(A: NDArray, axis: int = 0, samesize: bool = True) -> NDArray:
    """First-order difference along an axis.

    Like ``np.diff`` but, when ``samesize`` is True, the output
    keeps the input shape with zeros at the first position along
    ``axis``.
    """
    if samesize:
        B = np.zeros_like(A)
        key = [slice(None)]*A.ndim
        key[axis] = slice(1, None, None)
        B[tuple(key)] = np.diff(A, axis=axis)[:]
        return B
    else:
        return np.diff(A, axis=axis)


def get_bands(srf_wvl: list[NDArray], rsrf: list[NDArray]) -> NDArray:
    """Compute the SRF-weighted central wavelength of each band.

    Return the array of the central wavelengths (rounded to 0.1 nm)
    of the bands described by ``srf_wvl`` and ``rsrf``.
    """
    nbands = len(srf_wvl)
    bands = []
    for iband in range(0, nbands):
        bands.append(round(
            simpson(srf_wvl[iband]*rsrf[iband], x=srf_wvl[iband])
            / simpson(rsrf[iband], x=srf_wvl[iband]), 1
        ))
    bands = np.array(bands, dtype=np.float64)
    return bands


def check_input_ckdmip2od(
    gas: str, wvn_min: float, wvn_max: float, ckdmip_files: list[Path]
) -> None:
    """Validate the inputs of ckdmip2od.

    Raise NameError if the gas, the wavenumber interval or the
    number of ckdmip files is invalid.
    """
    gas_ok = ['H2O', 'CO2', 'O2', 'O3', 'N2O', 'N2', 'CH4']

    if gas not in gas_ok:
        raise NameError(
            "the gas '", gas,
            "' is not accepted! Choose between: 'H2O', 'O3', 'CO2', 'O2'"
        )

    if (wvn_min < 250 or wvn_max > 50000):
        raise NameError(
            "Wavenumber range is from 250 to 50000 cm-1. Choose "
            "wvn_min and/or wvn_max between this interval!"
        )

    if (wvn_min > wvn_max):
        raise NameError("wvn_min must be smaller than wvn_max!")

    nfiles = len(ckdmip_files)
    if gas == 'H2O' and nfiles != 12:
        raise NameError(f"The number of {gas} files must be equal to 12!")

    if gas != 'H2O' and nfiles != 1:
        raise NameError(f"The number of {gas} files must be equal to 1!")


def ckdmip2od(
    gas: str,
    dir_ckdmip: str | Path,
    dir_atm: str | Path,
    atm: str = 'afglus',
    wvn_min: float = 2499.99,
    wvn_max: float = 50000,
    chunk: int = 500,
    save: bool = False,
    dir_save: str | Path = './',
    float_indexing: str = 'fast',
) -> xr.Dataset:
    """Use ckdmip shortwave idealized look-up tables to generate
    optical depth for a given atm.

    Parameters
    ----------
    gas : str
        Choose between -> 'H2O', 'CO2', 'O2', 'O3', 'N2O', 'N2',
        'CH4'
    dir_ckdmip : str or Path
        Directory where are located ckdmip look-up tables
    dir_atm : str or Path
        Directory where are located the afgl atmosphere look-up
        tables (see README.md)
    atm : str, optional
        Choose between -> 'afglus', 'afglt', 'afglms', 'afglmw',
        'afglss', 'afglsw'
    wvn_min, wvn_max : float, optional
        Wavenumber min and max interval values
    chunk : int, optional
        Number of wavenumber considered at each iteration
        (wavenumber dim is splited during interpolation)
    save : bool, optional
        If True, save output in netcdf format
    dir_save : str or Path, optional
        Output directory where to save the optical depth generated
        lut
    float_indexing : str, optional
        Choose between -> 'fast' and 'xarray'

    Returns
    -------
    L : xr.Dataset
        Look-up table with the gas optical depth for the specified
        atmosphere
    """

    files_name = "ckdmip_idealized_sw_spectra_" + gas.lower() + "_const*.h5"
    ckdmip_files = sorted(Path(dir_ckdmip).glob(files_name))

    check_input_ckdmip2od(gas, wvn_min, wvn_max, ckdmip_files)

    if 'nc' not in atm.split('.'):
        filename = atm + '.nc'
    else:
        filename = atm
    file_path = Path.joinpath(Path(dir_atm), filename)
    afgl_pro = xr.open_dataset(file_path)

    # Declaration of variables
    ds_gas_imf = xr.open_dataset(ckdmip_files[0])
    ds_gas_imf.close()
    wavn = ds_gas_imf.wavenumber[np.logical_and(
        ds_gas_imf.wavenumber >= wvn_min,
        ds_gas_imf.wavenumber <= wvn_max,
    )].values
    nwavn = len(wavn)
    P_fl = ds_gas_imf['pressure_fl'].values[0, :]
    nP = len(P_fl)
    T_fl = ds_gas_imf['temperature_fl'].values[:, :]
    T_fl_unique = np.unique(T_fl)
    nT_unique = len(T_fl_unique)
    nc = len(ds_gas_imf['pressure_fl'].values[:, 0])
    nlvl = len(ds_gas_imf['level'].values)
    P_hl = ds_gas_imf['pressure_hl'].values[0, :]
    nmf = len(ckdmip_files)
    M_air = MOLAR_MASS_AIR*1e-3

    z_afgl_hl = afgl_pro['z_atm'].values[:]  # in Km
    n_afgl_hl = len(z_afgl_hl)
    P_afgl_hl = afgl_pro['P'].values[:] * 1e2  # in Pa
    T_afgl_hl = afgl_pro['T'].values[:]

    # First step: Load optical depths and save it in numpy array
    mole_fraction = []
    OD_gas = np.zeros((nc, nlvl, nwavn, nmf), dtype=np.float32)
    with tqdm(total=nmf) as bar_mf:
        for imf in range(0, nmf):
            bar_mf.set_description(
                "Load " + gas.lower() + " ckdmip optical depth..."
            )
            if imf > 0:
                ds_gas_imf = xr.open_dataset(ckdmip_files[imf])
                ds_gas_imf.close()
            ds_gas_imf = ds_gas_imf.sel(wavenumber=wavn)
            OD_gas[:, :, :, imf] = ds_gas_imf['optical_depth'].values
            mole_fraction.append(ds_gas_imf['mole_fraction_hl'].values[0, 0])
            bar_mf.update(1)
    mole_fraction = np.array(mole_fraction)

    # Second step: Conversion and interpolations
    # Here we split wavenumber into several pieces (chunk) to save
    # memory and optimize calculations
    C_ext_gas = np.zeros((n_afgl_hl-1, nwavn), dtype=np.float64)

    T_int = interp1d(
        P_afgl_hl, T_afgl_hl, bounds_error=False, fill_value='extrapolate'
    )
    P_afgl_fl = P_afgl_hl[1:] - 0.5*np.diff(P_afgl_hl)
    T_afgl_fl = T_int(P_afgl_fl)
    mole_fraction_afgl_hl = (
        afgl_pro[gas].values[:]*1e6 * constants.Boltzmann*T_afgl_hl
    ) / (P_afgl_hl)
    z_afgl_fl = z_afgl_hl[1:] + 0.5*np.abs(np.diff(z_afgl_hl))
    mole_fraction_afgl_fl = interp1d(
        z_afgl_hl, mole_fraction_afgl_hl
    )(z_afgl_fl)

    # Bellow needed for vectorized float indexing
    cond = np.logical_and(P_afgl_fl < np.max(P_fl), P_afgl_fl > np.min(P_fl))
    if float_indexing == 'fast':
        idf_pfl = interp1d(P_fl, np.arange(nP))(P_afgl_fl[cond])
        idf_tfl = interp1d(T_fl_unique, np.arange(nT_unique))(T_afgl_fl[cond])
        if (gas == 'H2O'):
            idf_mffl = interp1d(mole_fraction, np.arange(nmf))(
                mole_fraction_afgl_fl[cond]
            )

    nwc = chunk
    nwc_ini = 0
    nwc_end = nwc_ini
    reste = nwavn

    with tqdm(total=int(reste/nwc) + 1) as bar_wavn:
        while (reste > 0):
            bar_wavn.set_description(
                "Convert to extinction coeff in m2 Kg-1 and "
                "interpolate..."
            )
            reste = reste - nwc
            if reste < 0:
                nwc_end += nwc - np.abs(reste)
            else:
                nwc_end += nwc

            C_ext_iw = np.zeros(
                (nP, nT_unique, nwc_end-nwc_ini, nmf), dtype=np.float64
            )
            for imf in range(0, nmf):
                OD_gas_imf_iw = OD_gas[:, :, nwc_ini:nwc_end, imf]
                # extinction coefficient in m2 kg-1
                C_ext_iw_imf_bis = np.zeros(
                    (nc, nlvl, nwc_end-nwc_ini), dtype=np.float64
                )
                for ilvl in range(0, nlvl):
                    for ic in range(0, nc):
                        C_ext_iw_imf_bis[ic, ilvl, :] = (
                            ACCEL_DUE_TO_GRAVITY * M_air
                            * OD_gas_imf_iw[ic, ilvl, :].astype(np.float64)
                        ) / (
                            mole_fraction[imf] * (P_hl[ilvl+1] - P_hl[ilvl])
                        )

                for ip in range(0, nP):
                    C_ext_iw[ip, :, :, imf] = make_interp_spline(
                        T_fl[:, ip], C_ext_iw_imf_bis[:, ip, :], k=1, axis=0
                    )(T_fl_unique)

            # ****** Vectorized float indexing ******
            if float_indexing == 'fast':
                if (gas == 'H2O'):
                    keys = [idf_pfl, idf_tfl,  # pyright: ignore
                            slice(None), idf_mffl]  # pyright: ignore
                    C_ext_gas[cond, nwc_ini:nwc_end] = vec_float_indexing(
                        C_ext_iw, keys
                    )
                else:
                    keys = [idf_pfl, idf_tfl,  # pyright: ignore
                            slice(None)]
                    C_ext_gas[cond, nwc_ini:nwc_end] = vec_float_indexing(
                        C_ext_iw[:, :, :, 0], keys
                    )

            elif float_indexing == 'xarray':
                if (gas == 'H2O'):
                    lut_C_ext = xr.DataArray(
                        C_ext_iw,
                        dims=['P', 'T', 'wvn', 'mf'],
                        coords={
                            'P': P_fl,
                            'T': T_fl_unique,
                            'wvn': wavn[nwc_ini:nwc_end],
                            'mf': mole_fraction,
                        },
                    )
                    C_ext_gas[cond, nwc_ini:nwc_end] = lut_C_ext.interp(**{
                        'P': xr.DataArray(P_afgl_fl[cond], dims=['z_atm']),
                        'T': xr.DataArray(T_afgl_fl[cond], dims=['z_atm']),
                        'mf': xr.DataArray(
                            mole_fraction_afgl_fl[cond], dims=['z_atm']
                        ),
                    })
                else:
                    lut_C_ext = xr.DataArray(
                        C_ext_iw[:, :, :, 0],
                        dims=['P', 'T', 'wvn'],
                        coords={
                            'P': P_fl,
                            'T': T_fl_unique,
                            'wvn': wavn[nwc_ini:nwc_end],
                        },
                    )
                    C_ext_gas[cond, nwc_ini:nwc_end] = lut_C_ext.interp(**{
                        'P': xr.DataArray(P_afgl_fl[cond], dims=['z_atm']),
                        'T': xr.DataArray(T_afgl_fl[cond], dims=['z_atm']),
                    })
            else:
                raise NameError(
                    "Unknown float_indexing value. Choose between 'fast' "
                    "or 'xarray'!"
                )
            # ***************************************
            nwc_ini = nwc_end
            bar_wavn.update(1)

    # Third step: reconversion to optical depth
    tau_gas = np.zeros((n_afgl_hl-1, nwavn), dtype=np.float64)
    print("reconvert to optical depth...")
    with tqdm(total=int(n_afgl_hl-1)) as bar_lvl:
        for ilvl in range(0, n_afgl_hl-1):
            tau_gas[ilvl] = (
                C_ext_gas[ilvl, :]
                * mole_fraction_afgl_fl[ilvl].astype(np.float64)
                * (P_afgl_hl[ilvl+1] - P_afgl_hl[ilvl])
            ) / (ACCEL_DUE_TO_GRAVITY * M_air)
            bar_lvl.update(1)
    print("reconverted to optical depth.")

    # Fourth step: Create final LUT and optionnaly save
    ds = xr.Dataset(coords={
        'level': (np.arange(n_afgl_hl-1)+1).astype(np.int32),
        'half_level': (np.arange(n_afgl_hl)+1).astype(np.int32),
        'wavenumber': wavn,
    })
    ds['P_fl'] = xr.DataArray(
        P_afgl_fl.astype(np.float32), dims=['level'],
        attrs={'units': 'Pa', 'description': 'Full level pressure'},
    )
    ds['T_fl'] = xr.DataArray(
        T_afgl_fl.astype(np.float32), dims=['level'],
        attrs={'units': 'K', 'description': 'Full level temperature'},
    )
    ds['P_hl'] = xr.DataArray(
        P_afgl_hl.astype(np.float32), dims=['half_level'],
        attrs={'units': 'Pa', 'description': 'half level pressure'},
    )
    ds['T_hl'] = xr.DataArray(
        T_afgl_hl.astype(np.float32), dims=['half_level'],
        attrs={'units': 'K', 'description': 'half level temperature'},
    )
    ds['mole_fraction_fl'] = xr.DataArray(
        mole_fraction_afgl_fl, dims=['level'],
        attrs={'units': 'None', 'description': 'full level mole fraction'},
    )
    ds['mole_fraction_hl'] = xr.DataArray(
        mole_fraction_afgl_hl, dims=['half_level'],
        attrs={'units': 'None', 'description': 'half level mole fraction'},
    )
    ds['optical_depth'] = xr.DataArray(
        tau_gas.astype(np.float32), dims=['level', 'wavenumber'],
        attrs={'units': 'None', 'description': 'optical depth'},
    )
    ds['z_atm'] = xr.DataArray(
        z_afgl_hl, dims=['half_level'],
        attrs={'units': 'km', 'description': 'atmosphere height profil'},
    )
    date = datetime.now().strftime("%Y-%m-%d")
    ds.attrs = {
        'name': 'Spectral optical depth profiles of ' + gas,
        'experiment': atm + ' based on Idealized CKDMIP interpolation',
        'date': date,
        'source': 'Created by HYGEOS, using CKDMIP data and '
                  f'GATIAB v{VERSION}',
    }
    if save:
        save_filename = f"od_{gas}_{atm}_ckdmip_idealized_solar_spectra.nc"
        path_to_file = Path.joinpath(Path(dir_save), save_filename)
        Path(dir_save).mkdir(parents=True, exist_ok=True)
        if os.path.isfile(path_to_file):
            os.remove(path_to_file)
        ds.to_netcdf(path_to_file)

    return ds


class Gatiab:
    """Gatiab object for gaseous transmission computations.

    Parameters
    ----------
    od_lut : str or Path or xr.Dataset
        Path or xr.Dataset of the gas optical depth LUT (created
        using ckdmip2od function)
    """

    def __init__(self, od_lut: str | Path | xr.Dataset) -> None:
        if isinstance(od_lut, xr.Dataset):
            od = od_lut
        else:
            od = xr.open_dataset(od_lut)
        self.od: xr.Dataset = od
        self.gas: str = od.attrs['name'].split(' ')[-1]  # gas name
        # atmosphere name
        self.atm: str = od.attrs['experiment'].split(' ')[0]
        self.wavenumber: NDArray = od['wavenumber'].values[:]  # in cm-1
        self.half_level: NDArray = od['half_level'].values[:]
        self.P_hl: NDArray = od['P_hl'].values[:]  # air pressure profil
        self.T_hl: NDArray = od['T_hl'].values[:]  # temperature profil
        # z profil as function of half_level
        self.z_atm: NDArray = od['z_atm'].values[:]
        self.mole_fraction_hl: NDArray = od['mole_fraction_hl'].values[:]
        # half level gas pressure
        self.p_gas_hl: NDArray = self.P_hl * self.mole_fraction_hl
        # half level gas density in cm-3
        self.dens_gas_hl: NDArray = (
            (self.p_gas_hl) / (constants.Boltzmann*self.T_hl) * 1e-6
        )

    def get_gas_content(self) -> float:
        """Compute the gas content.

        In DU for O3, in g cm-2 for other gas.
        """
        if self.gas == 'O3':
            gas_content = (1/2.6867e16) * (
                simpson(y=self.dens_gas_hl, x=-self.z_atm)*1e5
            )
        else:
            gas_content = (
                MOLAR_MASS[self.gas.lower()]/constants.Avogadro
            ) * (simpson(y=self.dens_gas_hl, x=-self.z_atm)*1e5)
        return cast(float, gas_content)

    def print_gas_content(self, fmt: str = '%.3F') -> None:
        """Print the gas content with its unit."""
        if self.gas == 'O3':
            print(
                "gas content (" + self.gas + ") = "
                + fmt % self.get_gas_content() + " DU"
            )
        else:
            print(
                "gas content (" + self.gas + ") = "
                + fmt % self.get_gas_content() + " g cm-2"
            )

    def calc(
        self,
        gas_content: NDArray,
        air_mass: NDArray,
        p0: NDArray,
        srf_wvl: list[NDArray],
        rsrf: list[NDArray],
        save: bool = False,
        dir_save: str | Path = './',
    ) -> xr.Dataset:
        """Compute gaseous transmissions.

        Parameters
        ----------
        gas_content : np.ndarray
            Gas content, in Dobson for O3 and in g cm-2 for other
            gas
        air_mass : np.ndarray
            Ratio of slant path optical depth and vertical optical
            depth
        p0 : np.ndarray
            Ground pressure(s) value(s) in hPa
        srf_wvl : list
            SRF wavelengths np.ndarray into an iband list
        rsrf : list
            relative SRF values np.ndarray into an iband list
        save : bool, optional
            If True, save output in netcdf format
        dir_save : str or Path, optional
            Output directory where to save the transmission
            generated lut

        Returns
        -------
        L : xr.Dataset
            Look-up table with the gas transmission as function of
            the instrument band, airmass, gas content and ground
            level pressure
        """

        n_U = len(gas_content)
        nbands = len(srf_wvl)
        n_M = len(air_mass)
        n_p0 = len(p0)

        dens_gas_FPhl = interp1d(
            self.P_hl, self.dens_gas_hl, bounds_error=False,
            fill_value='extrapolate'
        )
        half_level_FPhl = interp1d(
            self.P_hl, self.half_level, bounds_error=False,
            fill_value='extrapolate'
        )

        trans_gas = np.zeros((nbands, n_U, n_M, n_p0), dtype=np.float64)

        wav_min = 1e7/self.wavenumber[-1]
        wav_max = 1e7/self.wavenumber[0]

        for iband in range(0, nbands):

            # Check that current band is defined in the range of
            # the LUT
            if (
                (srf_wvl[iband][0] < wav_min)
                or (srf_wvl[iband][0] > wav_max)
            ):
                raise ValueError(
                    f'band {iband} ({srf_wvl[iband][-1]} to '
                    f'{srf_wvl[iband][0]} nm) is defined outside range '
                    f'({wav_min} to {wav_max} nm)'
                )

            wvn_bi = np.float64(1.)/srf_wvl[iband].astype(np.float64)*1e7
            wvn_bi_extented = self.wavenumber[
                (lambda x: np.logical_and(
                    x >= np.min(wvn_bi)-1, x <= np.max(wvn_bi)+1
                ))(self.wavenumber)
            ]
            lut_gas_bi = self.od.sel(wavenumber=wvn_bi_extented)
            isrf_int = interp1d(
                wvn_bi, rsrf[iband], bounds_error=False, fill_value=0.
            )(wvn_bi_extented)
            for iU in range(0, n_U):
                for ip in range(0, n_p0):
                    p_ip = np.sort(np.concatenate((
                        self.P_hl[self.P_hl[:] < p0[ip]*1e2],
                        np.array([p0[ip]*1e2], dtype='float32'),
                    )))
                    half_level_ip = half_level_FPhl(p_ip)
                    dens_gas_ip = dens_gas_FPhl(p_ip).astype(np.float64)
                    z_atm_ib = interp1d(
                        self.half_level, self.z_atm, bounds_error=False,
                        fill_value='extrapolate'
                    )(half_level_ip)

                    if self.gas == 'O3':
                        dens_gas_iUp = dens_gas_ip * (
                            2.6867e16 * gas_content[iU]
                            / (simpson(y=dens_gas_ip, x=-z_atm_ib) * 1e5)
                        )
                    else:
                        dens_gas_iUp = dens_gas_ip * (
                            gas_content[iU] / MOLAR_MASS[self.gas.lower()]
                            * constants.Avogadro
                            / (simpson(y=dens_gas_ip, x=-z_atm_ib) * 1e5)
                        )

                    # convert to abs coeff then interpolate
                    ot = lut_gas_bi['optical_depth'].values.astype(
                        np.float64
                    )
                    ot = np.append(
                        np.zeros((1, len(wvn_bi_extented))), ot, axis=0
                    ).astype(np.float64)
                    ot = np.swapaxes(ot, 0, 1)
                    dz = diff1(self.z_atm).astype(np.float64)
                    with np.errstate(invalid='ignore'):
                        k = abs(ot/dz)
                    k[np.isnan(k)] = 0
                    sl = slice(None, None, 1)
                    k = k[:, sl]
                    C_abs = np.swapaxes(k, 0, 1)
                    C_abs_ib = make_interp_spline(
                        self.z_atm[::-1], C_abs[::-1, :], k=1, axis=0
                    )(z_atm_ib[::-1])[::-1, :]

                    # reconvert to optical depth
                    dz_ib = np.abs(diff1(z_atm_ib))
                    od_ib = dz_ib[:, None] * C_abs_ib

                    fac_iUp = dens_gas_iUp[:]/dens_gas_ip[:]
                    tau_zw = fac_iUp[:, None] * od_ib[:]
                    tau_zw[tau_zw < 0] = 0.
                    tau_w = np.sum(tau_zw, axis=0)

                    # Consider air_mass
                    for iM in range(0, n_M):
                        tau_wM = tau_w * air_mass[iM]
                        trans_wM = np.exp(-tau_wM)
                        # simpson to consider the case where dw is
                        # varying, this is the case at 625 and
                        # 1941.75 nm
                        num = simpson(
                            y=trans_wM[isrf_int > 0]*isrf_int[isrf_int > 0],
                            x=wvn_bi_extented[isrf_int > 0],
                        )
                        den = simpson(
                            y=isrf_int[isrf_int > 0],
                            x=wvn_bi_extented[isrf_int > 0],
                        )
                        trans = num/den
                        trans_gas[iband, iU, iM, ip] = trans

        bands = get_bands(srf_wvl, rsrf)

        ds = xr.Dataset()
        ds['trans'] = xr.DataArray(
            trans_gas.astype(np.float32),
            dims=["lambda", "U", "M", "p0"],
            coords={
                'lambda': bands, 'U': gas_content,
                'M': air_mass, 'p0': p0,
            },
        )

        ds['lambda'].attrs = {
            'units': 'nm', 'description': 'Instrument averaged bands'
        }
        if self.gas == 'O3':
            ds['U'].attrs = {
                'units': 'Dobson',
                'description': 'Total column content of the gas',
            }
        else:
            ds['U'].attrs = {
                'units': 'g/cm²',
                'description': 'Total column content of the gas',
            }
        ds['M'].attrs = {
            'units': 'None',
            'description': 'Airmass i.e. ratio of slant path optical '
                           'depth and vertical optical depth',
        }
        ds['p0'].attrs = {
            'units': 'hPa', 'description': 'Pressure at ground level'
        }
        ds['trans'].attrs = {
            'units': 'None', 'description': self.gas + ' transmission'
        }

        date = datetime.now().strftime("%Y-%m-%d")
        ds.attrs = {
            'atm': self.atm, 'date': date,
            'source': f'Created using GATIAB v{VERSION}',
        }
        if save:
            save_filename = f"trans_{self.gas}_{self.atm}_gatiab.nc"
            path_to_file = Path.joinpath(Path(dir_save), save_filename)
            if os.path.isfile(path_to_file):
                os.remove(path_to_file)
            ds.to_netcdf(path_to_file)
        return ds

    def update_gas_content(
        self,
        gas_content: float,
        save: bool = False,
        dir_save: str | Path = './',
    ) -> None:
        """Update the gas optical depth LUT with a new columnar gas
        content.

        - Update self.od, self.mole_fraction_hl and
          self.dens_gas_hl

        Parameters
        ----------
        gas_content : float
            Gas content, in Dobson for O3 and in g cm-2 for other
            gas
        save : bool, optional
            If True, save output in netcdf format
        dir_save : str or Path, optional
            Output directory where to save the updated optical
            depth lut
        """

        # convert to abs coeff then interpolate
        ot = self.od['optical_depth'].values.astype(np.float64)
        ot = np.append(
            np.zeros((1, len(self.wavenumber))), ot, axis=0
        ).astype(np.float64)
        ot = np.swapaxes(ot, 0, 1)
        dz = diff1(self.z_atm).astype(np.float64)
        with np.errstate(invalid='ignore'):
            k = abs(ot/dz)
        k[np.isnan(k)] = 0
        sl = slice(None, None, 1)
        k = k[:, sl]
        C_abs = np.swapaxes(k, 0, 1)

        if self.gas == 'O3':
            dens_gas_U = self.dens_gas_hl * (
                2.6867e16 * gas_content
                / (simpson(y=self.dens_gas_hl, x=-self.z_atm) * 1e5)
            )
            units = 'Dobson'
        else:
            dens_gas_U = self.dens_gas_hl * (
                gas_content / MOLAR_MASS[self.gas.lower()]
                * constants.Avogadro
                / (simpson(y=self.dens_gas_hl, x=-self.z_atm) * 1e5)
            )
            units = 'g/cm²'

        # reconvert to optical depth
        dz = np.abs(diff1(self.z_atm))
        od = dz[:, None] * C_abs
        fac_U = dens_gas_U[:]/self.dens_gas_hl[:]
        tau = fac_U[:, None] * od[:]
        tau[tau < 0] = 0.

        # compute the new mole fractions
        mole_fraction_hl = (
            dens_gas_U[:]*1e6 * constants.Boltzmann*self.T_hl
        ) / (self.P_hl)
        z_fl = self.z_atm[1:] + 0.5*np.abs(np.diff(self.z_atm))
        mole_fraction_fl = interp1d(self.z_atm, mole_fraction_hl)(z_fl)

        # create the new od LUT
        ds = self.od.copy()
        ds['optical_depth'] = xr.DataArray(
            tau[1:, :].astype(np.float32), dims=['level', 'wavenumber'],
            attrs={'units': 'None', 'description': 'optical depth'},
        )
        ds['mole_fraction_fl'] = xr.DataArray(
            mole_fraction_fl, dims=['level'],
            attrs={
                'units': 'None',
                'description': 'full level mole fraction',
            },
        )
        ds['mole_fraction_hl'] = xr.DataArray(
            mole_fraction_hl, dims=['half_level'],
            attrs={
                'units': 'None',
                'description': 'half level mole fraction',
            },
        )
        date = datetime.now().strftime("%Y-%m-%d")
        ds = ds.assign_attrs(
            experiment=(
                f"{self.atm} rescaled for {self.gas} columnar content "
                f"of {gas_content:.3F} {units}, based on Idealized "
                "CKDMIP interpolation"
            ),
            date=date,
            source=ds.source + f' before and v{VERSION} after '
            'rescaling',
        )

        if save:
            save_filename = (
                f"od_{self.gas}_{self.atm}_U-{gas_content:.2E}"
                "_ckdmip_idealized_solar_spectra.nc"
            )
            path_to_file = Path.joinpath(Path(dir_save), save_filename)
            if os.path.isfile(path_to_file):
                os.remove(path_to_file)
            ds.to_netcdf(path_to_file)

        # Update the attributs
        self.od = ds
        self.mole_fraction_hl = mole_fraction_hl
        self.dens_gas_hl = dens_gas_U
