# GATIAB
GATIAB (Gasous Absorption Transmissions at Instrument Averaged Bands). \
This module provides gaseous transmissions base on CKDMIP idealized Look-Up tables and sensor spectral response functions.

Mustapha Moulana  
[HYGEOS](www.hygeos.com)

-----------------------------------------


## CKDMIP data
The CKDMIP (the Correlated K-Distribution Model Intercomparison Project) Idealized Look-Up tables are necessary to use the gatiab module. See: Hogan et al. ->  https://gmd.copernicus.org/articles/13/6501/2020/

The CKDMIP documentation and data are available here: https://confluence.ecmwf.int/display/CKDMIP

## AFGL atmophere LUTs
The gatiab module needs afgl atmosphere profils. Those atmophere look-up tables can be dowloaded from SMART-G (Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU) -> https://github.com/hygeos/smartg

The SMART-G makefile can be used:
```
$ make auxdata_atm
```

## Installation
The module can be installed using the following command:
```
$ pip install git+https://github.com/hygeos/gatiab.git
```

## Testing
Example of pytest.ini file:
```
[pytest]
addopts=
    --dir-ckdmip="/path/to/ckdmip/dir/"
    --dir-atm="path/to/atm/dir/"
    -s -v
```
Run the command `pytest tests/` to check that everything is running correctly.

## Exemples
```
from gatiab import ckdmip2od, Gatiab
import xarray as xr
import glob

# Specify the ckdmip and atmophere directory paths
dir_ckdmip = "path/to/ckdmip/dir/" 
dir_atm = "path/to/atm/dir/"

# First create optical depth LUTs of a given atmosphere
# wavenumber units -> cm-1
ds = ckdmip2od(gas='O3', dir_ckdmip=dir_ckdmip, dir_atm=dir_atm, atm='afglus',
               wvn_min = 4000., wvn_max=26000., save=True)

# Second we need the instrument spectral response
# Here we use the LUTs we have created for testing of Sentinel 3A OLCI
# Other SRF intruments -> https://nwp-saf.eumetsat.int/site/software/rttov/download/coefficients/spectral-response-functions/
rsrf_files = sorted(glob.glob("./tests/S3A_OLCI_rsrf/*.nc"))
nbands = len(rsrf_files)
rsrf = [] # iband list with relative spectral response as function of wavelength
srf_wvl = [] # iband list with wavelength in nanometer
for i in range (0, nbands):
    with xr.open_dataset(rsrf_files[i]) as ds:
        rsrf.append(ds['rsrf'].values)
        srf_wvl.append(ds['wvl'].values)

# Third compute the gas transmission as function of gas content, airmass, ground pressure and wavelength
gt = Gatiab(ds)
gt.print_gas_content() # print the standard afgl total column content of O3

gas_content = np.array([250., 300., 350.]) # in DU
air_mass = np.array([3., 4., 5.])
p0 = gt.od['P_hl'][-2:].data*1e-2 # ground pressure in hectopascal
ds_gt = gt.calc(gas_content, air_mass, p0, srf_wvl, rsrf)
print(ds_gt)
```
