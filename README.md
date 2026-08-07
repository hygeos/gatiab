<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="https://raw.githubusercontent.com/hygeos/gatiab/master/gatiab/img/gatiab-logo-horizontal-dark-bg.png">
    <img alt="GATIAB" width="320"
         src="https://raw.githubusercontent.com/hygeos/gatiab/master/gatiab/img/gatiab-logo-horizontal-light-bg.png">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/gatiab/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/gatiab"></a>
  <a href="https://anaconda.org/conda-forge/gatiab"><img alt="conda-forge version" src="https://img.shields.io/conda/vn/conda-forge/gatiab"></a>
  <a href="https://github.com/hygeos/gatiab/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/hygeos/gatiab/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://github.com/hygeos/gatiab/actions/workflows/ruff.yml"><img alt="Ruff" src="https://github.com/hygeos/gatiab/actions/workflows/ruff.yml/badge.svg"></a>
  <a href="https://pepy.tech/project/gatiab"><img alt="Downloads" src="https://pepy.tech/badge/gatiab"></a>
</p>

<p align="center">
  <em>Gaseous Absorption Transmissions at Instrument Averaged Bands</em>
</p>

<p align="center">
  <a href="#installation">Installation</a> &middot;
  <a href="#input-data">Input data</a> &middot;
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#api-overview">API</a> &middot;
  <a href="#testing">Testing</a>
</p>

---

**GATIAB** computes gaseous transmissions based on the CKDMIP idealized
look-up tables and on the spectral response functions of an instrument.

Mustapha Moulana &mdash; [HYGEOS](https://hygeos.com/en/)

## Overview

The package provides:

1. **Generation of gas optical depth look-up tables** for a given
   atmosphere, from the CKDMIP shortwave idealized spectra
   (`ckdmip2od`).
2. **Computation of the gaseous transmissions at instrument averaged
   bands**, as a function of the gas content, the airmass and the
   ground pressure, using the spectral response functions of the
   instrument (`Gatiab.calc`).
3. **Rescaling of an optical depth look-up table** to a new columnar
   gas content (`Gatiab.update_gas_content`).

The gases handled by the CKDMIP idealized spectra are `H2O`, `CO2`,
`O2`, `O3`, `N2O`, `N2` and `CH4`. Any atmosphere profile can be used,
as long as it follows the SMART-G atmosphere file standard &mdash; see
[Atmosphere profiles](#atmosphere-profiles).

## Installation

The module can be installed using one of the following commands:

```shell
conda install -c conda-forge gatiab
```

```shell
pip install gatiab
```

```shell
pip install git+https://github.com/hygeos/gatiab.git
```

Python 3.10 or newer is required. See
[CHANGELOG.md](https://github.com/hygeos/gatiab/blob/master/CHANGELOG.md)
for the release history.

## Input data

Two external datasets are needed. They are **not** shipped with the
package.

### CKDMIP idealized look-up tables

The CKDMIP (Correlated K-Distribution Model Intercomparison Project)
idealized spectra are the spectroscopic source of the optical depth
LUTs. See [Hogan et al.
(2020)](https://gmd.copernicus.org/articles/13/6501/2020/).

The documentation and the data are available on the
[CKDMIP page](https://confluence.ecmwf.int/display/CKDMIP).

`ckdmip2od` reads the files named
`ckdmip_idealized_sw_spectra_<gas>_const*.h5` from the directory given
by `dir_ckdmip`.

### Atmosphere profiles

The module needs an atmosphere profile giving the vertical pressure,
temperature and gas number density. A set of AFGL profiles can be
downloaded with the `auxdata` module of
[SMART-G](https://github.com/hygeos/smartg) (Speed-up Monte Carlo
Advanced Radiative Transfer Code using GPU):

```python
from pathlib import Path
from smartg.auxdata import download

# data_type='atm' downloads the atmosphere profiles only
download(Path("/dir/where/to/save/data"), data_type='atm')
```

This creates `/dir/where/to/save/data/atmospheres/`, the directory to
pass as `dir_atm`. SMART-G expects its `SMARTG_DIR_AUXDATA`
environment variable to point at the parent directory, so with a
SMART-G installation already configured:

```python
dir_atm = Path(os.environ['SMARTG_DIR_AUXDATA']) / "atmospheres"
```

Six AFGL profiles are provided:

| `atm` | Profile |
| --- | --- |
| `afglus` | U.S. Standard (1976) |
| `afglt` | Tropic (15N annual average) |
| `afglms` | Mid-Latitude Summer (45N July) |
| `afglmw` | Mid-Latitude Winter (45N January) |
| `afglss` | Sub-Arctic Summer (60N July) |
| `afglsw` | Sub-Arctic Winter (60N January) |

#### Using your own atmosphere

Those six names are not a closed list: `atm` is simply the name of a
file looked up in `dir_atm`, so any profile works as long as the file
follows the SMART-G atmosphere standard. It must be a netCDF4 file
with a single dimension named `z_atm`, holding at least:

| Variable | Unit | Description |
| --- | --- | --- |
| `z_atm` | km | Altitude grid, from the top of the atmosphere down to the ground |
| `P` | hPa | Pressure |
| `T` | K | Temperature |
| `<GAS>` | cm-3 | Number density, one variable per gas, named after the gas (`O3`, `H2O`, ...) |

Only the variable matching the requested `gas` is read, so a file does
not need to contain them all. The SMART-G files additionally provide
`dens`, `CO`, `NO2` and `SO2`, which gatiab ignores.

Two points to respect:

- The altitude grid must be **decreasing** (top of the atmosphere
  first, ground last), as in the SMART-G files.
- Pass the bare file stem (`atm='myatm'` for `myatm.nc`) and avoid
  spaces in it: the name is stored in, and read back from, the
  `experiment` attribute of the optical depth LUT, and it is reused in
  the output file names.

### Instrument spectral response functions

Any SRF can be used, as long as it is provided as a list of wavelength
arrays (in nm) and a list of relative response arrays, one per band. A
large collection is available from the
[NWP SAF / RTTOV spectral response functions](https://nwp-saf.eumetsat.int/site/software/rttov/download/coefficients/spectral-response-functions/).

The Sentinel-3A OLCI responses used by the tests are bundled in
`tests/S3A_OLCI_rsrf/`.

## Quickstart

```python
import glob

import numpy as np
import xarray as xr

from gatiab import Gatiab, ckdmip2od

# Specify the ckdmip and atmosphere directory paths
dir_ckdmip = "/path/to/ckdmip/dir/"
dir_atm = "/path/to/atm/dir/"

# 1. Create the optical depth LUT of a gas for a given atmosphere
#    (wavenumber units -> cm-1)
od_lut = ckdmip2od(gas='O3', dir_ckdmip=dir_ckdmip, dir_atm=dir_atm,
                   atm='afglus', wvn_min=4000., wvn_max=26000.,
                   save=True)

# 2. Read the instrument spectral response functions. Here the
#    Sentinel-3A OLCI responses bundled with the tests are used.
rsrf_files = sorted(glob.glob("./tests/S3A_OLCI_rsrf/*.nc"))
rsrf = []     # per band: relative spectral response
srf_wvl = []  # per band: wavelength in nanometer
for rsrf_file in rsrf_files:
    with xr.open_dataset(rsrf_file) as ds_srf:
        rsrf.append(ds_srf['rsrf'].values)
        srf_wvl.append(ds_srf['wvl'].values)

# 3. Compute the gas transmissions as a function of the gas content,
#    the airmass, the ground pressure and the wavelength
gt = Gatiab(od_lut)
gt.print_gas_content()  # standard afgl total column content of O3

gas_content = np.array([250., 300., 350.])   # in DU (g cm-2 if not O3)
air_mass = np.array([3., 4., 5.])            # slant / vertical OD ratio
p0 = gt.od['P_hl'][-2:].data * 1e-2          # ground pressure in hPa
p0 = p0[::-1]                                # P_hl starts at the TOA
trans = gt.calc(gas_content, air_mass, p0, srf_wvl, rsrf)
print(trans)
```

`Gatiab.calc` returns a dataset with the `trans` variable of dimensions
`[lambda, U, M, p0]`, where `lambda` holds the SRF-weighted central
wavelengths of the bands in nm, `U` the columnar gas content, `M` the
airmass and `p0` the ground pressure in hPa. With `save=True`, it is
written as `trans_{gas}_{atm}_gatiab.nc`.

An existing LUT can be rescaled to another columnar gas content, in
place:

```python
gt = Gatiab("od_O3_afglus_ckdmip_idealized_solar_spectra.nc")
gt.update_gas_content(250.)  # in DU for O3, g cm-2 otherwise
gt.print_gas_content()
```

## API overview

Everything below is importable directly from `gatiab`.

| Object | Description |
| --- | --- |
| `ckdmip2od(gas, dir_ckdmip, dir_atm, ...)` | Generate the optical depth LUT of a gas for a given atmosphere, from the CKDMIP idealized look-up tables. Returns a `Dataset`. |
| `Gatiab(od_lut)` | Gatiab object built from an optical depth LUT (path or `Dataset`). |
| `Gatiab.calc(gas_content, air_mass, p0, srf_wvl, rsrf, ...)` | Compute the gaseous transmissions at the instrument averaged bands. |
| `Gatiab.get_gas_content()` | Columnar gas content of the LUT. |
| `Gatiab.print_gas_content(fmt='%.3F')` | Print the gas content with its unit. |
| `Gatiab.update_gas_content(gas_content, ...)` | Rescale the LUT to a new columnar gas content, in place. |
| `get_bands(srf_wvl, rsrf)` | SRF-weighted central wavelength of each band, in nm. |
| `get_zatm(P_hl, T_fl, M_air, g, ..., method='barometric')` | Altitude profile in km, from the pressure and temperature variability ('barometric' or 'hypsometric'). |
| `vec_float_indexing(data, keys)` | Vectorized multilinear interpolation indexing with float scalars or 1-D arrays, based on the `Idx` method of [luts](https://github.com/hygeos/luts). |
| `get_binary_mat(ndim)` | Matrix whose row `i` holds the little-endian binary digits of `i`. |
| `VERSION` | Version of the installed package. |
| `ACCEL_DUE_TO_GRAVITY` | Gravitational acceleration, 9.80665 m s-2. |
| `MOLAR_MASS_AIR` | Dry air molar mass, 28.970 g mol-1. |
| `MOLAR_MASS` | Molar masses in g mol-1, keyed by lowercase gas name. |

### Units

| Quantity | Unit |
| --- | --- |
| Gas content (`U`, `gas_content`) | DU for `O3`, g cm-2 for the other gases |
| Ground pressure (`p0`) | hPa |
| SRF wavelengths (`srf_wvl`, `lambda`) | nm |
| Wavenumbers (`wvn_min`, `wvn_max`) | cm-1 |
| Pressure / temperature in the LUT (`P_hl`, `T_fl`) | Pa / K |
| Altitude (`z_atm`) | km |

Docstrings are the reference documentation:

```python
>>> import gatiab
>>> help(gatiab.ckdmip2od)
```

## Testing

```shell
pytest tests/
```

Some tests need the CKDMIP and atmosphere data, and are skipped if the
`--dir-ckdmip` and `--dir-atm` options are not provided. Example of
`pytest.ini` file:

```ini
[pytest]
addopts=
    --dir-ckdmip="/path/to/ckdmip/dir/"
    --dir-atm="/path/to/atm/dir/"
    -s -v
```

## References

- Hogan, R. J. and Matricardi, M.: *Evaluating and improving the
  treatment of gases in radiation schemes: the Correlated K-Distribution
  Model Intercomparison Project (CKDMIP)*, Geosci. Model Dev., 13,
  6501-6521, 2020.
  <https://gmd.copernicus.org/articles/13/6501/2020/>
- CKDMIP documentation and data:
  <https://confluence.ecmwf.int/display/CKDMIP>
- SMART-G (atmosphere profiles):
  <https://github.com/hygeos/smartg>
- NWP SAF / RTTOV spectral response functions:
  <https://nwp-saf.eumetsat.int/site/software/rttov/download/coefficients/spectral-response-functions/>

## License

GATIAB is free of charge for non-commercial, scientific research
purposes. It may not be transferred or sublicensed to a third party.
For commercial use, please [contact HYGEOS](https://hygeos.com/en/).

See [LICENSE.txt](https://github.com/hygeos/gatiab/blob/master/LICENSE.txt)
for the complete terms of use.
