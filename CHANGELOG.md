
# GATIAB CHANGELOG


## v1.1.2
Release date: 05-08-2026

* Bug fixes
  - Correct wrong interpolation results in `vec_float_indexing` when a slice
    precedes an interpolated dimension (the corner-selection bit was read at
    the raw dimension position instead of the position among interpolated
    keys, unlike `LUT.__getitem__` of the luts package). This affected the
    H2O path of `ckdmip2od` with `float_indexing='fast'`
  - `vec_float_indexing` no longer crashes with 1-D data or with 5 or more
    interpolated dimensions

* Code organization and modernization
  - Move the package constants into a new `gatiab/constants.py` module
    and re-export them (with `VERSION`) from the package init
  - The version is now read automatically, with pyproject.toml as single
    source of truth (`GATIAB_VERSION` replaced by `VERSION`)
  - Add type hints to all modules and reformat to PEP 8 line lengths
    (79 for code, 72 for docstrings and comments)
  - Rename module constants to upper snake case
  - Add pyright type checking with a "typecheck" pixi task

* Tests
  - Track tests/conftest.py and skip the tests needing the CKDMIP and
    atmosphere data when the `--dir-ckdmip` and `--dir-atm` options are
    not provided (fixes the continuous integration runs)
  - Add exhaustive unit tests for `vec_float_indexing`
    (tests/test_vec_float_indexing.py), cross-checked against scipy
    `RegularGridInterpolator` and the luts package
  - Add pytest to the pixi environment and a "test" pixi task


## v1.1.1
Release date: 04-05-2026

* Robustness and error handling improvements
  - Avoid crash with `ckdmip2od` when the output directory does not exist
  - Fail loudly when a spectral band is not defined over the LUT range
  - Correct out-of-bounds error in `vec_float_indexing` with key=N-1

* Performance and code maintenance
  - Optimize `get_binary_mat` function
  - Use `Path.glob` instead of glob and automatically convert ckdmip input
    to Path objects

* Packaging and metadata updates
  - Add pixi section in pyproject toml file
  - Use standard units abbreviations

* Documentation updates
  - Update README with corrected HYGEOS URL and additional usage information
  - Add Anaconda/conda-forge URL in README
  - General README update and cleanup


## v1.1.0
Release date: 21-11-2024

* Add of the method `update_gas_content` in Gatiab class
  - The initial od LUT can now be modified to match with the new columnar
    gas content

* Some improvements and corrections
  - Now package data are considered in the pyproject toml file
  - New optional parameter in method `print_gas_content`
  - Correction in the example section of the README file


## v1.0.2
Release date: 20-11-2024

* Correct license name incoherence between licence file name and name given
  in pyproject toml file

* Add missing classifiers in pyproject toml file


## v1.0.1
Release date: 18-11-2024

* Add missing N2 molar mass


## v1.0.0
Release date: 17-10-2024

First public release.
