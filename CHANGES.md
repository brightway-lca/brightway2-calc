# Changelog

### 1.7.1 (2018-02-14)

Compatibility with `presamples` release version

## 1.7 (2018-01-18)

Add compatibility with `bw_presamples`

### 1.6.4 (2018-01-12)

Really fix bug in seed generation for pooled Monte Carlo calculations

### 1.6.3 (2018-01-11)

* [JOSS submission](https://joss.theoj.org/papers/6c24869ed7f1e66b3b837c31579c6fe5)
* Fix bug in MultiMonteCarlo
* Add some logging to support presamples in the future

### 1.6.2 (2017-04-17)

Fix license text

### 1.6.1 (2017-04-06)

Simplify indexing

## 1.6 (2017-04-05)

Replace `bw2speedups` indexing with numpy array trickiness which is ~5 times faster

### 1.5.4 (2017-02-24)

Remove non-ascii characters from license text, because [setuptools](ttps://github.com/pypa/setuptools/issues/984)

### 1.5.3 (2016-10-28)

* Restructure imports to not depend on `bw2data`
* Use `io.open` in `setup.py`

### 1.5.2 (2016-10-28)

Specify encoding of license file

### 1.5.1 (2016-09-15)

Bugfix for broken import statement

## 1.5 (2016-09-15)

Merge pull request from Adrian Haas to enable Pardiso solver usage when available.

## 1.4 (2016-07-14)

* Added utility functions for `load_calculation_package` and `save_calculation_package` for independent LCAs and cloud computing.
* Compatibility with `bw2data` 2.3

### 1.3.6 (2016-07-01)

Fixed bugs where RNG and technosphere matrix builder would change values in arrays meant to be static

### 1.3.5 (2016-07-01)

Fix bugs and add tests for `ParameterVectorLCA`

### 1.3.4 (2016-06-10)

Changed `ParameterVectorLCA`: Can no longer be called, split off `rebuild_all` into a separate method, added tests.

### 1.3.3 (2016-06-10)

Better test coverage and Windows comaptibility

### 1.3.2 (2016-06-08)

* FEATURE: Add class and mixin for Monte Carlo using direct solvers
* CHANGE: Move tests to root directory and add Monte Carlo tests
* CHANGE: Consistent use of `__next__` and `next()` so that all Monte Carlo iterator classes are Py2/3 compatible and programmed the same way. `ParameterVectorLCA.next()` will no longer work on Python 3; instead, call `next(ParameterVectorLCA)`. When providing a new vector, call the class itself (after it is instantiated): `pv = ParameterVectorLCA(args); pv(new_vector)`.

### 1.3.1 (2016-06-06)

* CHANGE:Updates for bw2data 2.2
* BUGFIX: Correctly handle regionalized CFs in site-generic calculations
* FEATURE: Add contribution methods to LCA classes

## 1.3 (2016-05-28)

BUGFIX: Correctly handle project names in multiprocess calculations

### 1.2.1 (2016-03-14)

BUGFIX: `switch_*` was seriously broken due to new handling of processed arrays filepaths

## 1.2 (2016-03-14)

* Feature: Py3 compatibility
* FEATURE: Independent LCAs which don't rely on bw2data and the brightway2 ecosystem
* Feature: Add `DenseLCA`
* FEATURE: Added `switch_*` and `to_dataframe` methods to LCA class
* FEATURE: Allow graph traversal to skip links in static databases
* BUGFIX: Terminate multiprocessing pools after calculations
* CHANGE: Load data automatically in Monte Carlo
* CHANGE: Automatically clean dirty databases before starting calculations

Plus lots of small bugfixes, and compatibility with `projects`.

# 1.0 (2015-03-08)

CHANGE: Split activities and products

### 0.17.1 (2015-02-13)

BUGFIX: Don't require substitution in `TYPE_DICTIONARY`.

## 0.17 (2015-02-13)

* FEATURE: Properly handle `substitution` type exchanges
* CHANGE: Compatible with bw2data version 2
* BUGFIX: Fix handling of nested dependent databases

### 0.16.1 (2014-12-05)

* CHANGE: Better documentation for most code.
* BUGFIX: Graph traversal handles most coproducts, and raises sensible errors when it can't.

## 0.16 (2014-08-03)

FEATURE: Changes in MatrixBuilder should make normal static LCA calculations about three times faster.

### 0.15.1 (2014-07-30)

Update dependencies.

## 0.15 (2014-06-11)

BREAKING CHANGE: Use `Database.filename` for processed data. Requires update to bw2data version 0.16 or greater.

## 0.13 (2014-04-16)

* BREAKING CHANGE: LCA.fix_dictionaries now sets/uses `_mapped_dict` to determine if `fix_dictionaries` has been called.
* BUGFIX: LCA.build_demand_array doesn't break if `fix_dictionaries` has been called.

## 0.12 (2014-02-13)

BREAKING CHANGE: Matrix builder will only include parameter array rows that are correctly mapped, instead of raising an error when unmapped rows occur. This behaviour can be turned off by passing `drop_missing=False`.

### 0.11.1 (2014-01-29)

BUGFIX: Change column names in method matrix building to be consistent with `bw2data` 0.11

## 0.11 (2014-01-26)

* BREAKING CHANGE: Graph traversal was reworked, and some functionality for interpreting the output was moved to `bw2analyzer`.
* BREAKING CHANGE: Deleted `SimpleRegionalizedLCA` class. Regionalization will be provided in bw2regional.
* BREAKING CHANGE: Deleted initial sensitivity work, moved for now to branch, as it was not yet usable.
* FEATURE: Much better and more thorough documentation.
* FEATURE: Improved testing and test coverage
