# calc Changelog

## 2.0.DEV13 (2023-05-07)

* CI workflow updates
* Merge [#65](https://github.com/brightway-lca/brightway2-calc/pull/65): Add PyPI and conda-forge badge
* Fix hidden dependency on `bw2data`

## 2.0.DEV12 (2022-09-19)

* Add some backwards compatiblity methods

## 2.0.DEV11 (2022-08-31)

* Merged [PR #63 Multifunctional graph traversal](https://github.com/brightway-lca/brightway2-calc/pull/63)
* Changed `GraphTraversal` to `AssumedDiagonalGraphTraversal`. `GraphTraversal` still exists as a proxy but raises a `DeprecationWarning`

## 2.0.DEV10 (2022-08-19)

* Add ``LCA.to_dataframe``, based on work by Ben Portner

## 2.0.DEV9 (2022-07-07)

* [#61 wrap_functional_unit missing from multi_lca.py](https://github.com/brightway-lca/brightway2-calc/pull/58)
* MultiLCA is useable again

## 2.0.DEV8 (2022-06-28)

* [#58 use logger at module level, not from LCA](https://github.com/brightway-lca/brightway2-calc/pull/58)
* [#59 Fix a number of testing issues](https://github.com/brightway-lca/brightway2-calc/pull/59)

## 2.0.DEV7 (2022-05-22)

* Add `LCA.keep_first_iteration` to make iteration simpler

## 2.0.DEV6 (2022-04-23)

* Add an optional warning on LCA instantiation if excluding resources (arrays or distributions) which could be useful
* Add function stubs to be used by subclasses on iteration

## 2.0.DEV5 (2021-11-26)

* Fix a bug in `switch_method` if given a `bw2data` method tuuple instead of a list of datapackages.

## 2.0.DEV4 (2021-11-03)

* Add `invert_technosphere_matrix` with algo from @haasad
* Fix `switch_method`, `switch_normalization`, `switch_weighting`

Compatibility changes:

* `LCA.score` will return weighted or normalized score, if weighting or normalization has been performed
* `LCA.weighting` will now trigger a deprecation warning. Switch to `.weight` instead.
* `LCA.redo_lci` deprecated in favor of `LCA.lci(demand)`; `LCA.redo_lcia` deprecated in favor of `LCA.lcia(demand)`

## 2.0.DEV3 (2021-10-17)

* Fix for constructing characterization matrices with semi-regionalized impact categories

## 2.0.DEV2 (2021-10-01)

* More 2.5 work and fixes

# 2.0.DEV1

Version 2.0 brings a number of large changes, while maintaining backwards compatibility (except for dropping Py2). The net result of these changes is to prepare for a future where data management is separated from calculations, and where working with large, complicated models is much easier.

## Future DEV releases

Before 2.0 is released, the following features will be added:

* Presamples will be adapted to use `bw_processing`
* Logging will be taken seriously :)
* ~~LCA results to dataframes~~

## Breaking changes

### Simplification of user endpoints

The structure of this library has been simplified, as the `LCA` class can now perform static, stochastic (Monte Carlo), iterative (scenario-based), and single-matrix LCA calculations. Matrix building has been moved to the [matrix_utils](https://github.com/brightway-lca/matrix_utils) library.

### Python 2 compatibility removed

Removing the Python 2 compatibility layer allows for much cleaner and more compact code, and the use of some components from the in-development Brightway version 3 libraries. Compatible with `bw2data` version 4.0.

### Removal of classes and methods

* `LCA.rebuild_*_matrix` methods are removed. See the [TODO]() notebook for alternatives.
* `DirectSolvingMixin` and `DirectSolvingMonteCarloLCA` are removed, direct solving is now the default
* `ComparativeMonteCarlo` is removed, use `MultiLCA(use_distributions=True)` instead
* `SingleMatrixLCA` is remove, use `LCA` instead. It allows for empty biosphere matrices.

### Simplified handling of mapping dictionaries

Mapping dictionaries map the database identifiers to row and column indices. In 2.5, these mapping dictionaries are only created on demand; avoiding their creation saves a bit of time and memory.

Added a new class (`DictionaryManager`) and made it simpler reverse, remap, and get the original dictionaries inside an `LCA`. Here is an example:

```python
LCA.dicts.biosphere[x]
>> y
LCA.dicts.biosphere.original # if remapped with activity keys
LCA.dicts.biosphere.reversed[y]  # (generated on demand)
>> x
```

The dictionaries in a conventional LCA are:

* LCA.dicts.product
* LCA.dicts.activity
* LCA.dicts.biosphere

~~`LCA.reverse_dict` is removed; all reversed dictionaries are available at `LCA.dicts.{name}.reversed`~~.

In 2.5, these mapping dictionaries are not automatically "remapped" to the `(database name, activity code)` keys. You will need to call `.remap_inventory_dicts()` after doing an inventory calculation to get mapping dictionaries in this format.

### Weighting is a diagonal matrix instead of a single number

It is easier to have everything in the same mode of operation. This also allows for the use of arrays, distributions, interfaces, etc. in weighting. Implemented in new `SingleValueDiagonalMatrix` class.

## Architectual changes

### Use of `bw_processing`

We now use [bw_processing](https://github.com/brightway-lca/bw_processing) to load processed arrays. `bw_processing` has separate files for the technosphere and biosphere arrays, and explicit indication of . Therefore, the `TechnosphereBiosphereMatrixBuilder` is no longer necessary, and is removed.

### No dependency on `bw2data`

`bw2data` is now an optional install, and even if available only a single utility function is used to prepare input data. `bw2calc` is primarily intended to be used as an independent library.

### Changes in Monte Carlo


## Smaller changes

### New LCA input specification

The existing input specification is still there, but this release also adds the ability to specify input arguments compatible with Brightway version 3. Previously, we would write `LCA({some demand}, method=foo)` - this requires `bw2calc` to use `bw2data` to figure out the dependent databases of the functional unit in `some demand`, and then to get the file paths of all the necessary files for both the inventory and impact assessment. The new syntax is `LCA({some demand}, data_objs)`, where `some demand` is already integer IDs, and `data_objects` is a lists of data packages (either in memory or on the filesystem).

`bw2data` has a helper function to prepare arguments in the new syntax: `prepare_lca_inputs`.

This new input syntax, with consistent column labels for all structured arrays, removes the need for `IndependentLCAMixin`. This is deleted, and the methods `get_vector`, `get_vector_metadata`, and `set_vector` are added.

### More robust matrix building

More tests were identified, and undefined behaviour is now specified. For example, the previous matrix builders assumed that the values in the provided row or column dictionaries were sequential integers starting from zero - this assumption is now relaxed, and we allow this dictionary values to start with an offset. There are also tests and documentation on what happens under various cases when `drop_missing` is `False`, but missing values are present.

### 1.8.0 (2020-02-27)

* Replace `.todense` with `.toarray` to satisfy changes in Scipy API
* Add `atol` parameter to iterative solver to satisfy changes in Scipy API
* Fix regression in 1.7.7 which raises errors when no new `demand` was present ([PR #6](https://bitbucket.org/cmutel/brightway2-calc/pull-requests/6))

### 1.7.8 (2019-11-01)

* Add check to make sure not all arrays are empty during matrix construction
* Allow numpy loading pickled data

### 1.7.7 (2019-10-31)

Switch `lca.demand` when running `.redo_lci` or `.redo_lcia`. Thanks Aleksandra Kim!

### 1.7.6 (2019-10-22)

Fixed [#25](https://bitbucket.org/cmutel/brightway2-calc/issues/25/function-load_arrays-in-utilspy-unsorted): Sort array filepaths when loading. Thanks Pedro Anchieta!

### 1.7.5 (2019-09-19)

Merged [Pull Request #4](https://bitbucket.org/cmutel/brightway2-calc/pull-requests/4/numpy-array-passthrough/diff) to directly pass Numpy or byte arrays instead of filepaths. Thanks Jan Machacek!

### 1.7.4 (2019-08-23)

* Improved support for independent LCA calculations (i.e. without Brightway2 databases, only processed arrays)
* Added ability to calculate LCAs in a single matrix (for BONSAI)

### 1.7.3 (2018-10-24)

Updated Monte Carlo for upstream presamples changes

### 1.7.2 (2018-08-21)

Merged [Pull Request #3](https://bitbucket.org/cmutel/brightway2-calc/pull-requests/3/correcting-flow-and-impact-calculations/diff) to fix some attributes in graph traversals. Thanks Bernhard Steubing!

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
