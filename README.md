---
title: Brightway2 Calculations
---

[![pypi version](https://img.shields.io/pypi/v/bw2calc.svg)](https://pypi.org/project/bw2calc/)

[![conda-forge version](https://img.shields.io/conda/vn/conda-forge/bw2calc.svg)](https://anaconda.org/conda-forge/bw2calc)

[![bw2calc appveyor build status](https://ci.appveyor.com/api/projects/status/uqixaochulbu6vjv?svg=true)](https://ci.appveyor.com/project/cmutel/brightway2-calc)

[![Test coverage report](https://coveralls.io/repos/bitbucket/cmutel/brightway2-calc/badge.svg?branch=master)](https://coveralls.io/bitbucket/cmutel/brightway2-calc?branch=default)

This package provides the calculation engine for the [Brightway2 life
cycle assessment framework](https://brightway.dev). [Online
documentation](https://docs.brightway.dev) is available, and the source
code is hosted on
[Github](https://github.com/brightway-lca/brightway2-calc).

The emphasis here has been on speed of solving the linear systems, for
normal LCA calculations, graph traversal, or Monte Carlo uncertainty
analysis.

Relies on [bw_processing](https://github.com/brightway-lca/bw_processing)
for input array formatting.
