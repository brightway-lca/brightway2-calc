Brightway2 calculations
=======================

.. image:: https://ci.appveyor.com/api/projects/status/uqixaochulbu6vjv?svg=true
   :target: https://ci.appveyor.com/project/cmutel/brightway2-calc
   :alt: bw2calc appveyor build status

.. image:: https://coveralls.io/repos/bitbucket/cmutel/brightway2-calc/badge.svg?branch=master
    :target: https://coveralls.io/bitbucket/cmutel/brightway2-calc?branch=default
    :alt: Test coverage report

This package provides the calculation engine for the `Brightway2 life cycle assessment framework <https://brightway.dev>`_. `Online documentation <https://docs.brightway.dev>`_ is available, and the source code is hosted on `Github <https://github.com/brightway-lca/brightway2-calc>`_.

The emphasis here has been on speed of solving the linear systems, for normal LCA calculations, graph traversal, or Monte Carlo uncertainty analysis.

Relies on `bw_processing <https://github.com/brightway-lca/bw_processing>`__ for input array formatting.
