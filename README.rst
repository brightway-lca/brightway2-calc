Brightway2 calculations
=======================

This package provides the calculation engine for the `Brightway2 life cycle assessment framework <https://brightwaylca.org>`_. `Online documentation <https://docs.brightwaylca.org/>`_ is available, and the source code is hosted on `Bitucket <https://bitbucket.org/cmutel/brightway2-calc>`_.

The emphasis here has been on speed of solving the linear systems, for normal LCA calculations, graph traversal, or Monte Carlo uncertainty analysis.

The Monte Carlo LCA class can do about 30 iterations a second (on a 2011 MacBook Pro). Instead of doing LU factorization, it uses an initial guess and the conjugant gradient squared algorithm.

The multiprocessing Monte Carlo class (ParallelMonteCarlo) can do about 100 iterations a second, using 7 virtual cores. The MultiMonteCarlo class, which does Monte Carlo for many processes (and hence can re-use the factorized technosphere matrix), can do about 500 iterations a second, using 7 virtual cores. Both these algorithms perform best when the initial setup for each worker job is minimized, e.g. by dispatching big chunks.
