===============
Brightway2-calc
===============

The calculation engine for the Brightway2 life cycle assessment framework.

The emphasis here has been on speed of solving the linear systems, for normal LCA calculations, graph traversal, or Monte Carlo uncertainty analysis.

The Monte Carlo LCA class can do about 30 iterations a second (on my 2011 MacBook Pro). Instead of doing LU factorization, it uses an initial guess and the conjugant gradient squared algorithm.

The multiprocessing Monte Carlo class (ParallelMonteCarlo) can do about 100 iterations a second, using 7 virtual cores. The MultiMonteCarlo class, which does Monte Carlo for many processes (and hence can re-use the factorized technosphere matrix), can do about 500 iterations a second, using 7 virtual cores. Both these algorithms perform best when the initial setup for each worker job is minimized, e.g. by dispatching big chunks.

Roadmap
*******

* 0.8: Current release. All LCA and Monte Carlo functions.
* 0.9: Documentation and inclusion of tests from Brightway 1 (updated and with coverage)
* 1.0: Bugfixes and small changes from 0.9
* 1.1: Graph traversal