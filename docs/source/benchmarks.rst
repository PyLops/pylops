.. _benchmarks:

|:alarm_clock:| Benchmarks
##########################

PyLops tracks the performance of all its operators with
`airspeed velocity <https://asv.readthedocs.io>`_ (asv). For every operator, the
suite measures the wall time and the peak memory of the forward (``matvec``) and
adjoint (``rmatvec``) passes.

The benchmark suite lives in a dedicated repository,
`PyLops/pylops-asv <https://github.com/PyLops/pylops-asv>`_. A GitHub Action in that
repository benchmarks the heads of the ``master`` and ``dev`` branches every night
(as well as the latest release), and the resulting website is published at
https://pylops.github.io/pylops-asv/. Older releases can be benchmarked retroactively
by triggering the same workflow manually.

When adding a new operator to PyLops, a companion pull request adding its benchmark
to ``pylops-asv`` is expected (see the :ref:`addingoperator` checklist). The
``README`` of ``pylops-asv`` explains how to write a benchmark, validate the suite
and run it locally against a local checkout of PyLops.
