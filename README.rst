EcoPy: Python for Ecological Data Analyses
******************************************

.. image:: https://zenodo.org/badge/17555/Auerilas/ecopy.svg
   :target: https://zenodo.org/badge/latestdoi/17555/Auerilas/ecopy

**EcoPy** provides tools for ecological data analyses. In general, it focuses
on multivariate data analysis, which can be useful in any field, but with
particular attention to those methods widely used in ecology.
`The homepage, with full documentation and examples, can be found here <http://ecopy.readthedocs.io>`_

Install via ``pip install ecopy`` or with uv::

    uv pip install ecopy

For development installation (editable mode)::

    git clone https://github.com/csaltikov/ecopy.git
    cd ecopy
    uv pip install -e ".[dev]"


What's New
==========

0.1.3.0
-------
- Added ``cap()`` — Canonical Analysis of Principal Coordinates (CAP),
  also known as distance-based Redundancy Analysis (dbRDA), following
  McArdle & Anderson (2001) *Ecology* 82:290–297.

  - Accepts any user-supplied dissimilarity matrix (e.g. Bray-Curtis)
  - Supports continuous, categorical, and mixed environmental variables
  - Permutation F-test (default 999 permutations) for model significance
  - Biplot with centroids and spider legs for replicated designs
  - Validated against ``vegan::capscale()`` in R

- Fixed circular import in ``matrix_comp/cca.py``
- Updated Python classifiers: 3.9 – 3.13
- Dropped Python 2.7 and 3.4 support

0.1.2.3
-------
- Fixed compatibility problems in functions ``cca()``, ``simper()``, and ``transform()``
- Added Ochiai's binary coefficient to ``distance()`` function
- Added ``bioenv()`` function

0.1.2.2
-------
- More Python 3.x compatibility
- Fix typos in code and examples on readthedocs. Thorough code check


License
=======
**EcoPy** is distributed under the MIT license


Version
=======
0.1.3.0


Examples
========

Transforming a site x species matrix, dividing by site totals::

    import ecopy as ep
    varespec = ep.load_data('varespec')
    newMat = ep.transform(varespec, method='total', axis=1)

Calculating Bray-Curtis dissimilarities on the new matrix::

    brayMat = ep.distance(newMat, method='bray')

PCA on US Arrests data::

    USArrests = ep.load_data('USArrests')
    prcomp = ep.pca(USArrests, scale=True)
    prcomp.biplot(type='distance')
    prcomp.biplot(type='correlation')

Canonical Analysis of Principal Coordinates (CAP / dbRDA)::

    import ecopy as ep
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform

    # Synthetic community: 12 sites x 20 species with a depth gradient
    np.random.seed(42)
    n_sites, n_spp = 12, 20
    depths = np.array([5, 5, 5, 10, 10, 10, 20, 20, 20, 30, 30, 30],
                      dtype=float)
    counts = np.zeros((n_sites, n_spp))
    for i in range(n_sites):
        counts[i, :10] = np.random.poisson(max(20 - depths[i], 2), 10)
        counts[i, 10:] = np.random.poisson(max(depths[i] - 5,  2), 10)

    site_names = [f'Site{i+1}' for i in range(n_sites)]

    # Bray-Curtis dissimilarity matrix (n_sites x n_sites)
    D = pd.DataFrame(
        squareform(pdist(counts + 1, metric='braycurtis')),
        index=site_names, columns=site_names
    )

    # Environmental variables — one row per site, aligned to D
    env = pd.DataFrame({
        'Depth': depths,
        'Temp':  [25, 25, 25, 22, 22, 22, 15, 15, 15, 10, 10, 10],
    }, index=site_names)

    # Fit CAP model
    result = ep.cap(D, env, nperm=999, seed=42)
    result.summary()
    result.anova()
    result.biplot(color_by=env['Depth'].values)

Full online documentation is a work in progress.


AI Assistance Disclosure
========================
The ``cap()`` implementation was developed with AI assistance (Claude, Anthropic).
The mathematical framework follows McArdle & Anderson (2001) and Legendre &
Anderson (1999). Results were independently validated against ``vegan::capscale()``
in R by the maintainer.

References:

- McArdle BH, Anderson MJ (2001) Fitting multivariate models to community
  data: a comment on distance-based redundancy analysis. *Ecology* 82:290–297.
- Legendre P, Anderson MJ (1999) Distance-based redundancy analysis: testing
  multispecies responses in multifactorial ecological experiments.
  *Ecological Monographs* 69:1–24.


TO-DO
=====

Incorporate DECORANA and TWINSPAN into EcoPy
--------------------------------------------

1. Modified ``write_cep`` to handle integer row names (common in pandas DataFrames)
2. Pre-processing code works for DECORANA
3. **Need to get decorana Fortran function working on UNIX systems (.exe binary only works for Windows)**

   - This is going to be difficult because converting the decorana function to Python
     pulls in the numerical subroutines. There is, as yet, no way to simply pass the
     data file to a terminal command like used in CornPy. The Fortran subroutines
     would need to be rewritten, as ``vegan`` does.

4. **Need to get TWINSPAN functional**

Procrustes Rotation
-------------------

Linear/surface environmental fitting
-------------------------------------

Axis-by-axis permutation tests for CAP
---------------------------------------

MaxEnt Wrapper
--------------

Clustering
----------
