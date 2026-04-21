EcoPy: Python for Ecological Data Analyses
******************************************

.. image:: https://zenodo.org/badge/17555/Auerilas/ecopy.svg
   :target: https://zenodo.org/badge/latestdoi/17555/Auerilas/ecopy

**EcoPy** provides tools for ecological data analyses. In general, it focuses
on multivariate data analysis, which can be useful in any field, but with
particular attention to those methods widely used in ecology.
`The homepage, with full documentation and examples, can be found here <http://ecopy.readthedocs.io>`_

Install via ``pip install ecopy2`` or with uv::

    uv pip install ecopy2

For development installation (editable mode)::

    git clone https://github.com/csaltikov/ecopy.git
    cd ecopy
    uv pip install -e ".[dev]"


What's New
==========

0.1.3.0
-------
- Added ``asca()`` — ANOVA-Simultaneous Component Analysis (ASCA),
  following Smilde et al. (2005) *Bioinformatics* 21:3043–3048.

  - Type I (sequential) and Type III (marginal) sums of squares decomposition
  - Type III SS is order-independent, appropriate for unbalanced designs
  - PCA on each effect matrix with score, loading, and biplot methods
  - Multiprocessing permutation testing with per-term label shuffling
  - DNA helix CLI spinner for progress feedback
  - Validated against Bertinetto, Engel & Jansen (2020)
    *Analytica Chimica Acta: X* 6:100061 example dataset

- Added ``cap()`` — Canonical Analysis of Principal Coordinates (CAP),
  also known as distance-based Redundancy Analysis (dbRDA), following
  McArdle & Anderson (2001) *Ecology* 82:290–297.

  - Accepts any user-supplied dissimilarity matrix (e.g. Bray-Curtis)
  - Supports continuous, categorical, and mixed environmental variables
  - Permutation F-test (default 999 permutations) for model significance
  - Biplot with centroids and spider legs for replicated designs
  - Validated against ``vegan::capscale()`` in R to 4 decimal places

- Fixed ``simper()``: replaced deprecated ``DataFrame.append()`` with
  ``pd.concat()``, ``normed=`` with ``density=``, ``.ix[]`` with ``.loc[]``
- Fixed ``anosim()``: replaced structured numpy array with pandas DataFrame,
  precomputed masks for performance improvement
- Fixed circular import in ``matrix_comp/cca.py``
- Updated Python classifiers: 3.9 – 3.13
- Dropped Python 2.7 and 3.4 support
- Modernized build: ``pyproject.toml``, ``importlib.metadata`` versioning,
  GitHub Actions CI

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

ANOVA-Simultaneous Component Analysis (ASCA)::

    import ecopy as ep
    import numpy as np
    import pandas as pd
    from itertools import product

    # Balanced two-factor design: depth x year, 2 replicates
    factors = pd.DataFrame({
        'depth': ['shallow'] * 6 + ['deep'] * 6,
        'year':  ['2014', '2014', '2014', '2015', '2015', '2015'] * 2,
    })

    np.random.seed(42)
    n = len(factors)
    X = np.zeros((n, 10))
    X[:, 0] = np.where(factors['depth'] == 'shallow', 8, 2)  # depth signal
    X[:, 1] = np.where(factors['depth'] == 'shallow', 6, 3)  # depth signal
    X[:, 2] = np.where(factors['year'] == '2014', 7, 3)       # year signal
    X += np.random.normal(0, 0.5, X.shape)

    # Fit ASCA model — Type III SS for unbalanced designs
    result = ep.asca(X, factors, decomp_type=3, nperm=999)
    result.summary()

    # Score plot for depth effect
    result.plot('depth', kind='scores')

    # Biplot with species loadings
    var_names = np.array(['Sp1', 'Sp2', 'Sp3', 'Sp4', 'Sp5',
                          'Sp6', 'Sp7', 'Sp8', 'Sp9', 'Sp10'])
    result.plot('depth', kind='biplot', var_names=var_names)

    # Loading bar chart
    result.plot('depth', kind='loading', var_names=var_names, n_load=10)

Full online documentation is a work in progress.


AI Assistance Disclosure
========================
The ``cap()`` and ``asca()`` implementations were developed with AI assistance
(Claude, Anthropic). The mathematical frameworks follow the references listed
below. Results were independently validated against R reference implementations
by the maintainer.

References:

- McArdle BH, Anderson MJ (2001) Fitting multivariate models to community
  data: a comment on distance-based redundancy analysis. *Ecology* 82:290–297.
- Legendre P, Anderson MJ (1999) Distance-based redundancy analysis: testing
  multispecies responses in multifactorial ecological experiments.
  *Ecological Monographs* 69:1–24.
- Smilde AK, Jansen JJ, Hoefsloot HCJ, Lamers RJAN, van der Greef J,
  Timmerman ME (2005) ANOVA-simultaneous component analysis (ASCA): a new
  tool for analyzing designed metabolomics data. *Bioinformatics* 21:3043–3048.
- Bertinetto C, Engel J, Jansen J (2020) ANOVA simultaneous component
  analysis: a tutorial review. *Analytica Chimica Acta: X* 6:100061.
- Thiel M, Féraud B, Govaerts B (2017) ASCA+ and APCA+: extensions of ASCA
  and APCA in the analysis of unbalanced multifactorial studies.
  *Chemometrics and Intelligent Laboratory Systems* 171:33–41.


TO-DO
=====

ASCA
----

1. ASCA+ full implementation for unbalanced designs (Thiel et al. 2017)
2. Projected residuals on score plots (Bertinetto et al. 2020 Fig. 2)
3. Confidence ellipses on score plots

CAP
---

1. Axis-by-axis permutation tests
2. Partial CAP (conditioning variables)

General
-------

- Incorporate DECORANA and TWINSPAN into EcoPy

  1. Modified ``write_cep`` to handle integer row names
  2. Pre-processing code works for DECORANA
  3. **Need to get decorana Fortran function working on UNIX systems**
  4. **Need to get TWINSPAN functional**

- Procrustes Rotation
- Linear/surface environmental fitting
- MaxEnt Wrapper
- Clustering
