
ecopy2 Development
------------------
Package: ecopy2 (fork of ecopy by Nathan Lemoine)
Maintainer: Chad Saltikov <saltikov@ucsc.edu>
Repository: https://github.com/csaltikov/ecopy
Version: 0.1.3.0

New in 0.1.3.0:
- cap.py — CAP/dbRDA canonical ordination (McArdle & Anderson 2001)
- Fixed circular import in matrix_comp/cca.py
- Python 3.9-3.13 support
- pyproject.toml modernized, setup.py minimized

Installing ecopy2:
    pip install ecopy2
    # or
    uv pip install ecopy2

    # Development install
    git clone https://github.com/csaltikov/ecopy.git
    cd ecopy
    uv pip install -e ".[dev]"

CAP Usage:
    from ecopy2 import cap
    import pandas as pd

    result = cap(D_df, env_df, nperm=999, seed=42)
    result.summary()
    result.anova()
    result.biplot(color_by=depth_values)

CAP Implementation Notes:
- Follows McArdle & Anderson (2001) hat matrix / dbRDA approach
- NOT the Anderson & Willis (2003) LDA-based approach
- Validated against vegan::capscale() — results match to 4 decimal places
- 62 unit tests in tests/test_cap.py
- AI assistance disclosure: developed with Claude (Anthropic)
  Mathematical framework and validation by Chad Saltikov


Known Issues / TODO
-------------------
- Axis-by-axis permutation test not yet implemented (available in vegan)
- Partial CAP (conditioning variables) not yet implemented
- test_ecopy_unittest.py: test_rarefy depends on network access to
  Auerilas/ecopy-data GitHub repository


References
----------
McArdle BH, Anderson MJ (2001) Fitting multivariate models to community
    data: a comment on distance-based redundancy analysis.
    Ecology 82:290-297.

Legendre P, Anderson MJ (1999) Distance-based redundancy analysis:
    testing multispecies responses in multifactorial ecological
    experiments. Ecological Monographs 69:1-24.

Anderson MJ, Willis TJ (2003) Canonical analysis of principal
    coordinates: a useful method of constrained ordination for ecology.
    Ecology 84:511-525.

Liang KYH et al. (2021) Roseobacters in a Sea of Poly- and Paraphyly.
    Front Microbiol 12:683109.