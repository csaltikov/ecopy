import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from warnings import warn
from collections import defaultdict

from numpy import ndarray
from sklearn.linear_model import LinearRegression


class cap(object):
    """
    Docstring for function ecopy.cap
    ====================
    Conducts Canonical Analysis of Principal Coordinates (CAP), also known
    as distance-based Redundancy Analysis (dbRDA), following McArdle &
    Anderson (2001). Accepts any user-supplied dissimilarity matrix and
    one or more continuous or categorical constraining variables.

    Unlike standard RDA (ecopy.rda), which operates on a raw species matrix,
    CAP begins from a dissimilarity matrix — allowing the use of metrics such
    as Bray-Curtis that are ecologically appropriate but not Euclidean.

    Workflow
    --------
    1. PCoA of the dissimilarity matrix (Gower 1966) retaining only
       positive-eigenvalue axes (negative eigenvalues from non-Euclidean
       metrics are dropped).
    2. Hat-matrix projection of PCoA coordinates onto the space defined
       by the environmental matrix X (ordinary least squares).
    3. SVD of the fitted (constrained) matrix → CAP axes.
    4. SVD of the residual matrix → residual PC axes (for plotting when
       fewer constrained axes than needed).
    5. Permutation F-test of overall model significance.

    Use
    ----
    cap(D, X, condition=None, scale_x=True, varNames_x=None,
        siteNames=None, pTypes=None, nperm=999, seed=42)

    Returns an object of class cap

    Parameters
    ----------
    D:
        Square symmetric dissimilarity matrix. pandas.DataFrame or
        numpy.ndarray. Typical input is a Bray-Curtis matrix computed
        from Hellinger-transformed or raw species counts.
    X:
        Environmental constraining variables. pandas.DataFrame or
        numpy.ndarray (sites x variables). May contain continuous
        (quantitative) or categorical (factor) variables. Categorical
        columns should be object/string dtype in a DataFrame and will
        be automatically dummy-coded.
    condition:
        Optional pandas.DataFrame of conditioning (nuisance) variables
        to partial out before the constrained analysis. These variables
        are removed from the PCoA coordinates via linear regression
        before CAP is applied to X. Equivalent to vegan's Condition()
        argument in capscale(). Typical use: condition=metadata[['block']]
        in a randomized block design to remove block effects before
        testing treatment. Numeric and categorical columns are both
        supported; categorical columns are dummy-coded automatically.
        Conditioning variables are never scaled.
    scale_x:
        Whether to standardize continuous columns of X to zero
        mean and unit variance before analysis (True, recommended).
        Categorical dummy columns are never scaled.
    varNames_x:
        Optional list of names for environmental variables.
        If None, taken from DataFrame columns or auto-generated.
    siteNames:
        Optional list of site/sample names. If None, taken
        from the index of D (if DataFrame) or auto-generated.
    pTypes:
        Optional list of 'q' (quantitative) or 'f' (factor) for
        each column of X. If None, inferred automatically from dtype.
    nperm:
        Number of permutations for the F-test (default 999).
    seed:
        Random seed for reproducibility (default 42).

    Attributes
    ----------
    condition_vars:
        List of conditioning variable names that were partialled out,
        or None if no condition was supplied.
    capScores:
        Site scores on constrained CAP axes (DataFrame,
        sites x CAP axes). These are your primary ordination
        coordinates for plotting.
    resScores:
        Site scores on residual (unconstrained) PC axes
        (DataFrame, sites x residual axes). Use as the Y-axis
        when only one CAP axis exists (single constraining var).
    envScores:
        Correlation of each environmental variable with each
        CAP axis (DataFrame, variables x CAP axes). Used for
        biplot arrows.
    cap_evals:
        Eigenvalues of the constrained axes (numpy array).
    res_evals:
        Eigenvalues of the residual axes (numpy array).
    total_inertia:
        Total inertia (sum of positive PCoA eigenvalues).
    R2:
        Proportion of total inertia explained by the model.
    R2adj:
        Adjusted R2 (Peres-Neto et al. 2006).
    F_stat:
        Pseudo-F statistic for the overall model.
    p_value:
        Permutation p-value for the overall model.
    df_c:
        Degrees of freedom for constrained component (rank of X).
    df_r:
        Degrees of freedom for residual component (n - rank - 1).
    imp:
        DataFrame of Std Dev, Prop. Variance, Cumulative Variance
        for all CAP + residual PC axes (mirrors ecopy.rda style).
    n_cap_axes:
        Number of constrained (CAP) axes.
    n_res_axes:
        Number of residual PC axes.
    n_neg_evals:
        Number of negative PCoA eigenvalues (non-Euclidean
        axes dropped before constrained analysis). Zero for
        Euclidean dissimilarities.
    neg_inertia:
        Sum of |λ| over the negative PCoA eigenvalues.
    neg_fraction:
        neg_inertia / Σ|λ_all|. Fraction of total eigenvalue
        mass on non-Euclidean axes. A useful diagnostic for
        how far the input dissimilarity departs from Euclidean
        geometry — values >0.1 suggest meaningful distortion
        and you may want to consider sqrt-transforming D or
        applying a Cailliez/Lingoes correction upstream.

    Methods
    -------
    summary():
        Prints a text summary of constrained and residual inertia, R2,
        adjusted R2, the permutation F-test result, and variance
        explained per axis.

    anova(nperm=None):
        Runs (or re-runs) the permutation F-test. Returns F statistic
        and p-value. nperm overrides the value set at construction.

    biplot(xax=1, yax=2, color_by=None, markers=None,
           arrow_scale=0.8, site_labels=True, figsize=(7,6), **kwargs):
        Produces a CAP biplot with site scores and environmental
        variable arrows. When only one CAP axis exists, yax
        automatically uses the first residual PC.

        xax: Integer, which CAP axis on x (default 1).
        yax: Integer, which CAP/residual axis on y (default 2).
        color_by: Optional array-like of length n for point colors.
        markers: Optional array-like of length n for point markers.
        arrow_scale: Scale factor for biplot arrows (default 0.8).
        site_labels: Whether to annotate site names (default True).
        figsize: Figure size tuple.
        kwargs:
            f_name: string or path to save the biplot
        return:
            fig, ax

    Example
    -------
    import ecopy as ep
    import numpy as np
    import pandas as pd

    # Species count matrix (sites x species)
    BCI = ep.load_data('BCI')

    # Bray-Curtis dissimilarity
    D = ep.distance(BCI, method='bray')

    # Environmental variables
    env = pd.DataFrame({
        'Depth': [5, 10, 15, 20, 25, 30],
        'pH':    [8.1, 8.3, 8.5, 8.7, 8.8, 8.9]
    })

    # Run CAP
    result = ep.cap(D, env)
    print(result.summary())
    result.anova()
    fig, ax = result.biplot()

    References
    ----------
    McArdle BH, Anderson MJ (2001) Fitting multivariate models to
        community data: a comment on distance-based redundancy analysis.
        Ecology 82:290-297.
    Legendre P, Anderson MJ (1999) Distance-based redundancy analysis:
        testing multispecies responses in multifactorial ecological
        experiments. Ecol Monogr 69:1-24.
    Peres-Neto PR et al. (2006) Variation partitioning of species data
        matrices: estimation and comparison of fractions. Ecology 87:2614-2625.
    """

    def __init__(self, D, X, condition=None, scale_x=True, varNames_x=None,
                 siteNames=None, pTypes=None, nperm=999, seed=42):

        # ---- Input validation ----
        if not isinstance(D, (pd.DataFrame, np.ndarray)):
            raise ValueError('D must be a pandas.DataFrame or numpy.ndarray')
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError('X must be a pandas.DataFrame or numpy.ndarray')

        D_arr = np.array(D, dtype='float')
        if D_arr.ndim != 2 or D_arr.shape[0] != D_arr.shape[1]:
            raise ValueError('D must be a square matrix')
        if not np.allclose(D_arr, D_arr.T, atol=1e-8):
            raise ValueError('D must be symmetric')
        if np.any(D_arr < 0):
            raise ValueError('D cannot contain negative values')
        if np.isnan(D_arr).any():
            raise ValueError('D contains NaN values')

        n = D_arr.shape[0]

        # Site names
        if siteNames is not None:
            self.siteNames = list(siteNames)
        elif isinstance(D, pd.DataFrame):
            self.siteNames = list(D.index)
        else:
            self.siteNames = ['Site {}'.format(i) for i in range(1, n + 1)]

        # ---- Process environmental matrix ----
        if isinstance(X, pd.DataFrame):
            X_mat, varNames_x, pTypes = _dummy_matrix(X, scale_x)
        elif isinstance(X, np.ndarray):
            if X.dtype == object:
                raise ValueError('numpy.ndarray X cannot have object dtype; use a DataFrame for categorical variables')
            if np.isnan(X).any():
                raise ValueError('X contains NaN values')
            X_mat = X.astype(float).copy()
            if varNames_x is None:
                varNames_x = ['Pred {}'.format(i) for i in range(1, X_mat.shape[1] + 1)]
            if pTypes is None:
                pTypes = ['q'] * X_mat.shape[1]
            if scale_x:
                col_std = X_mat.std(axis=0, ddof=1)
                col_std[col_std == 0] = 1.0
                X_mat = (X_mat - X_mat.mean(axis=0)) / col_std

        if X_mat.shape[0] != n:
            raise ValueError('D and X must have the same number of rows (sites)')

        self.varNames_x = list(varNames_x)
        self.pTypes = list(pTypes)
        self._nperm = nperm
        self._seed = seed
        self._X_mat = X_mat

        # ---- Validate and process condition matrix ----
        if condition is not None:
            if not isinstance(condition, pd.DataFrame):
                raise ValueError('condition must be a pandas.DataFrame')
            if condition.shape[0] != n:
                raise ValueError('condition must have the same number of rows as D')
            cond_mat, cond_names, _ = _dummy_matrix(condition, scale=False)
            self.condition_vars = cond_names
        else:
            cond_mat = None
            self.condition_vars = None

        # ---- Step 1: PCoA of dissimilarity matrix ----
        F_pcoa, evals_pos, evals_all = _pcoa(D_arr)
        self.total_inertia = float(evals_pos.sum())
        self._evals_pcoa = evals_pos

        # ---- PCoA non-Euclidean diagnostics ----
        neg_evals = evals_all[evals_all < -1e-10]
        self.n_neg_evals = int(len(neg_evals))
        self.neg_inertia = float(np.abs(neg_evals).sum())
        total_abs = float(np.abs(evals_all).sum())
        self.neg_fraction = self.neg_inertia / total_abs if total_abs > 0 else 0.0

        # ---- Step 1b: Partial out condition variables (if supplied) ----
        # Regress each PCoA axis on the condition matrix and retain residuals.
        # This removes nuisance variance (e.g. block effects) from the
        # coordinate space before constraining on X — equivalent to vegan's
        # Condition() / partial dbRDA.
        if cond_mat is not None:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(cond_mat, F_pcoa)
            F_pcoa = F_pcoa - reg.predict(cond_mat)
            # Recompute total inertia in the partialled space so that R² and
            # F are expressed relative to the variance available after
            # conditioning — consistent with vegan's capscale() behaviour.
            _, evals_partial, _ = _pcoa(
                _euclidean_dist(F_pcoa)
            )
            self.total_inertia = float(evals_partial.sum())
            F_pcoa_analysis = F_pcoa
        else:
            F_pcoa_analysis = F_pcoa

        self._F_pcoa = F_pcoa_analysis   # used for permutation test

        # ---- Step 2–4: Hat matrix projection + SVD ----
        result = _dbrda(F_pcoa_analysis, X_mat)

        self.cap_evals  = result['cap_evals']
        self.res_evals  = result['res_evals']
        self.n_cap_axes = len(self.cap_evals)
        self.n_res_axes = len(self.res_evals)
        self.R2         = result['R2']
        self.R2adj      = 1.0 - (1.0 - self.R2) * (n - 1) / max(n - result['rank'] - 1, 1)
        self.F_stat     = result['F_stat']
        self.df_c       = result['df_c']
        self.df_r       = result['df_r']

        # ---- Site scores ----
        cap_names = ['CAP {}'.format(i) for i in range(1, self.n_cap_axes + 1)]
        res_names = ['ResPC {}'.format(i) for i in range(1, self.n_res_axes + 1)]

        self.capScores = pd.DataFrame(result['cap_scores'],
                                   index=self.siteNames,
                                   columns=cap_names)
        self.resScores = pd.DataFrame(result['res_scores'],
                                   index=self.siteNames,
                                   columns=res_names)

        # ---- Environmental correlations (biplot arrows) ----
        # Correlate each env variable with each CAP axis
        env_corr = np.zeros((len(self.varNames_x), self.n_cap_axes))
        for i in range(X_mat.shape[1]):
            for j in range(self.n_cap_axes):
                env_corr[i, j] = np.corrcoef(
                    X_mat[:, i], result['cap_scores'][:, j])[0, 1]
        self.envScores = pd.DataFrame(env_corr,
                                   index=self.varNames_x,
                                   columns=cap_names)

        # ---- Importance table ----
        all_evals = np.concatenate([self.cap_evals, self.res_evals])
        all_evals_pos = all_evals[all_evals > 0]
        sds   = np.sqrt(all_evals_pos)
        props = all_evals_pos / self.total_inertia
        cums  = np.cumsum(all_evals_pos) / self.total_inertia
        all_names = cap_names + res_names[:len(self.res_evals[self.res_evals > 0])]
        self.imp = pd.DataFrame(
            np.vstack([sds, props, cums]),
            index=['Std Dev', 'Prop Var', 'Cum Var'],
            columns=all_names[:len(all_evals_pos)]
        )

        # ---- Permutation test ----
        self.p_value = self._permtest(nperm, seed)

    # ------------------------------------------------------------------
    def summary(self):
        """Print a formatted summary of the CAP analysis."""
        sep = '-' * 52
        print('\nCAP / dbRDA Summary')
        print(sep)
        print('  Samples (n):        {}'.format(len(self.siteNames)))
        if self.condition_vars:
            print('  Conditioned on:     {}'.format(', '.join(self.condition_vars)))
        print('  Constraining vars:  {}'.format(len(self.varNames_x)))
        print('  CAP axes:           {}'.format(self.n_cap_axes))
        print(sep)
        print('  Total inertia:      {:.4f}'.format(self.total_inertia))
        print('  Constrained:        {:.4f}  ({:.1f}%)'.format(
            self.cap_evals.sum(),
            self.cap_evals.sum() / self.total_inertia * 100))
        print('  Residual:           {:.4f}  ({:.1f}%)'.format(
            self.res_evals.sum(),
            self.res_evals.sum() / self.total_inertia * 100))
        print(sep)
        print('  R²:                 {:.4f}'.format(self.R2))
        print('  R²adj:              {:.4f}'.format(self.R2adj))
        print('  F({},{}):           {:.3f}'.format(
            self.df_c, self.df_r, self.F_stat))
        print('  p-value ({} perm): {}'.format(
            self._nperm,
            '<0.001' if self.p_value < 0.001 else '{:.3f}'.format(self.p_value)))
        if self.n_neg_evals > 0:
            print(f'  PCoA neg. eigenvalues: {self.n_neg_evals} ({self.neg_fraction:0.2%} of |inertia|)')
            print('   (dissimilarity is non-Euclidean; negative axes dropped)')
        print(sep)
        print('\nVariance per axis:')
        print(self.imp.round(4).to_string())
        print('\nEnvironmental correlations with CAP axes:')
        print(self.envScores.round(4).to_string())
        print()

    # ------------------------------------------------------------------
    def anova(self, nperm=None):
        """
        Run (or re-run) the permutation F-test.

        Parameters
        ----------
        nperm: int or None. If None, uses the value from construction.

        Returns
        -------
        dict with keys 'F_stat', 'p_value', 'nperm'
        """
        if nperm is None:
            nperm = self._nperm
        p = self._permtest(nperm, self._seed)
        self.p_value = p
        print('Model F({},{}) = {:.3f}'.format(self.df_c, self.df_r, self.F_stat))
        print('p = {} ({} permutations)'.format(
            '<0.001' if p < 0.001 else '{:.3f}'.format(p), nperm))
        return {'F_stat': self.F_stat, 'p_value': p, 'nperm': nperm}

    # ------------------------------------------------------------------
    def biplot(self, xax=1, yax=2, color_by: list|ndarray=None, markers: list|ndarray =None,
               arrow_scale=0.8, site_labels=True, figsize=(7, 6), **kwargs):
        """
        Produce a CAP biplot with site scores and environmental arrows.

        Parameters
        ----------
        xax:
            int — which CAP axis on x-axis (1-indexed, default 1).
        yax: int — which CAP/ResPC axis on y-axis (1-indexed, default 2).
             If fewer than 2 CAP axes exist, yax automatically refers to
             the first residual PC axis regardless of this value.
        color_by:
            array-like of length n, used to color points. If numeric,
             a colormap is applied. If categorical, unique colors per group.
        markers:
            array-like of length n, matplotlib marker strings per site.
        arrow_scale:
            float, scale factor for biplot arrows (default 0.8).
        site_labels:
            bool, whether to annotate site names (default True).
        figsize:
            tuple, figure size (default (7, 6)).
        kwargs: str|Path
            f_name: this is string/path for saving the biplot

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object. Use these to further customize the plot
            before calling plt.show() or fig.savefig().
        """
        xi = xax - 1

        # Determine y-axis source (constrained or residual)
        if self.n_cap_axes >= 2:
            x_scores = self.capScores.iloc[:, xi].values
            y_scores = self.capScores.iloc[:, yax - 1].values
            x_label  = self.capScores.columns[xi]
            y_label  = self.capScores.columns[yax - 1]
            # env arrow correlations
            arrow_x = self.envScores.iloc[:, xi].values
            arrow_y = self.envScores.iloc[:, yax - 1].values
        else:
            # Single CAP axis — use ResPC1 as Y
            x_scores = self.capScores.iloc[:, 0].values
            y_scores = self.resScores.iloc[:, 0].values
            x_label  = self.capScores.columns[0]
            y_label  = self.resScores.columns[0] + ' (unconstrained)'
            arrow_x = self.envScores.iloc[:, 0].values
            # correlation of env vars with ResPC1
            arrow_y = np.array([
                np.corrcoef(self._X_mat[:, i], y_scores)[0, 1]
                for i in range(self._X_mat.shape[1])
            ])
            if yax > 1:
                warn('Only 1 CAP axis exists; y-axis shows ResPC1 (residual PC).')

        # Add % inertia to axis labels
        if xi < len(self.cap_evals):
            pct_x = self.cap_evals[xi] / self.total_inertia * 100
            x_label += ' ({:.1f}%)'.format(pct_x)
        if self.n_cap_axes >= 2 and (yax - 1) < len(self.cap_evals):
            pct_y = self.cap_evals[yax - 1] / self.total_inertia * 100
            y_label_base = self.capScores.columns[yax - 1]
            y_label = y_label_base + ' ({:.1f}%)'.format(pct_y)

        n = len(self.siteNames)

        # ---- Colors ----
        if color_by is not None:
            color_by = np.array(color_by)
            if np.issubdtype(color_by.dtype, np.number):
                import matplotlib.cm as mcm
                import matplotlib.colors as mcolors
                cmap = mcm.viridis
                norm = mcolors.Normalize(vmin=color_by.min(), vmax=color_by.max())
                colors = [cmap(norm(v)) for v in color_by]
            else:
                uniq = list(dict.fromkeys(color_by))
                cmap_cat = plt.cm.tab10
                color_map = {u: cmap_cat(i / max(len(uniq) - 1, 1))
                             for i, u in enumerate(uniq)}
                colors = [color_map[v] for v in color_by]
        else:
            colors = ['steelblue'] * n

        # ---- Markers ----
        if markers is not None:
            markers = list(markers)
        else:
            markers = ['o'] * n

        # ---- Arrow scale ----
        # max_score = max(np.abs(x_scores).max(), np.abs(y_scores).max())
        # sc = max_score * arrow_scale
        x_range = x_scores.max() - x_scores.min()
        y_range = y_scores.max() - y_scores.min()
        sc_x = x_range * arrow_scale * 0.5
        sc_y = y_range * arrow_scale * 0.5

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')

        # Site scores
        unique_markers = list(dict.fromkeys(markers))
        for mk in unique_markers:
            idx_mk = [i for i, m in enumerate(markers) if m == mk]
            ax.scatter(x_scores[idx_mk], y_scores[idx_mk],
                       c=[colors[i] for i in idx_mk],
                       marker=mk, s=70, edgecolors='white',
                       linewidths=0.5, zorder=3)

        if site_labels:
            _label_sites(x_scores, y_scores, self.siteNames, ax)

        # Biplot arrows (quantitative variables only)
        for i, (vname, ptype) in enumerate(zip(self.varNames_x, self.pTypes)):
            if ptype == 'q':
                ax.annotate('',
                            xy=(arrow_x[i] * sc_x, arrow_y[i] * sc_y),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->',
                                            color='firebrick',
                                            lw=1.8,
                                            mutation_scale=14))
                ax.text(arrow_x[i] * sc_x * 1.15,
                        arrow_y[i] * sc_y * 1.15,
                        vname, color='firebrick',
                        fontsize=9, fontweight='bold',
                        ha='center', va='center')
            elif ptype == 'f':
                # Factor centroids shown as text
                ax.text(arrow_x[i] * sc_x, arrow_y[i] * sc_y,
                        '[' + vname + ']',
                        color='darkgreen', fontsize=9,
                        ha='center', va='center')

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title('CAP Biplot\nR²={:.3f}  R²adj={:.3f}  F({},{})={:.2f}  {}'.format(
            self.R2, self.R2adj, self.df_c, self.df_r, self.F_stat,
            'p<0.001' if self.p_value < 0.001 else 'p={:.3f}'.format(self.p_value)),
            fontsize=10)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        f_name = kwargs.get("f_name", None)
        if f_name:
            plt.savefig(f_name, dpi=300, bbox_inches='tight')
        return fig, ax

    # ------------------------------------------------------------------
    def _permtest(self, nperm, seed):
        """Permutation F-test: shuffle rows of X, refit, compare F."""
        np.random.seed(seed)
        n = self._F_pcoa.shape[0]
        perm_F = np.empty(nperm)
        for i in range(nperm):
            idx = np.random.permutation(n)
            r = _dbrda(self._F_pcoa, self._X_mat[idx])
            perm_F[i] = r['F_stat']
        p = (np.sum(perm_F >= self.F_stat) + 1) / (nperm + 1)
        return float(p)


# ======================================================================
# Module-level helper functions (private)
# ======================================================================

def _euclidean_dist(coords):
    """Compute Euclidean distance matrix from coordinate matrix."""
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(coords, metric='euclidean'))


def _pcoa(D):
    """
    Principal Coordinates Analysis (Gower 1966).
    Returns F (site scores, n x k) and positive eigenvalues.
    Only positive-eigenvalue axes are retained.
    """
    n = D.shape[0]
    A = -0.5 * D ** 2
    ones = np.ones(n)
    # Double centering
    row_mean   = A.mean(axis=1, keepdims=True)
    col_mean   = A.mean(axis=0, keepdims=True)
    grand_mean = A.mean()
    G = A - row_mean - col_mean + grand_mean

    eigenvalues, eigenvectors = np.linalg.eigh(G)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Retain only positive eigenvalues
    pos = eigenvalues > 1e-10
    evals_pos = eigenvalues[pos]
    evecs_pos = eigenvectors[:, pos]
    F = evecs_pos * np.sqrt(evals_pos)
    return F, evals_pos, eigenvalues


def _dbrda(F, X):
    """
    Core dbRDA computation (McArdle & Anderson 2001).
    Projects PCoA coordinates F onto the subspace defined by X
    via the hat matrix, then decomposes via SVD.

    Returns dict with cap_scores, res_scores, eigenvalues, R2, F_stat etc.
    """
    n = F.shape[0]

    # Hat matrix via QR decomposition
    Q, _ = np.linalg.qr(X)
    rank = int(np.linalg.matrix_rank(X))
    Q = Q[:, :rank]

    # Constrained (fitted) and residual components
    F_hat = Q @ Q.T @ F
    F_res = F - F_hat

    # SVD of constrained part
    U_c, s_c, _ = np.linalg.svd(F_hat, full_matrices=False)
    keep_c = s_c > 1e-10
    s_c = s_c[keep_c];  U_c = U_c[:, keep_c]
    cap_evals = s_c ** 2
    constrained = cap_evals.sum()

    # SVD of residual part
    U_r, s_r, _ = np.linalg.svd(F_res, full_matrices=False)
    keep_r = s_r > 1e-10
    s_r = s_r[keep_r];  U_r = U_r[:, keep_r]
    res_evals = s_r ** 2
    residual = res_evals.sum()

    total = constrained + residual
    R2    = constrained / total if total > 0 else 0.0

    df_c  = rank
    df_r  = n - rank - 1
    F_stat = (constrained / df_c) / (residual / df_r) if df_r > 0 and residual > 0 else np.nan

    return dict(
        cap_scores  = U_c * s_c,      # n x n_cap_axes
        res_scores  = U_r * s_r,      # n x n_res_axes
        cap_evals   = cap_evals,
        res_evals   = res_evals,
        constrained = constrained,
        residual    = residual,
        R2          = R2,
        F_stat      = F_stat,
        df_c        = df_c,
        df_r        = df_r,
        rank        = rank,
    )


def _dummy_matrix(X_df, scale):
    """
    Convert a DataFrame with mixed numeric/categorical columns into a
    design matrix. Categorical columns are dummy-coded (one-hot, dropping
    first level to avoid multicollinearity). Numeric columns are optionally
    standardized.

    Returns (X_mat, varNames, pTypes)
    """
    rows = X_df.shape[0]
    X_mat    = np.zeros((rows, 0))
    varNames = []
    pTypes   = []

    for col in X_df.columns:
        series = X_df[col]

        # Try numeric coercion first — handles object-dtype columns that
        # contain numeric strings (e.g. from QIIME2 metadata TSV files
        # where the #q2:types row causes everything to be read as object).
        coerced = pd.to_numeric(series, errors='coerce')
        is_numeric = pd.api.types.is_numeric_dtype(series) or not coerced.isna().any()

        if is_numeric:
            vals = coerced.values.astype(float)
            if scale:
                std = vals.std(ddof=1)
                if std > 0:
                    vals = (vals - vals.mean()) / std
                else:
                    vals = vals - vals.mean()
            X_mat = np.hstack([X_mat, vals.reshape(-1, 1)])
            varNames.append(col)
            pTypes.append('q')
        else:
            # Genuinely categorical — dummy code, drop first level
            dummies = pd.get_dummies(series, drop_first=True)
            level_names = ['{}: {}'.format(col, lv) for lv in dummies.columns]
            X_mat = np.hstack([X_mat, dummies.values.astype(float)])
            varNames.extend(level_names)
            pTypes.extend(['f'] * len(level_names))

    return X_mat, varNames, pTypes


def _label_sites(scores_x, scores_y, site_names, ax):
    """
    Place site labels on an ordination biplot with edge-aware offsets and
    grouping for near-coincident points.

    Points within ~3% of the larger axis range of each other are grouped and
    their labels stacked vertically. Labels for points in the outer 25% of
    either axis are flipped to stay inside the plot region.

    Parameters
    ----------
    scores_x, scores_y : array-like
        Site coordinates on the x and y axes of the biplot.
    site_names : sequence of str
        Labels for each site, same length as scores_x and scores_y.
    ax : matplotlib Axes
        The axis to annotate.

    Returns
    -------
    ax : matplotlib Axes
        The same axis, with annotations added.
    """
    scores_x = np.asarray(scores_x)
    scores_y = np.asarray(scores_y)

    x_min, x_max = float(scores_x.min()), float(scores_x.max())
    y_min, y_max = float(scores_y.min()), float(scores_y.max())
    x_range = x_max - x_min
    y_range = y_max - y_min

    # A point is "near an edge" if it sits in the outer 25% of the axis span.
    edge_frac = 0.25
    x_right_edge = x_max - edge_frac * x_range
    y_top_edge   = y_max - edge_frac * y_range

    # Group points within ~3% of the larger axis range (Euclidean distance).
    tol = 0.03 * max(x_range, y_range)
    tol_sq = tol * tol
    n = len(scores_x)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            dx = scores_x[i] - scores_x[j]
            dy = scores_y[i] - scores_y[j]
            if dx * dx + dy * dy < tol_sq:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    for indices in groups.values():
        cx = float(np.mean([scores_x[i] for i in indices]))
        cy = float(np.mean([scores_y[i] for i in indices]))

        if cx > x_right_edge:
            sign_x, ha = -1, 'right'
        else:
            sign_x, ha = 1, 'left'

        if cy > y_top_edge:
            sign_y, va = -1, 'top'
        else:
            sign_y, va = 1, 'bottom'

        if len(indices) == 1:
            i = indices[0]
            ax.annotate(site_names[i],
                        (scores_x[i], scores_y[i]),
                        fontsize=8, color='#333333',
                        xytext=(sign_x * 4, sign_y * 3),
                        textcoords='offset points',
                        ha=ha, va=va)
        else:
            for k, i in enumerate(indices):
                ax.annotate(site_names[i],
                            (scores_x[i], scores_y[i]),
                            fontsize=7, color='#333333',
                            xytext=(sign_x * 6, sign_y * (3 + k * 10)),
                            textcoords='offset points',
                            ha=ha, va=va)
    return ax
