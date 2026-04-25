from pathlib import Path
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.decomposition import PCA
from formulaic import model_matrix
import re
import matplotlib.pyplot as plt
from ecopy.matrix_comp.asca_permute import run_permutations, _decompose_standalone


MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '*', 'p']


class ASCA:
    """ANOVA-Simultaneous Component Analysis (ASCA) for multivariate data.

    ASCA decomposes a multivariate data matrix X into contributions from
    experimental factors and their interactions, then applies PCA to each
    effect matrix to extract the dominant patterns of variation attributable
    to each factor.

    The decomposition follows:

        X = grand_mean + X_A + X_B + X_AB + X_residual

    where X_A and X_B are the pure effect matrices for factors A and B,
    X_AB is the interaction effect, and X_residual contains unexplained
    variance. PCA is then applied to each effect matrix separately.

    This is particularly useful for microbial community datasets with
    structured experimental designs (e.g., multiple sampling depths,
    time points, or geochemical gradients) where standard PCA would
    confound multiple sources of variation.

    Args:
        X (np.ndarray): X is an n-dimensional multivariate data matrix, shape (n_samples, n_features).
        factors (pd.DataFrame): Experimental design matrix, shape (n_samples, n_factors).
            Each column is a categorical factor (e.g., 'depth', 'season').
        decomp_type (int): Decomposition method for effect matrix estimation.
            1 = Type I (sequential) SS — effects estimated in model entry order,
                each term adjusted only for prior terms. Order-dependent.
            3 = ASCA+ (Thiel et al. 2017) — Type III marginal SS via full-minus-reduced
                least squares projection. Each effect matrix estimated by subtracting
                the reduced model fit (all other terms) from the full model fit.
                Order-independent; preferred for unbalanced designs. Residuals
                projected back onto effect matrices prior to PCA. Defaults to 1.
        nperm (int): Number of permutations for significance testing. Defaults to 999.
        verbose (bool): If True, print progress during decomposition and permutation
            testing. Defaults to False.

    Attributes:
        grand_mean (np.ndarray): Column-wise grand mean of X, shape (n_features,).
        X_centered (np.ndarray): Mean-centered data matrix, shape (n_samples, n_features).
        effect_matrices (dict[str, np.ndarray]): Effect matrices keyed by
            factor name (e.g., 'depth', 'time', 'interaction').
        SS (dict[str, float]): Sum of squares for each model term, keyed by factor name.
        SS_total (float): Total sum of squares of X_centered.
        SS_residual (float): Residual sum of squares after removing all factor effects.
        E (np.ndarray): Residual matrix after factor effect removal, shape (n_samples, n_features).
            In ASCA+, projected back onto effect matrices before PCA.
        pcas (dict[str, PCA]): ...
        scores (dict[str, np.ndarray]): PCA scores for each effect matrix.
        loadings (dict[str, np.ndarray]):  PCA loadings for each effect matrix.
        pvalues (dict[str, float]): Permutation-derived p-values for each model term
        perm_ss (dict[str, np.ndarray]): ...

    Raises:
        ValueError: If decomp_type is not 1 or 3.

    Notes:
        When decomp_type=1, the order of columns in `factors` determines
        the sequential partitioning of variance. For unbalanced designs or
        when factor order is arbitrary, prefer decomp_type=3.

    References:
        Smilde, A.K., et al. (2005). ANOVA-simultaneous component analysis
        (ASCA): a new tool for analyzing designed metabolomics data.
        Bioinformatics, 21(13), 3043-3048.

        Vis, D.J., et al. (2007). Statistical validation of megavariate effects
        in ASCA. BMC Bioinformatics, 8, 322.
    """
    def __init__(self, X, factors, decomp_type=1, nperm=999, verbose=False):
        self.X = X
        self.factors = factors
        self.verbose = verbose
        self.decomp_type = decomp_type

        if self.decomp_type not in (1, 3):
            raise ValueError(f"decomp_type must be 1 or 3, got {self.decomp_type}")

        self.effect_matrices, self.SS, self.E, \
            self.grand_mean, self.X_centered, \
            self.SS_total, self.SS_residual = \
            self._decompose(self.X, self.factors)

        self.pcas = {}
        self.scores = {}
        self.loadings = {}

        for term, mat in self.effect_matrices.items():
            p = PCA()
            self.scores[term] = p.fit_transform(mat)
            self.loadings[term] = p.components_.T
            self.pcas[term] = p

        self.nperm = nperm
        self.pvalues, self.perm_ss = run_permutations(
            self.SS, self.factors, self.X, self.decomp_type, self.nperm)

    def _get_term(self, col):
        """Extract term name from a patsy column name"""
        return ":".join(part.split('[')[0] for part in col.split(':'))

    def _get_effect_cols(self, design, term):
        """Get all design matrix columns belonging to exactly this term"""
        return [c for c in design.columns
                if c != 'Intercept' and self._get_term(c) == term]

    def _decompose(self, X, factors):
        return _decompose_standalone(X, factors, self.decomp_type)

    def _decompose_(self, X, factors):
        grand_mean = X.mean(axis=0)
        X_centered = X - grand_mean

        factor_names = list(factors.columns)
        formula = "*".join([f"C({f}, Sum)" for f in factor_names])
        design = model_matrix(formula, factors)

        effect_groups = {}
        for col in design.columns:
            if col == "Intercept":
                continue
            term = self._get_term(col)
            if term not in effect_groups:
                effect_groups[term] = self._get_effect_cols(design, term)

        effect_matrices = {}
        if self.decomp_type ==  1:
            # Project onto each effect's full subspace
            X_remaining = X_centered.copy()
            for term, cols in effect_groups.items():
                D = design[cols].values
                beta, _, _, _ = np.linalg.lstsq(D, X_remaining, rcond=None)
                effect_matrices[term] = D @ beta
                X_remaining = X_remaining - effect_matrices[term]

            E =  X_remaining

            reconstruction =grand_mean + sum(effect_matrices.values()) + E
            assert np.allclose(reconstruction, X), "Decomposition failed"

            SS_total = np.sum(X_centered ** 2)
            SS = {col: np.sum(mat ** 2) for col, mat in effect_matrices.items()}
            SS_residual = np.sum(E ** 2)
            if self.verbose:
                print(f"Variance check: {sum(SS.values()) + SS_residual:.4f} == {SS_total:.4f}")
            return effect_matrices, SS, E, grand_mean, X_centered, SS_total, SS_residual

        elif self.decomp_type == 3:
            # ASCA+ (Thiel et al. 2017): Type III effect matrices and SS
            D_full = design.drop(columns='Intercept').values
            beta_full, _, _, _ = np.linalg.lstsq(D_full, X_centered, rcond=None)
            fitted_full = D_full @ beta_full

            # Effect matrices AND SS via full-minus-reduced projection
            SS = {}
            for term, cols in effect_groups.items():
                other_cols = [c for c in design.columns if c != "Intercept" and c not in cols]
                if other_cols:
                    D_reduced = design[other_cols].values
                    beta_red, _, _, _ = np.linalg.lstsq(D_reduced, X_centered, rcond=None)
                    fitted_reduced = D_reduced @ beta_red
                    effect_matrices[term] = fitted_full - fitted_reduced
                else:
                    effect_matrices[term] = fitted_full.copy()
                SS[term] = np.sum(effect_matrices[term] ** 2)

            # Residuals
            E = X_centered - fitted_full
            SS_residual = np.sum(E ** 2)
            SS_total = np.sum(X_centered ** 2)
            if self.verbose:
                print(f"SS_total: {SS_total:.4f}, SS_residual: {SS_residual:.4f}")
            return effect_matrices, SS, E, grand_mean, X_centered, SS_total, SS_residual
        else:
            return "Select a 3 for type III decomposition"

    def _permute(self, nperm):
        perm_ss = {term: np.zeros(nperm) for term in self.SS}

        for term in self.SS:
            term_factors = [f for f in self.factors.columns if f"C({f}, Sum)" in term]

            for i in range(nperm):
                factors_perm = self.factors.copy()
                for col in term_factors:
                    factors_perm[col] = np.random.permutation(self.factors[col].values)

                # decompose
                _, SS_perm, _, _, _, _, _ = self._decompose(self.X, factors_perm)
                perm_ss[term][i] = SS_perm[term]
                if i % 100 == 0:
                    print(f"nperm {i} of {nperm} for {term}")

        # pvalues
        pvalues = {term: np.mean(perm_ss[term] >= self.SS[term]) for term in self.SS}
        return pvalues, perm_ss

    def _clean_term(self, term):
        parts = term.split(':')
        cleaned = [re.search(r'C\((\w+)', p).group(1) for p in parts]
        return ':'.join(cleaned)

    def summary(self):
        print(f"\n{'Effect':<40} {'SS':>12} {'%Var':>8} {'p-value':>10}")
        print("-" * 74)
        for term, ss in self.SS.items():
            pct = 100 * ss / self.SS_total
            pval = self.pvalues.get(term, np.nan)
            term = self._clean_term(term)
            print(f"{term:<40} {ss:>12.4f} {pct:>7.1f}% {pval:>10.3f}")
        print(f"{'Residual':<40} {self.SS_residual:>12.4f} "
              f"{100 * self.SS_residual / self.SS_total:>7.1f}%")
        print(f"{'Total':<40} {self.SS_total:>12.4f} {'100.0':>7}%")

    def _find_term(self, name):
        """Match clean name to internal term key"""
        for term in self.effect_matrices:
            if name in term:
                return term
        raise ValueError(f"Term '{name}' not found. Available: {list(self.SS.keys())}")

    def _get_factor_levels(self, term_name):
        """Get factor levels for coloring — handles main effects and interactions"""
        parts = term_name.split(':')
        if len(parts) == 1:
            return self.factors[term_name].values
        else:
            # combine levels for interaction terms
            return pd.Series(
                ['|'.join(row) for row in self.factors[parts].values.astype(str)]
            ).values

    def _plot_scores(self, scores, term_name, var, kind,
                     loadings=None, var_names=None, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (7, 5)))
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')

        levels = self._get_factor_levels(term_name)
        unique_levels = np.unique(levels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_levels)))

        for i, (level, color) in enumerate(zip(unique_levels, colors)):
            mask = levels == level

            ax.scatter(scores[mask, 0], scores[mask, 1],
                       label=level,
                       color=color,
                       marker=MARKERS[i % len(MARKERS)],
                       s=60,
                       edgecolors='black',
                       linewidths=0.5,
                       zorder=3)

        margin = 1.5
        ax.set_xlim(scores[:, 0].min() * margin, scores[:, 0].max() * margin)
        ax.set_ylim(scores[:, 1].min() * margin, scores[:, 1].max() * margin)
        ax.legend(title=term_name, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(f"PC1 {var[0]:0.1%}", fontsize=11)
        ax.set_ylabel(f"PC2 {var[1]:0.1%}", fontsize=11)

        scores_plot = scores # / scores.std(axis=0)

        if kind == "biplot" and loadings is not None:
            n_arrows = kwargs.get('n_arrows', 10)
            magnitudes = np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2)
            # filter to meaningful magnitude only
            threshold = magnitudes.max() * 0.1  # at least 10% of max magnitude
            candidates = np.where(magnitudes > threshold)[0]
            idx = candidates[np.argsort(magnitudes[candidates])[::-1][:n_arrows]]

            print(f"Showing {len(idx)} arrows, threshold={threshold:.4f}")
            print(f"Top species magnitudes: {magnitudes[idx]}")

            ax.set_title(f"{term_name} Biplot")

            # scale — use only PC1 and PC2 of loadings
            load2d = loadings[:, :2]
            scale = np.max(np.abs(scores_plot)) / np.max(np.abs(loadings[:, :2]))
            idx = np.argsort(np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2))[::-1][:n_arrows]
            for i in idx:
                lx = load2d[i, 0] * scale / 2
                ly = load2d[i, 1] * scale /2
                ax.annotate('',
                            xy=(lx, ly),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle='-|>',
                                            color='firebrick',
                                            lw=.75,
                                            mutation_scale=14))
                label = var_names[i] if var_names is not None else str(i)
                ax.text(lx * 1.1, ly * 1.1, label, fontsize=9, color='firebrick')
        else:
            ax.set_title(f"{term_name} Scores")
        save_path = kwargs.get('savefig', None)
        if save_path:
            plt.savefig(Path(save_path) / f"asca_{term_name}.pdf")
        return fig, ax

    def _plot_loadings(self, loadings, term_name, n_load: int =15, var_names=None):
        fig, ax = plt.subplots(figsize=(6, 8))

        # top loadings by PC1 absolute value
        idx = np.argsort(np.abs(loadings[:, 0]))[::-1][:n_load]
        vals = loadings[idx, 0]
        labels = var_names[idx] if var_names is not None else [str(i) for i in idx]

        colors = ['steelblue' if v >= 0 else 'firebrick' for v in vals]
        ax.barh(range(len(idx)), vals, color=colors)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(-1, 1)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel("PC1 loading")
        ax.set_title(f"Loadings: {term_name}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig, ax

    def plot(self, term_name: str, kind: str ="scores", var_names: list|ndarray=None, **kwargs):
        '''biplot of scores/loadings for specified factor

        Args:
            term_name (str): name of experimental variable/term
            kind (str): 'scores', 'loading', or 'biplot'
            var_names (list|ndarray): list of variable names
            **kwargs: additional arguments
                figsize (tuple): figure size
                n_load (int): number of arrows

        '''
        term = self._find_term(term_name)
        scores = self.scores.get(term)
        var = self.pcas.get(term).explained_variance_ratio_
        loadings = self.loadings.get(term)

        figsize = kwargs.get('figsize', (7, 5))
        n_load = kwargs.get('n_load', len(loadings) if loadings is not None else 15)
        fig, ax = None, None
        if kind=="scores":
            fig, ax = self._plot_scores(scores, term_name, var, kind, figsize=figsize)
        elif kind=="loading":
            fig, ax = self._plot_loadings(loadings, term_name, n_load, var_names)
        elif kind=="biplot":
            fig, ax = self._plot_scores(scores,
                              term_name,
                              var,
                              kind='biplot',
                              loadings=loadings,
                              var_names=var_names,
                              **kwargs)
        if not fig or not ax:
            raise "There's a problem with the figure and axes"
        return fig, ax


if __name__ == '__main__':
    help(ASCA)


