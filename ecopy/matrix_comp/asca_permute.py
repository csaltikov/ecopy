from multiprocessing import Pool, cpu_count, get_context
import numpy as np
from formulaic import model_matrix
import time
import sys


DNA_SPINNER = ['∙∙∙∙∙∙', 'A∙∙∙∙∙', 'AT∙∙∙∙', 'ATC∙∙∙',
               'ATCG∙∙', 'ATCGA∙', 'ATCGAT', '∙TCGAT',
               '∙∙CGAT', '∙∙∙GAT', '∙∙∙∙AT', '∙∙∙∙∙T', '∙∙∙∙∙∙']


def _get_term(col):
    """Extract term name from a patsy column name"""
    return ":".join(part.split('[')[0] for part in col.split(':'))


def _get_effect_cols(design, term):
    """Get all design matrix columns belonging to exactly this term"""
    return [c for c in design.columns
            if c != 'Intercept' and _get_term(c) == term]


def _build_effect_groups(design):
    effect_groups = {}
    for col in design.columns:
        if col == "Intercept":
            continue
        term = _get_term(col)
        if term not in effect_groups:
            effect_groups[term] = _get_effect_cols(design, term)
    return effect_groups


def _decompose_standalone(X, factors, decomp_type=1):
    """Standalone decompose for multiprocessing — no class reference"""
    grand_mean = X.mean(axis=0)
    X_centered = X - grand_mean

    factor_names = list(factors.columns)
    formula = "*".join([f"C({f}, Sum)" for f in factor_names])
    design = model_matrix(formula, factors)

    effect_groups = _build_effect_groups(design)
    effect_matrices = {}

    if decomp_type == 1:
        # Project onto each effect's full subspace
        X_remaining = X_centered.copy()
        for term, cols in effect_groups.items():
            D = design[cols].values
            beta, _, _, _ = np.linalg.lstsq(D, X_remaining, rcond=None)
            effect_matrices[term] = D @ beta
            X_remaining = X_remaining - effect_matrices[term]
        E = X_remaining  # self.X_centered - sum(self.effect_matrices.values())

        SS_total = np.sum(X_centered ** 2)
        SS = {t: np.sum(mat ** 2) for t, mat in effect_matrices.items()}
        SS_residual = np.sum(E ** 2)

        return effect_matrices, SS, E, grand_mean, X_centered, SS_total, SS_residual

    elif decomp_type == 3:
        # ASCA+ (Thiel et al. 2017): Type III effect matrices and SS
        # Full model fit
        D_full = design.drop(columns='Intercept').values
        beta_full, _, _, _ = np.linalg.lstsq(D_full, X_centered, rcond=None)
        fitted_full = D_full @ beta_full

        # Effect matrices AND SS via full-minus-reduced projection
        # Each effect matrix = fitted_full - fitted_reduced (Eq. 18, Bertinetto 2020)
        # This ensures both the PCA scores/loadings AND the SS
        # reflect each term's unique contribution
        SS = {}
        for term, cols in effect_groups.items():
            other_cols = [c for c in design.columns if c != "Intercept" and c not in cols]
            if other_cols:
                D_reduced = design[other_cols].values
                beta_red, _, _, _ = np.linalg.lstsq(D_reduced, X_centered, rcond=None)
                fitted_reduced = D_reduced @ beta_red
                effect_matrices[term] = fitted_full - fitted_reduced
            else:
                # This term is the only predictor
                effect_matrices[term] = fitted_full.copy()
            SS[term] = np.sum(effect_matrices[term] ** 2)

        # Residuals
        E = X_centered - fitted_full
        SS_total = np.sum(X_centered ** 2)
        SS_residual = np.sum(E ** 2)
        return effect_matrices, SS, E, grand_mean, X_centered, SS_total, SS_residual


def _permute_worker(args):
    term, term_factors, factors, X, SS_obs, nperm, decomp_type = args
    rng = np.random.default_rng()
    perm_ss = np.zeros(nperm)

    for i in range(nperm):
        factors_perm = factors.copy()
        for col in term_factors:
            factors_perm[col] = rng.permutation(factors[col].values)
        # decompose
        _, SS_perm, _, _, _, _, _ = _decompose_standalone(X, factors_perm, decomp_type)
        perm_ss[i] = SS_perm[term]
    pval = float(np.mean(perm_ss >= SS_obs))
    return term, perm_ss, pval


def _spin(async_result, message="Permuting"):
    unicode_ok = (sys.stdout.encoding or '').lower() in ('utf-8', 'utf-16')
    spinner = DNA_SPINNER if unicode_ok else ['.  ', '.. ', '...']
    i = 0
    while not async_result.ready():
        print(f"\r{message} {spinner[i % len(spinner)]}", end='', flush=True)
        i += 1
        time.sleep(0.15)
    for frame in ['.  ', '.. ', '...', ' ✓ ']:
        print(f"\rDone {frame}          ", end='', flush=True)
        time.sleep(0.15)
    print()


def run_permutations(SS, factors, X, decomp_type, nperm=999, verbose=True):
    """Call from ASCA.__init__ - build args and run pool"""
    args_list = []
    for term in SS:
        term_factors = [f for f in factors.columns if f"C({f}, Sum)" in term]
        args_list.append((
            term,
            term_factors,
            factors.copy(),
            X,
            SS[term],
            nperm,
            decomp_type
        ))
    ncores = max(1, cpu_count() - 1)

    with get_context("fork").Pool(ncores) as pool:
        if verbose:
            print(f"Permutation testing: {len(args_list)} terms × "
                  f"{nperm} permutations ({ncores} cores)")
        async_result = pool.map_async(_permute_worker, args_list)
        if verbose:
            _spin(async_result)
        results = async_result.get()

    perm_ss = {}
    pvalues = {}
    for term, ps, pval in results:
        perm_ss[term] = ps
        pvalues[term] = pval

    return pvalues, perm_ss


if __name__ == "__main__":
    print("ANOVA-Simultaneous Component Analysis")
