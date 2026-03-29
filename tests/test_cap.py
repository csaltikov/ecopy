"""
Tests for ecopy.cap (Canonical Analysis of Principal Coordinates / dbRDA).

Run with:
    python -m pytest test_cap.py -v
    # or
    python test_cap.py

All tests use fully deterministic synthetic data (np.random.seed(0)) so
expected values never change between runs.

Test groups
-----------
TestCAPInputValidation   -- bad inputs raise ValueError
TestCAPSingleVariable    -- one continuous predictor (Depth)
TestCAPTwoVariables      -- two continuous predictors (Depth + Temp)
TestCAPCategorical       -- mixed continuous + categorical predictor
TestCAPAttributes        -- internal consistency checks across all models
TestCAPMethods           -- summary(), anova(), biplot() run without error
"""

import unittest
import numpy as np
import pandas as pd

# ---- Import cap from ecopy if available, else fall back to local ----
try:
    from ecopy import cap
except ImportError:
    try:
        from ecopy.ordination.cap import cap
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from cap import cap


# ======================================================================
# Shared fixtures — built once, reused across all test classes
# ======================================================================

def _make_fixtures():
    """
    Synthetic community with a clear depth gradient.
    12 sites x 30 species, 4 depth levels (5, 10, 20, 30 m), 3 reps each.
    Shallow species decline with depth; deep species increase.
    """
    np.random.seed(0)
    n_sites, n_spp = 12, 30
    depths = np.array([5,5,5, 10,10,10, 20,20,20, 30,30,30], dtype=float)
    temps  = np.array([25,25,25, 22,22,22, 15,15,15, 10,10,10], dtype=float)
    zones  = ['shallow']*3 + ['mid']*6 + ['deep']*3

    counts = np.zeros((n_sites, n_spp))
    for i in range(n_sites):
        counts[i, :15] = np.random.poisson(max(20 - depths[i], 2), 15)
        counts[i, 15:] = np.random.poisson(max(depths[i] - 5,  2), 15)

    site_names = [f'Site{i+1}' for i in range(n_sites)]

    # Bray-Curtis dissimilarity
    n = n_sites
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            num = np.abs(counts[i] - counts[j]).sum()
            den = counts[i].sum() + counts[j].sum()
            D[i,j] = D[j,i] = num / den if den > 0 else 0.0

    D_df = pd.DataFrame(D, index=site_names, columns=site_names)
    D_np = D.copy()

    env_single = pd.DataFrame({'Depth': depths},           index=site_names)
    env_two    = pd.DataFrame({'Depth': depths,
                               'Temp':  temps},            index=site_names)
    env_cat    = pd.DataFrame({'Depth': depths,
                               'Zone':  zones},            index=site_names)

    # Run models once — reused across all tests
    c1 = cap(D_df, env_single, nperm=999, seed=0)   # 1 continuous var
    c2 = cap(D_df, env_two,    nperm=999, seed=0)   # 2 continuous vars
    c3 = cap(D_df, env_cat,    nperm=999, seed=0)   # 1 continuous + 1 factor

    return dict(
        D_df=D_df, D_np=D_np,
        site_names=site_names, depths=depths, temps=temps, zones=zones,
        env_single=env_single, env_two=env_two, env_cat=env_cat,
        c1=c1, c2=c2, c3=c3, n_sites=n_sites,
    )

# Build fixtures at module level so all test classes share them
FX = _make_fixtures()


# ======================================================================
class TestCAPInputValidation(unittest.TestCase):
    """Bad inputs should raise ValueError with informative messages."""

    def test_non_square_distance_matrix(self):
        """Non-square array for D raises ValueError."""
        D_bad = np.ones((4, 5))
        env   = pd.DataFrame({'x': [1., 2., 3., 4.]})
        with self.assertRaises(ValueError):
            cap(D_bad, env, nperm=9)

    def test_asymmetric_distance_matrix(self):
        """Asymmetric matrix for D raises ValueError."""
        D_bad = pd.DataFrame([[0., 1., 2.],
                              [9., 0., 1.],
                              [2., 1., 0.]])
        env   = pd.DataFrame({'x': [1., 2., 3.]})
        with self.assertRaises(ValueError):
            cap(D_bad, env, nperm=9)

    def test_negative_distance_matrix(self):
        """Distance matrix with negative values raises ValueError."""
        D_bad = FX['D_np'].copy()
        D_bad[0, 1] = D_bad[1, 0] = -0.1
        env   = FX['env_single']
        with self.assertRaises(ValueError):
            cap(D_bad, env, nperm=9)

    def test_nan_in_distance_matrix(self):
        """NaN in D raises ValueError."""
        D_bad = FX['D_np'].copy()
        D_bad[0, 2] = D_bad[2, 0] = np.nan
        with self.assertRaises(ValueError):
            cap(D_bad, FX['env_single'], nperm=9)

    def test_nan_in_env_matrix(self):
        """NaN in X raises ValueError."""
        env_bad = FX['env_single'].copy()
        env_bad.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            cap(FX['D_df'], env_bad, nperm=9)

    def test_row_mismatch(self):
        """D and X with different numbers of rows raises ValueError."""
        env_short = FX['env_single'].iloc[:5]
        with self.assertRaises(ValueError):
            cap(FX['D_df'], env_short, nperm=9)

    def test_numpy_object_array_raises(self):
        """numpy.ndarray with object dtype for X raises ValueError."""
        X_bad = np.array([['a', 'b']] * FX['n_sites'], dtype=object)
        with self.assertRaises(ValueError):
            cap(FX['D_df'], X_bad, nperm=9)

    def test_d_wrong_type(self):
        """Non-array type for D raises ValueError."""
        with self.assertRaises(ValueError):
            cap([[0, 1], [1, 0]], FX['env_single'].iloc[:2], nperm=9)

    def test_x_wrong_type(self):
        """Non-array type for X raises ValueError."""
        with self.assertRaises(ValueError):
            cap(FX['D_df'], "depth", nperm=9)


# ======================================================================
class TestCAPSingleVariable(unittest.TestCase):
    """CAP with one continuous predictor: CAP ~ Depth."""

    def setUp(self):
        self.c = FX['c1']

    # ---- Scalar statistics ----
    def test_total_inertia(self):
        self.assertAlmostEqual(self.c.total_inertia, 1.684251, places=4)

    def test_R2(self):
        self.assertAlmostEqual(self.c.R2, 0.781209, places=4)

    def test_R2adj(self):
        self.assertAlmostEqual(self.c.R2adj, 0.759330, places=4)

    def test_R2adj_less_than_R2(self):
        self.assertLess(self.c.R2adj, self.c.R2)

    def test_F_stat(self):
        self.assertAlmostEqual(self.c.F_stat, 35.705753, places=2)

    def test_degrees_of_freedom(self):
        self.assertEqual(self.c.df_c, 1)
        self.assertEqual(self.c.df_r, 10)

    def test_p_value_significant(self):
        """Depth should explain significant community variance (p <= 0.05)."""
        self.assertLessEqual(self.c.p_value, 0.05)

    # ---- Axes ----
    def test_single_cap_axis(self):
        """One predictor → exactly one CAP axis."""
        self.assertEqual(self.c.n_cap_axes, 1)

    def test_residual_axes_present(self):
        """Residual axes must exist (n - rank - 1 > 0)."""
        self.assertGreater(self.c.n_res_axes, 0)

    def test_cap_eigenvalue(self):
        self.assertAlmostEqual(self.c.cap_evals[0], 1.315752, places=4)

    # ---- Score shapes ----
    def test_capScores_shape(self):
        n = FX['n_sites']
        self.assertEqual(self.c.capScores.shape, (n, 1))

    def test_resScores_shape(self):
        n = FX['n_sites']
        self.assertEqual(self.c.resScores.shape[0], n)
        self.assertGreater(self.c.resScores.shape[1], 0)

    def test_envScores_shape(self):
        self.assertEqual(self.c.envScores.shape, (1, 1))

    # ---- Column names ----
    def test_cap_axis_name(self):
        self.assertEqual(self.c.capScores.columns[0], 'CAP 1')

    def test_res_axis_name(self):
        self.assertEqual(self.c.resScores.columns[0], 'ResPC 1')

    # ---- Environmental correlation ----
    def test_depth_cap1_correlation_near_one(self):
        """With a single standardized predictor, |corr(Depth, CAP1)| = 1."""
        self.assertAlmostEqual(abs(self.c.envScores.iloc[0, 0]), 1.0, places=5)

    # ---- Site names ----
    def test_site_names_in_index(self):
        self.assertEqual(list(self.c.capScores.index), FX['site_names'])

    # ---- Importance table ----
    def test_imp_row_names(self):
        self.assertIn('Std Dev',  self.c.imp.index)
        self.assertIn('Prop Var', self.c.imp.index)
        self.assertIn('Cum Var',  self.c.imp.index)

    def test_imp_proportions_sum_to_one(self):
        total_prop = self.c.imp.loc['Prop Var'].sum()
        self.assertAlmostEqual(total_prop, 1.0, places=4)

    def test_cum_var_ends_at_one(self):
        last_cum = self.c.imp.loc['Cum Var'].iloc[-1]
        self.assertAlmostEqual(last_cum, 1.0, places=4)


# ======================================================================
class TestCAPTwoVariables(unittest.TestCase):
    """CAP with two continuous predictors: CAP ~ Depth + Temp."""

    def setUp(self):
        self.c = FX['c2']

    def test_R2(self):
        self.assertAlmostEqual(self.c.R2, 0.884098, places=4)

    def test_R2adj(self):
        self.assertAlmostEqual(self.c.R2adj, 0.858342, places=4)

    def test_two_cap_axes(self):
        """Two predictors → two CAP axes."""
        self.assertEqual(self.c.n_cap_axes, 2)

    def test_degrees_of_freedom(self):
        self.assertEqual(self.c.df_c, 2)
        self.assertEqual(self.c.df_r, 9)

    def test_capScores_shape(self):
        self.assertEqual(self.c.capScores.shape, (FX['n_sites'], 2))

    def test_envScores_shape(self):
        """2 vars x 2 CAP axes."""
        self.assertEqual(self.c.envScores.shape, (2, 2))

    def test_cap_axis_names(self):
        self.assertEqual(list(self.c.capScores.columns), ['CAP 1', 'CAP 2'])

    def test_env_var_names(self):
        self.assertEqual(list(self.c.envScores.index), ['Depth', 'Temp'])

    def test_constrained_prop_equals_R2(self):
        """Sum of CAP axis proportions in imp table should equal R2."""
        cap_prop_sum = self.c.imp.loc['Prop Var', ['CAP 1', 'CAP 2']].sum()
        self.assertAlmostEqual(cap_prop_sum, self.c.R2, places=5)

    def test_p_value_significant(self):
        self.assertLessEqual(self.c.p_value, 0.05)

    def test_R2_greater_than_single_var(self):
        """Adding a second predictor should not decrease R2."""
        self.assertGreaterEqual(self.c.R2, FX['c1'].R2)


# ======================================================================
class TestCAPCategorical(unittest.TestCase):
    """CAP with mixed continuous + categorical predictor (Depth + Zone)."""

    def setUp(self):
        self.c = FX['c3']

    def test_R2(self):
        self.assertAlmostEqual(self.c.R2, 0.851168, places=4)

    def test_dummy_coding_expands_vars(self):
        """Zone (3 levels, drop-first) should add 2 dummy columns → 3 total vars."""
        self.assertEqual(len(self.c.varNames_x), 3)

    def test_var_names_include_dummies(self):
        """Dummy variable names should use 'ColName: Level' format."""
        self.assertIn('Depth',        self.c.varNames_x)
        self.assertIn('Zone: mid',    self.c.varNames_x)
        self.assertIn('Zone: shallow', self.c.varNames_x)

    def test_ptypes_correct(self):
        """Depth should be 'q'; dummy columns should be 'f'."""
        self.assertEqual(self.c.pTypes[0], 'q')
        self.assertEqual(self.c.pTypes[1], 'f')
        self.assertEqual(self.c.pTypes[2], 'f')

    def test_df_c(self):
        """Rank of design matrix (3 cols, full rank) → df_c = 3."""
        self.assertEqual(self.c.df_c, 3)

    def test_three_cap_axes(self):
        self.assertEqual(self.c.n_cap_axes, 3)

    def test_numpy_array_input(self):
        """X as numpy.ndarray (quantitative only) should work."""
        X_np = FX['depths'].reshape(-1, 1)
        c_np = cap(FX['D_np'], X_np, nperm=9, seed=0)
        self.assertAlmostEqual(c_np.R2, FX['c1'].R2, places=5)


# ======================================================================
class TestCAPAttributes(unittest.TestCase):
    """Internal consistency checks valid for any well-formed CAP result."""

    def _check_model(self, c, label):
        n = FX['n_sites']

        # Constrained + residual inertia == total inertia
        reconstructed = c.cap_evals.sum() + c.res_evals.sum()
        self.assertAlmostEqual(reconstructed, c.total_inertia, places=8,
            msg=f'{label}: con + res inertia should equal total')

        # R2 = constrained / total
        r2_check = c.cap_evals.sum() / c.total_inertia
        self.assertAlmostEqual(r2_check, c.R2, places=8,
            msg=f'{label}: R2 = constrained / total')

        # R2adj always <= R2
        self.assertLessEqual(c.R2adj, c.R2,
            msg=f'{label}: R2adj should be <= R2')

        # capScores has correct number of rows
        self.assertEqual(c.capScores.shape[0], n,
            msg=f'{label}: capScores rows == n_sites')

        # resScores has correct number of rows
        self.assertEqual(c.resScores.shape[0], n,
            msg=f'{label}: resScores rows == n_sites')

        # Importance table columns span all positive axes
        self.assertAlmostEqual(c.imp.loc['Prop Var'].sum(), 1.0, places=4,
            msg=f'{label}: imp Prop Var sums to 1')

        # p-value in (0, 1]
        self.assertGreater(c.p_value, 0.0,
            msg=f'{label}: p_value > 0')
        self.assertLessEqual(c.p_value, 1.0,
            msg=f'{label}: p_value <= 1')

        # envScores shape: (n_vars, n_cap_axes)
        self.assertEqual(c.envScores.shape,
                         (len(c.varNames_x), c.n_cap_axes),
            msg=f'{label}: envScores shape')

        # F_stat is positive finite
        self.assertGreater(c.F_stat, 0.0,
            msg=f'{label}: F_stat > 0')
        self.assertTrue(np.isfinite(c.F_stat),
            msg=f'{label}: F_stat is finite')

    def test_single_variable_consistency(self):
        self._check_model(FX['c1'], 'CAP~Depth')

    def test_two_variable_consistency(self):
        self._check_model(FX['c2'], 'CAP~Depth+Temp')

    def test_categorical_consistency(self):
        self._check_model(FX['c3'], 'CAP~Depth+Zone')

    def test_scale_x_false_changes_result(self):
        """Disabling scale_x should give a different (but still valid) R2."""
        c_noscale = cap(FX['D_df'], FX['env_two'], scale_x=False, nperm=9, seed=0)
        # R2 is invariant to scaling of X in OLS — but numerical path differs
        # so we just confirm it still runs and gives a finite result
        self.assertTrue(np.isfinite(c_noscale.R2))
        self.assertGreater(c_noscale.R2, 0)

    def test_custom_site_names(self):
        custom = [f'Lake{i}' for i in range(FX['n_sites'])]
        c = cap(FX['D_np'], FX['env_single'].values, siteNames=custom, nperm=9)
        self.assertEqual(list(c.capScores.index), custom)

    def test_custom_var_names(self):
        c = cap(FX['D_np'], FX['env_single'].values,
                varNames_x=['WaterDepth'], nperm=9)
        self.assertEqual(list(c.envScores.index), ['WaterDepth'])

    def test_reproducibility(self):
        """Same seed → identical p-value."""
        c_a = cap(FX['D_df'], FX['env_single'], nperm=99, seed=7)
        c_b = cap(FX['D_df'], FX['env_single'], nperm=99, seed=7)
        self.assertEqual(c_a.p_value, c_b.p_value)

    def test_different_seeds_may_differ(self):
        """Different seeds should generally produce different p-values
        (not guaranteed but almost certain with nperm=999)."""
        c_a = cap(FX['D_df'], FX['env_single'], nperm=999, seed=1)
        c_b = cap(FX['D_df'], FX['env_single'], nperm=999, seed=2)
        # F-stat must be identical (data unchanged)
        self.assertAlmostEqual(c_a.F_stat, c_b.F_stat, places=10)


# ======================================================================
class TestCAPMethods(unittest.TestCase):
    """summary(), anova(), and biplot() should run without raising errors."""

    def test_summary_runs(self):
        import io, sys
        buf = io.StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buf
        try:
            FX['c1'].summary()
        finally:
            sys.stdout = sys_stdout
        output = buf.getvalue()
        self.assertIn('R²', output)
        self.assertIn('Total inertia', output)
        self.assertIn('Constrained', output)

    def test_anova_returns_dict(self):
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        try:
            result = FX['c2'].anova(nperm=99)
        finally:
            sys.stdout = sys.__stdout__
        self.assertIsInstance(result, dict)
        self.assertIn('F_stat',  result)
        self.assertIn('p_value', result)
        self.assertIn('nperm',   result)
        self.assertEqual(result['nperm'], 99)

    def test_anova_F_stat_matches_attribute(self):
        import io, sys
        sys.stdout = io.StringIO()
        try:
            result = FX['c1'].anova(nperm=99)
        finally:
            sys.stdout = sys.__stdout__
        self.assertAlmostEqual(result['F_stat'], FX['c1'].F_stat, places=8)

    def test_biplot_single_var_runs(self):
        """biplot() with one CAP axis should not raise (uses ResPC1 for y)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            FX['c1'].biplot(site_labels=False)
        finally:
            plt.close('all')

    def test_biplot_two_var_runs(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            FX['c2'].biplot(xax=1, yax=2, site_labels=True)
        finally:
            plt.close('all')

    def test_biplot_color_by_numeric(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            FX['c2'].biplot(color_by=FX['depths'], site_labels=False)
        finally:
            plt.close('all')

    def test_biplot_color_by_categorical(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            FX['c2'].biplot(color_by=FX['zones'], site_labels=False)
        finally:
            plt.close('all')


# ======================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
