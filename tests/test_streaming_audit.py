# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Tests for the stratified streaming paired audit."""

import warnings

import numpy as np
import pytest

from deel.puncc.streaming_audit import AuditStateError
from deel.puncc.streaming_audit import SourceOverlapError
from deel.puncc.streaming_audit import StratifiedStreamingPairedAudit


# --------------------------------------------------------------------------- #
# Deterministic fake interval models
# --------------------------------------------------------------------------- #
class FixedIntervalModel:
    """Interval model with a tunable constant half-width.

    Records the exact X batch it predicted on, so tests can assert that
    both models run on the very same retained records (paired design).
    """

    def __init__(self, halfwidth):
        self.halfwidth = float(halfwidth)
        self.predicted_batches = []

    def predict(self, X, alpha=None):
        X = np.asarray(X, dtype=float)
        n_records = X.shape[0]
        self.predicted_batches.append(X.copy())
        y_pred = np.zeros(n_records)
        return y_pred, y_pred - self.halfwidth, y_pred + self.halfwidth


class PrescribedWidthModel:
    """Model whose absolute interval widths are looked up by record id.

    The record id is the (single) feature value. ``width_A - width_B``
    between two such models reproduces any prescribed difference series.
    """

    def __init__(self, widths):
        self.widths = np.asarray(widths, dtype=float)

    def predict(self, X, alpha=None):
        # Robust to both stacked 2D batches and 1D object-array batches:
        # take the first feature of every record as the lookup id.
        X_arr = np.asarray(X)
        if X_arr.dtype == object and X_arr.ndim == 1:
            idx = np.array([int(np.asarray(rec).reshape(-1)[0]) for rec in X_arr])
        else:
            idx = X_arr.reshape(X_arr.shape[0], -1)[:, 0].astype(int)
        width = self.widths[idx]
        y_pred = np.zeros(len(idx))
        return y_pred, y_pred - width / 2.0, y_pred + width / 2.0


class NoAlphaModel(FixedIntervalModel):
    """Interval model whose predict does not accept alpha."""

    def predict(self, X):  # type: ignore[override]
        return super().predict(X)


class BadModel:
    def predict(self, X, alpha=None):
        return np.zeros(len(X))  # not a 3-tuple


# --------------------------------------------------------------------------- #
# Reservoir / streaming bookkeeping
# --------------------------------------------------------------------------- #
def test_reservoir_keeps_at_most_K_and_exact_population_count():
    audit = StratifiedStreamingPairedAudit(K=10, random_state=0)
    X = np.arange(250).reshape(-1, 1)
    y = np.arange(250.0)
    for start in range(0, 250, 13):  # stream in many small batches
        end = min(start + 13, 250)
        audit.update(
            X[start:end],
            y[start:end],
            np.zeros(end - start, dtype=int),
        )
    assert audit.population_size(0) == 250
    assert audit.sample_size(0) == 10
    X_ret, y_ret, src_ret = audit.retained_records(0)
    assert len(X_ret) == len(y_ret) == len(src_ret) == 10
    # Reservoir only stores retained records, never the full history.
    assert audit.sample_size(0) < audit.population_size(0)


def test_reservoir_is_uniform_without_replacement():
    # Every population record must have inclusion probability ~ K/N and
    # appear at most once per realized sample (SRSWOR).
    n_pop, k_cap, reps = 100, 20, 800
    counts = np.zeros(n_pop)
    for rep in range(reps):
        audit = StratifiedStreamingPairedAudit(K=k_cap, random_state=rep)
        X = np.arange(n_pop).reshape(-1, 1)
        audit.update(X, np.zeros(n_pop), np.zeros(n_pop, dtype=int))
        retained, _, _ = audit.retained_records(0)
        ids = [int(rec[0]) for rec in retained]
        assert len(set(ids)) == len(ids)  # without replacement
        counts[ids] += 1
    target = reps * k_cap / n_pop
    # Inclusion frequencies stay within a generous Monte Carlo band.
    assert np.all(counts > 0)
    assert abs(counts.mean() - target) < 0.15 * target


def test_stratification_and_source_metadata_are_tracked():
    audit = StratifiedStreamingPairedAudit(K=5, random_state=0)
    X = np.arange(40).reshape(-1, 1)
    y = np.arange(40.0)
    strata = np.array([0] * 20 + [1] * 20)
    sources = np.array(["s1"] * 20 + ["s2"] * 20, dtype=object)
    audit.update(X, y, strata, sources=sources)
    assert set(audit.strata) == {0, 1}
    assert audit.population_size(0) == 20
    assert audit.population_size(1) == 20
    assert audit.source_counts() == {"s1": 20, "s2": 20}
    assert audit.source_counts(stratum=0) == {"s1": 20}


def test_per_stratum_capacity_mapping():
    audit = StratifiedStreamingPairedAudit(K={0: 3}, default_K=7, random_state=0)
    X = np.arange(60).reshape(-1, 1)
    strata = np.array([0] * 30 + [1] * 30)
    audit.update(X, np.zeros(60), strata)
    assert audit.sample_size(0) == 3
    assert audit.sample_size(1) == 7
    with pytest.raises(ValueError):
        StratifiedStreamingPairedAudit(K={0: 3})  # missing default_K
    with pytest.raises(ValueError):
        StratifiedStreamingPairedAudit(K=0)


def test_scalar_stratum_and_length_validation():
    audit = StratifiedStreamingPairedAudit(K=4, random_state=0)
    audit.update(np.arange(6).reshape(-1, 1), np.arange(6.0), "only")
    assert audit.strata == ("only",)
    with pytest.raises(ValueError):
        audit.update(np.zeros((3, 1)), np.zeros(2), np.zeros(3, dtype=int))


# --------------------------------------------------------------------------- #
# Paired audit metrics
# --------------------------------------------------------------------------- #
def test_both_models_predict_on_same_retained_records_with_native_metrics():
    audit = StratifiedStreamingPairedAudit(K=8, random_state=1)
    X = np.arange(50).reshape(-1, 1)
    # Labels inside A (halfwidth 1) but outside B (halfwidth 0.2) around 0.
    y = np.full(50, 0.5)
    audit.update(X, y, np.zeros(50, dtype=int))
    model_a, model_b = FixedIntervalModel(1.0), FixedIntervalModel(0.2)
    result = audit.compute_audit(model_a, model_b, alpha=0.1)

    # Each model predicts once per stratum, on the identical retained X.
    assert len(model_a.predicted_batches) == 1
    np.testing.assert_array_equal(model_a.predicted_batches[0], model_b.predicted_batches[0])
    stratum = result.strata[0]
    assert stratum.coverage["A"] == 1.0
    assert stratum.coverage["B"] == 0.0
    assert stratum.width["A"] == 2.0
    assert stratum.width["B"] == pytest.approx(0.4)
    # Paired coverage difference is A minus B.
    assert stratum.mean_coverage_diff == 1.0
    assert stratum.mean_width_diff == pytest.approx(1.6)


def test_population_estimates_use_N_over_n_weights_not_equal_pooling():
    # Two strata of very unequal N but equal retained sample size: the
    # population estimate must be dominated by the large stratum.
    diff_small = np.full(20, 1.0)  # N = 20, diff = +1
    diff_large = np.full(400, -1.0)  # N = 400, diff = -1
    all_diff = np.concatenate([diff_small, diff_large])
    strata = np.array([0] * 20 + [1] * 400)
    X = np.arange(420).reshape(-1, 1)

    audit = StratifiedStreamingPairedAudit(K=10, random_state=0)
    audit.update(X, np.zeros(420), strata)
    width_a = 5.0 + all_diff
    result = audit.compute_audit(
        PrescribedWidthModel(width_a),
        PrescribedWidthModel(np.full(420, 5.0)),
    )
    # Weighted by N_h: (20 * +1 + 400 * -1) / 420, not the raw sample mean.
    assert result.mean_width_diff == pytest.approx((20 - 400) / 420)


def test_identical_models_give_zero_paired_difference_and_zero_se():
    audit = StratifiedStreamingPairedAudit(K=15, random_state=2)
    X = np.arange(120).reshape(-1, 1)
    audit.update(X, rng_normal(120), np.zeros(120, dtype=int))
    model = FixedIntervalModel(1.0)
    result = audit.compute_audit(model, FixedIntervalModel(1.0))
    assert result.mean_coverage_diff == 0.0
    assert result.mean_width_diff == 0.0
    assert result.se_coverage_diff == 0.0
    assert result.se_width_diff == 0.0


def rng_normal(n):
    return np.random.default_rng(0).normal(size=n)


# --------------------------------------------------------------------------- #
# Standard error behavior
# --------------------------------------------------------------------------- #
def test_census_stratum_has_zero_standard_error():
    audit = StratifiedStreamingPairedAudit(K=100, random_state=0)
    audit.update(
        np.arange(5).reshape(-1, 1),
        np.zeros(5),
        np.zeros(5, dtype=int),
    )
    result = audit.compute_audit(FixedIntervalModel(1.0), FixedIntervalModel(2.0))
    stratum = result.strata[0]
    assert stratum.census is True
    assert stratum.se_estimable is True
    assert stratum.se_coverage_diff == 0.0
    assert stratum.se_width_diff == 0.0
    assert result.se_coverage_diff == 0.0


def test_single_sample_non_census_stratum_marks_se_unestimable():
    audit = StratifiedStreamingPairedAudit(K=1, random_state=0)
    audit.update(
        np.arange(10).reshape(-1, 1),
        np.zeros(10),
        np.zeros(10, dtype=int),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = audit.compute_audit(FixedIntervalModel(1.0), FixedIntervalModel(2.0))
    stratum = result.strata[0]
    assert stratum.census is False
    assert stratum.se_estimable is False
    assert np.isnan(stratum.se_coverage_diff)
    assert np.isnan(stratum.se_width_diff)
    assert result.unestimable_strata == [0]
    assert np.isnan(result.se_coverage_diff)
    assert np.isnan(result.se_width_diff)
    assert any("not estimable" in str(w.message) for w in caught)


def test_standard_error_matches_sampled_record_closed_form():
    # The implementation must apply the exact SRSWOR-with-FPC formula to
    # the variance of the *paired* differences on the retained records.
    rng = np.random.default_rng(7)
    d0 = rng.normal(2.0, 3.0, size=200)
    d1 = rng.normal(-1.0, 2.0, size=300)
    all_diff = np.concatenate([d0, d1])
    strata = np.array([0] * 200 + [1] * 300)
    X = np.arange(500).reshape(-1, 1)
    k0, k1 = 40, 60

    audit = StratifiedStreamingPairedAudit(K={0: k0, 1: k1}, default_K=10, random_state=3)
    audit.update(X, np.zeros(500), strata)
    result = audit.compute_audit(
        PrescribedWidthModel(10.0 + all_diff),
        PrescribedWidthModel(np.full(500, 10.0)),
    )
    # Recompute the expected variance from the records actually retained.
    variance = 0.0
    for stratum_result in result.strata:
        label = stratum_result.stratum
        n_h = stratum_result.n
        n_h_pop = stratum_result.N
        diffs = result.records[label]["width_diff"]
        s2 = diffs.var(ddof=1)
        variance += n_h_pop**2 * (1 - n_h / n_h_pop) * s2 / n_h
    expected_se = np.sqrt(variance) / sum(s.N for s in result.strata)
    assert result.se_width_diff == pytest.approx(expected_se)
    # Per-stratum SE follows the same formula independently.
    for stratum_result in result.strata:
        label = stratum_result.stratum
        diffs = result.records[label]["width_diff"]
        fpc = 1 - stratum_result.n / stratum_result.N
        expected = np.sqrt(fpc * diffs.var(ddof=1) / stratum_result.n)
        assert stratum_result.se_width_diff == pytest.approx(expected)


def test_standard_error_is_unbiased_in_monte_carlo():
    # Across repeated SRSWOR draws, the average reported SE must match the
    # empirical standard deviation of the stratified estimator.
    rng = np.random.default_rng(7)
    d0 = rng.normal(2.0, 3.0, size=200)
    d1 = rng.normal(-1.0, 2.0, size=300)
    all_diff = np.concatenate([d0, d1])
    strata = np.array([0] * 200 + [1] * 300)
    X = np.arange(500).reshape(-1, 1)
    k0, k1 = 40, 60

    estimates, standard_errors = [], []
    for seed in range(300):
        audit = StratifiedStreamingPairedAudit(K={0: k0, 1: k1}, default_K=10, random_state=seed)
        audit.update(X, np.zeros(500), strata)
        result = audit.compute_audit(
            PrescribedWidthModel(10.0 + all_diff),
            PrescribedWidthModel(np.full(500, 10.0)),
        )
        estimates.append(result.mean_width_diff)
        standard_errors.append(result.se_width_diff)
    empirical_sd = np.std(estimates, ddof=1)
    assert np.mean(standard_errors) == pytest.approx(empirical_sd, rel=0.08)


def test_variances_are_paired_not_sum_of_independent_variances():
    # Two strongly co-varying interval models: independent variance
    # addition would grossly overstate the paired-difference variance.
    # Widths stay strictly positive (intervals cannot have negative width).
    rng = np.random.default_rng(11)
    n = 200
    shared = np.abs(rng.normal(scale=5.0, size=n)) + 10.0
    width_b = 4.0 + shared
    width_a = width_b + 0.5  # constant paired width difference
    X = np.arange(n).reshape(-1, 1)
    audit = StratifiedStreamingPairedAudit(K=50, random_state=4)
    audit.update(X, np.zeros(n), np.zeros(n, dtype=int))
    result = audit.compute_audit(PrescribedWidthModel(width_a), PrescribedWidthModel(width_b))
    # Constant difference -> zero paired variance despite large margins.
    assert result.se_width_diff == pytest.approx(0.0, abs=1e-12)
    independent = np.sqrt(width_a.var(ddof=1) + width_b.var(ddof=1))
    assert independent > 1.0  # the wrong formula would be far from zero


# --------------------------------------------------------------------------- #
# Merging states from unequal, disjoint sources
# --------------------------------------------------------------------------- #
def _single_source_audit(source, start, n, k_cap, seed):
    audit = StratifiedStreamingPairedAudit(K=k_cap, random_state=seed)
    X = np.arange(start, start + n).reshape(-1, 1)
    audit.update(X, np.zeros(n), np.zeros(n, dtype=int), sources=source)
    return audit


def test_merge_rejects_overlapping_sources():
    a = _single_source_audit("s1", 0, 50, 10, 0)
    b = _single_source_audit("s1", 100, 30, 10, 1)
    with pytest.raises(SourceOverlapError, match="overlapping sources"):
        a.merge(b)


def test_merge_aggregates_population_counts_and_disjoint_sources():
    a = _single_source_audit("s1", 0, 400, 50, 0)
    b = _single_source_audit("s2", 1000, 100, 50, 1)
    merged = StratifiedStreamingPairedAudit.merge_states([a, b], random_state=2)
    assert merged.population_size(0) == 500
    assert merged.source_counts() == {"s1": 400, "s2": 100}
    _, _, src = merged.retained_records(0)
    assert set(src) == {"s1", "s2"}


def test_merged_sample_is_uniform_over_pooled_population():
    # Unequal source sizes (400 vs 100), pooled sample of 50: every one of
    # the 500 population records must have inclusion probability ~ 0.1.
    reps = 300
    counts = np.zeros(500)
    for rep in range(reps):
        a = _single_source_audit("s1", 0, 400, 50, rep)
        b = _single_source_audit("s2", 1000, 100, 50, rep + 10_000)
        merged = StratifiedStreamingPairedAudit.merge_states([a, b], random_state=rep + 50_000)
        X_ret, _, _ = merged.retained_records(0)
        assert len(X_ret) == 50
        ids = np.array([int(rec[0]) for rec in X_ret])
        pooled_idx = np.where(ids >= 1000, ids - 1000 + 400, ids)
        assert len(set(pooled_idx)) == 50  # without replacement
        counts[pooled_idx] += 1
    target_rate = 50 / 500
    rate_s1 = counts[:400].mean() / reps
    rate_s2 = counts[400:].mean() / reps
    assert rate_s1 == pytest.approx(target_rate, abs=0.02)
    assert rate_s2 == pytest.approx(target_rate, abs=0.02)
    # Source composition follows population shares, not equal halves.
    s1_share = counts[:400].sum() / counts.sum()
    assert s1_share == pytest.approx(0.8, abs=0.05)


def test_merge_does_not_concatenate_equal_numbers_per_source():
    # Small source would otherwise contribute half of an equal concat.
    a = _single_source_audit("big", 0, 900, 30, 0)
    b = _single_source_audit("small", 5000, 100, 30, 1)
    merged = StratifiedStreamingPairedAudit.merge_states([a, b], random_state=2)
    _, _, src = merged.retained_records(0)
    n_small = int(np.sum(np.asarray(src) == "small"))
    # Population share of small source is 10%; equal concat would give 50%.
    assert n_small / len(src) < 0.25


def test_three_way_merge_keeps_uniform_inclusion():
    reps = 200
    counts = np.zeros(1000)
    for rep in range(reps):
        a = _single_source_audit("s1", 0, 700, 40, rep)
        b = _single_source_audit("s2", 5000, 200, 40, rep + 10_000)
        c = _single_source_audit("s3", 9000, 100, 40, rep + 20_000)
        merged = StratifiedStreamingPairedAudit.merge_states([a, b, c], random_state=rep + 50_000)
        X_ret, _, _ = merged.retained_records(0)
        ids = np.array([int(rec[0]) for rec in X_ret])
        pooled = np.where(
            ids >= 9000,
            ids - 9000 + 900,
            np.where(ids >= 5000, ids - 5000 + 700, ids),
        )
        counts[pooled] += 1
    target = reps * 40 / 1000
    assert counts[:700].mean() == pytest.approx(target, rel=0.15)
    assert counts[700:900].mean() == pytest.approx(target, rel=0.15)
    assert counts[900:].mean() == pytest.approx(target, rel=0.15)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_save_and_load_preserves_state_and_resumes_rng(tmp_path):
    audit = StratifiedStreamingPairedAudit(K=10, random_state=42)
    X = np.arange(120).reshape(-1, 1)
    strata = np.tile(np.array([0, 1, 2]), 40)
    audit.update(X, np.zeros(120), strata, sources="src1")
    audit.save(tmp_path)

    reopened = StratifiedStreamingPairedAudit.load(tmp_path)
    assert reopened.rng.bit_generator.state == audit.rng.bit_generator.state
    assert reopened.strata == audit.strata
    for label in audit.strata:
        assert reopened.population_size(label) == audit.population_size(label)
        before, _, _ = audit.retained_records(label)
        after, _, _ = reopened.retained_records(label)
        np.testing.assert_array_equal(
            np.asarray(list(before), dtype=float),
            np.asarray(list(after), dtype=float),
        )

    # Streaming more data after reload reproduces the exact same reservoir.
    audit.update(X[:20], np.zeros(20), strata[:20], sources="src2")
    reopened.update(X[:20], np.zeros(20), strata[:20], sources="src2")
    for label in audit.strata:
        before, _, _ = audit.retained_records(label)
        after, _, _ = reopened.retained_records(label)
        np.testing.assert_array_equal(
            np.asarray(list(before), dtype=float),
            np.asarray(list(after), dtype=float),
        )


def test_load_rejects_missing_or_corrupt_state(tmp_path):
    with pytest.raises(AuditStateError):
        StratifiedStreamingPairedAudit.load(tmp_path / "absent")
    audit = StratifiedStreamingPairedAudit(K=5, random_state=0)
    audit.update(np.arange(4).reshape(-1, 1), np.zeros(4), np.zeros(4, int))
    audit.save(tmp_path / "state")
    import json
    import os

    meta = tmp_path / "state" / "audit_state.json"
    payload = json.loads(meta.read_text())
    payload["version"] = 999
    meta.write_text(json.dumps(payload))
    with pytest.raises(AuditStateError):
        StratifiedStreamingPairedAudit.load(tmp_path / "state")
    assert os.path.isfile(tmp_path / "state" / "audit_data.npz")


def test_non_string_stratum_labels_round_trip(tmp_path):
    audit = StratifiedStreamingPairedAudit(K=3, random_state=0)
    audit.update(
        np.arange(12).reshape(-1, 1),
        np.zeros(12),
        np.array([0, 1, 2.5, 0] * 3),
    )
    audit.save(tmp_path)
    reopened = StratifiedStreamingPairedAudit.load(tmp_path)
    assert all(isinstance(label, (int, float)) for label in reopened.strata)
    assert set(reopened.strata) == {0, 1, 2.5}


# --------------------------------------------------------------------------- #
# Record details and model interface
# --------------------------------------------------------------------------- #
def test_record_details_contain_paired_predictions_and_weights():
    audit = StratifiedStreamingPairedAudit(K=6, random_state=0)
    audit.update(
        np.arange(20).reshape(-1, 1),
        np.full(20, 0.5),
        np.zeros(20, dtype=int),
        sources="src-x",
    )
    result = audit.compute_audit(FixedIntervalModel(1.0), FixedIntervalModel(2.0), alpha=0.1)
    details = result.records[0]
    assert len(details["y"]) == 6
    np.testing.assert_array_equal(
        details["coverage_diff"], details["coverage_A"] - details["coverage_B"]
    )
    np.testing.assert_array_equal(details["width_diff"], details["width_A"] - details["width_B"])
    assert np.all(details["sampling_weight"] == 20 / 6)
    frame = result.records_dataframe()
    assert {"stratum", "source", "width_diff", "sampling_weight"}.issubset(frame.columns)
    summary = result.strata_dataframe()
    assert set(summary["N"]) == {20}


def test_predict_without_alpha_and_bad_predict_output():
    audit = StratifiedStreamingPairedAudit(K=4, random_state=0)
    audit.update(np.arange(8).reshape(-1, 1), np.zeros(8), np.zeros(8, int))
    result = audit.compute_audit(NoAlphaModel(1.0), NoAlphaModel(1.5))
    assert result.strata[0].n == 4
    with pytest.raises(TypeError, match="y_pred_lower"):
        audit.compute_audit(BadModel(), FixedIntervalModel(1.0))


def test_dataframes_work_with_custom_model_names():
    audit = StratifiedStreamingPairedAudit(K=4, random_state=0)
    audit.update(np.arange(8).reshape(-1, 1), np.zeros(8), np.zeros(8, int))
    result = audit.compute_audit(
        FixedIntervalModel(1.0),
        FixedIntervalModel(1.5),
        name_a="narrow",
        name_b="wide",
    )
    assert result.model_names == {"A": "narrow", "B": "wide"}
    strata_frame = result.strata_dataframe()
    assert {"coverage_A", "coverage_B", "width_A", "width_B"}.issubset(strata_frame.columns)
    assert strata_frame["width_A"].iloc[0] == pytest.approx(2.0)
    records_frame = result.records_dataframe()
    assert len(records_frame) == 4
    assert set(records_frame["source"]) == {"default"}


def test_audit_requires_ingested_records():
    audit = StratifiedStreamingPairedAudit(K=5)
    with pytest.raises(AuditStateError):
        audit.compute_audit(FixedIntervalModel(1.0), FixedIntervalModel(2.0))


# --------------------------------------------------------------------------- #
# Integration with a real fitted puncc conformal regressor (sklearn backend)
# --------------------------------------------------------------------------- #
def test_paired_audit_with_real_fitted_splitcp_models():
    from sklearn.linear_model import LinearRegression

    from deel.puncc.api.prediction import BasePredictor
    from deel.puncc.regression import SplitCP

    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(scale=0.5, size=n)

    def fit_model():
        model = SplitCP(BasePredictor(LinearRegression(), is_trained=False))
        model.fit(
            X_fit=X[:200],
            y_fit=y[:200],
            X_calib=X[200:300],
            y_calib=y[200:300],
        )
        return model

    model_a, model_b = fit_model(), fit_model()
    audit = StratifiedStreamingPairedAudit(K=40, random_state=0)
    strata = rng.integers(0, 3, size=100)
    audit.update(X[300:], y[300:], strata, sources="holdout")
    result = audit.compute_audit(model_a, model_b, alpha=0.1)

    # Two identical-fitting models must agree on every retained record.
    assert result.mean_coverage_diff == pytest.approx(0.0, abs=1e-12)
    assert result.mean_width_diff == pytest.approx(0.0, abs=1e-12)
    for name in ("A", "B"):
        assert 0.0 <= result.overall_coverage[name] <= 1.0
        assert result.overall_width[name] > 0.0
    assert sum(s.n for s in result.strata) <= 3 * 40
    assert sum(s.N for s in result.strata) == 100
