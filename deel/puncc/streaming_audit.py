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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Stratified streaming paired audit of two fitted interval models.

When two interval models are compared on data coming from heterogeneous
sources, naively pooling / averaging per-source or per-model results is
biased whenever source sizes differ. This module implements a
**stratified, streaming, paired** audit that is valid under unequal source
sizes:

* Records :math:`(X, y)` arrive as a stream and are stratified. Within
  every stratum at most ``K`` records are kept by *reservoir sampling*
  (Vitter's Algorithm R), i.e. a simple random sample without
  replacement (SRSWOR) of the stream seen so far, while the full
  population count :math:`N_h` is retained for every stratum. Memory
  never exceeds :math:`K` records per stratum plus the necessary counts
  and source metadata; the complete history is never stored.
* States built on different sources can be merged. Records from each
  input state are combined through a two-phase stratified sub-sampling
  (population-proportional allocation of the pooled sample size, then a
  uniform without-replacement sub-draw inside every source), which
  yields a SRSWOR of the *pooled* population: the merged sample stays
  uniformly distributed over the pooled population instead of
  concatenating equal numbers of records per source. States with
  overlapping source identifiers are rejected, so that the same
  population records can never be counted twice.
* Both already-fitted scalar interval models predict on the **same
  retained records** and the native puncc metrics
  :func:`~deel.puncc.metrics.regression_mean_coverage` and
  :func:`~deel.puncc.metrics.regression_sharpness` are invoked.
* The population mean of the paired coverage difference and width
  difference is estimated through the stratified Horvitz-Thompson
  estimator. Variances are computed on the **paired differences** (the
  two models' variances are never added independently) with the finite
  population correction (FPC). A non-censused stratum that ends up with
  a single sampled record has no estimable standard error; a censused
  stratum (:math:`n_h = N_h`) has zero standard error.

The audit state can be saved to disk and reopened to continue streaming,
and the actual sampled record details (features, labels, source,
per-record paired indicators, per-stratum :math:`N_h/n_h` weights) are
fully exposed.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from deel.puncc.metrics import regression_mean_coverage
from deel.puncc.metrics import regression_sharpness

logger = logging.getLogger(__name__)

# File names used by the on-disk state format.
_META_FILENAME = "audit_state.json"
_DATA_FILENAME = "audit_data.npz"
_STATE_VERSION = 1


class _AuditStateEncoder(json.JSONEncoder):
    """JSON encoder that preserves numpy scalars and large uint64 values."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return {
                "__ndarray__": [self._scalar(v) for v in o.reshape(-1)],
                "shape": list(o.shape),
                "dtype": str(o.dtype),
            }
        if isinstance(o, np.generic):
            return self._scalar(o)
        return super().default(o)

    @staticmethod
    def _scalar(value: np.generic) -> Any:
        # Large uint64 values exceed signed int64 and must be serialized
        # as tagged strings to survive the JSON round trip.
        if np.issubdtype(type(value), np.unsignedinteger) and value > np.iinfo(np.int64).max:
            return {"__uint64__": str(int(value))}
        return value.item()


def _audit_state_hook(obj: Dict[str, Any]) -> Any:
    """Counterpart of :class:`_AuditStateEncoder` on load."""
    if "__uint64__" in obj:
        return int(obj["__uint64__"])
    if "__ndarray__" in obj:
        dtype = np.dtype(obj["dtype"])
        return np.array(obj["__ndarray__"], dtype=dtype).reshape(obj["shape"])
    return obj


class SourceOverlapError(ValueError):
    """Raised when two audit states to be merged share a source id.

    Overlapping sources mean the same population records could be counted
    twice, which destroys the without-replacement sampling design; merging
    is therefore refused.
    """


class AuditStateError(RuntimeError):
    """Raised when an audit state file is missing, corrupted or invalid."""


def _as_1d_array(values: Any, name: str) -> np.ndarray:
    """Coerce ``values`` to a 1D ndarray (object dtype if needed)."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim < 1:
        raise ValueError(f"`{name}` must contain at least one dimension (got {arr.shape}).")
    return arr


def _records_of(X: Any) -> np.ndarray:
    """Split a features batch into one object-array entry per record.

    Feature matrices are sliced row-wise; scalar / 1D inputs become
    one-element records as well. The object array keeps arbitrary record
    shapes (including non-numeric records) in a single reservoir.
    """
    arr = np.asarray(X)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.dtype == object and arr.ndim == 1:
        return arr
    out = np.empty(arr.shape[0], dtype=object)
    for i in range(arr.shape[0]):
        out[i] = arr[i]
    return out


def _stack_records(X_records: np.ndarray) -> np.ndarray:
    """Reconstruct a model-ready batch from per-record reservoir entries.

    The reservoir stores one object entry per record (each entry being a
    row of the original features). Before calling a model's ``predict``,
    the rows are stacked back into a numeric ndarray with shape
    ``(n_records, ...)``; heterogeneous / non-numeric records fall back to
    the 1D object array.
    """
    try:
        return np.stack([np.asarray(rec) for rec in X_records], axis=0)
    except (ValueError, TypeError):
        return X_records


def _coerce_strata(strata: Any, n_records: int) -> np.ndarray:
    """Normalize stratum labels to a 1D array of ``n_records`` labels."""
    if np.isscalar(strata):
        return np.full(n_records, strata)
    labels = _as_1d_array(strata, "strata")
    if labels.shape[0] != n_records:
        raise ValueError(
            f"`strata` has length {labels.shape[0]} but {n_records} records were provided."
        )
    return labels


def _coerce_sources(sources: Any, n_records: int, default_source: str) -> np.ndarray:
    """Normalize source ids to a 1D array; fill missing values with default."""
    if sources is None:
        return np.full(n_records, default_source, dtype=object)
    if isinstance(sources, str):
        return np.full(n_records, sources, dtype=object)
    src = np.asarray(sources, dtype=object).reshape(-1)
    if src.shape[0] != n_records:
        raise ValueError(
            f"`sources` has length {src.shape[0]} but {n_records} records were provided."
        )
    return np.array([default_source if s is None else s for s in src], dtype=object)


class _StratumState:
    """Streaming reservoir and bookkeeping for one stratum.

    The reservoir stores at most ``capacity`` (the per-stratum ``K``)
    records as three parallel object arrays ``X`` / ``y`` / ``source``.
    ``population_count`` is :math:`N_h`, the (exact) number of records
    observed for this stratum. ``source_counts`` holds the exact number
    of observed records per source, which is required both for overlap
    rejection and for weighted-reservoir merges.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.population_count: int = 0
        self.X: List[Any] = []
        self.y: List[Any] = []
        self.source: List[Any] = []
        self.source_counts: Dict[Any, int] = {}

    # ------------------------------------------------------------------ #
    # Streaming ingestion (Vitter's Algorithm R, uniform SRSWOR)
    # ------------------------------------------------------------------ #
    def add(
        self,
        x: Any,
        y: Any,
        source: Any,
        rng: np.random.Generator,
    ) -> None:
        """Ingest one record, updating the uniform reservoir."""
        self.population_count += 1
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        t = self.population_count
        if len(self.X) < self.capacity:
            self.X.append(x)
            self.y.append(y)
            self.source.append(source)
        else:
            # Uniform integer in [0, t-1]: replace with probability K/t.
            j = int(rng.integers(0, t))
            if j < self.capacity:
                self.X[j] = x
                self.y[j] = y
                self.source[j] = source

    # ------------------------------------------------------------------ #
    # Two-phase stratified merge (proportional allocation + SRS sub-draw)
    # ------------------------------------------------------------------ #
    def absorb(
        self,
        other: "_StratumState",
        rng: np.random.Generator,
    ) -> None:
        """Merge ``other`` into ``self`` through two-phase sub-sampling.

        The two input reservoirs are SRSWORs of their own (disjoint)
        source populations. A SRSWOR of the *pooled* population is
        reconstructed in two phases:

        1. **Allocation.** the pooled sample size :math:`n_{pooled}` is
           split across sources proportionally to their population
           sizes :math:`n^*_s = \\mathrm{round}(n_{pooled} N_s/N_{pooled})`
           (largest-remainder rounding, capped by each reservoir's
           available records and :math:`N_s`).
        2. **Sub-sampling.** :math:`n^*_s` records are drawn uniformly
           without replacement from each source's retained records.

        Conditional on the first-phase samples, this yields the same
        distribution as directly drawing a SRSWOR of the pooled
        population (every pooled record has inclusion probability
        :math:`n_{pooled}/N_{pooled}`). This deliberately replaces
        equal-size concatenation, which would over-represent small
        sources.
        """
        # Aggregate reservoir records and exact population counts by
        # source across both states (sources are disjoint by construction).
        records_by_source: Dict[Any, List[Tuple[Any, Any]]] = {}
        pop_by_source: Dict[Any, int] = {}
        for state in (self, other):
            for i, x_rec in enumerate(state.X):
                records_by_source.setdefault(state.source[i], []).append((x_rec, state.y[i]))
            for src, cnt in state.source_counts.items():
                pop_by_source[src] = pop_by_source.get(src, 0) + cnt

        pooled_N = sum(pop_by_source.values())
        # Keep as many pooled records as the merged capacity allows, but
        # never more than currently available.
        n_pooled = min(self.capacity, sum(len(v) for v in records_by_source.values()))
        allocations = self._largest_remainder_allocation(pop_by_source, n_pooled, records_by_source)

        new_X: List[Any] = []
        new_y: List[Any] = []
        new_source: List[Any] = []
        for src, n_star in allocations.items():
            if n_star == 0:
                continue
            pool = records_by_source[src]
            chosen_idx = rng.choice(len(pool), size=min(n_star, len(pool)), replace=False)
            for idx in chosen_idx:
                x_rec, y_rec = pool[int(idx)]
                new_X.append(x_rec)
                new_y.append(y_rec)
                new_source.append(src)

        self.X = new_X
        self.y = new_y
        self.source = new_source
        self.population_count = pooled_N
        self.source_counts = pop_by_source

    @staticmethod
    def _largest_remainder_allocation(
        pop_by_source: Dict[Any, int],
        n_pooled: int,
        records_by_source: Dict[Any, List[Tuple[Any, Any]]],
    ) -> Dict[Any, int]:
        """Allocate ``n_pooled`` slots proportionally to population sizes.

        Largest-remainder (Hare) rounding on the population shares, with
        each source's allocation bounded by both its population size
        and the number of retained records actually available. Any
        unallocatable remainder is redistributed to sources that still
        have capacity.
        """
        total_N = sum(pop_by_source.values())
        max_by_source = {
            src: min(pop_by_source[src], len(records_by_source.get(src, [])))
            for src in pop_by_source
        }
        # Feasible pooled size given per-source caps.
        n_pooled = min(n_pooled, sum(max_by_source.values()))

        raw = {src: n_pooled * pop_by_source[src] / total_N for src in pop_by_source}
        allocation = {src: int(np.floor(raw[src])) for src in pop_by_source}
        # Respect caps at the floor stage.
        for src in allocation:
            allocation[src] = min(allocation[src], max_by_source[src])

        remaining = n_pooled - sum(allocation.values())
        # Largest fractional remainders first, among sources with room.
        order = sorted(
            pop_by_source.keys(),
            key=lambda s: (raw[s] - np.floor(raw[s])),
            reverse=True,
        )
        while remaining > 0:
            progressed = False
            for src in order:
                if remaining == 0:
                    break
                if allocation[src] < max_by_source[src]:
                    allocation[src] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                # No source can take more records; stop (should only
                # happen if caps bind everywhere).
                break
        return allocation

    # ------------------------------------------------------------------ #
    @property
    def sampled_count(self) -> int:
        """Number of records currently held in the reservoir."""
        return len(self.X)

    @property
    def is_census(self) -> bool:
        """True iff the whole stratum population is retained."""
        return 0 < self.population_count <= len(self.X)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X, y, source)`` as parallel 1D object arrays.

        Always 1D with one entry per retained record. ``np.array`` on
        uniform-shaped numeric rows would stack them into a 2D array,
        which breaks per-record indexing once the batch is flattened
        inside a model.
        """
        x_obj = np.empty(len(self.X), dtype=object)
        x_obj[:] = self.X
        return (
            x_obj,
            np.array(self.y, dtype=object),
            np.array(self.source, dtype=object),
        )

    def replace_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
        source: np.ndarray,
    ) -> None:
        """Overwrite the reservoir contents (used when reloading state)."""
        self.X = list(X)
        self.y = list(y)
        self.source = list(source)


@dataclass
class StratumAuditResult:
    """Paired audit statistics for one stratum.

    :param stratum: stratum label.
    :param int N: population size :math:`N_h` (exact count of all seen
        records).
    :param int n: retained sample size :math:`n_h` (``<= K`` and
        ``<= N_h``).
    :param bool census: ``True`` iff :math:`n_h = N_h` (full census of
        the stratum).
    :param bool se_estimable: ``False`` iff the stratum is not censused
        and :math:`n_h = 1`: no variance can be estimated from a single
        sampled unit.
    :param float weight: sampling weight :math:`N_h/n_h` applied to each
        retained record.
    :param dict coverage: per-model empirical coverage on the retained
        records, computed with the native puncc metric.
    :param dict width: per-model mean interval width on the retained
        records, computed with the native puncc metric.
    :param float mean_coverage_diff: sample mean of the paired coverage
        difference (model A minus model B).
    :param float mean_width_diff: sample mean of the paired width
        difference (model A minus model B).
    :param se_coverage_diff: finite-population standard error of the
        stratum mean coverage difference (0 for a census, NaN when not
        estimable).
    :param se_width_diff: finite-population standard error of the
        stratum mean width difference (0 for a census, NaN when not
        estimable).
    """

    stratum: Any
    N: int
    n: int
    census: bool
    se_estimable: bool
    weight: float
    coverage: Dict[str, float]
    width: Dict[str, float]
    mean_coverage_diff: float
    mean_width_diff: float
    se_coverage_diff: float
    se_width_diff: float


@dataclass
class PairedAuditResult:
    """Population-level paired audit result.

    :param alpha: significance level used for the interval predictions.
    :param dict overall_coverage: population-estimated coverage of each
        model (stratified weighted mean).
    :param dict overall_width: population-estimated mean interval width
        of each model (stratified weighted mean).
    :param float mean_coverage_diff: estimated population mean of the
        paired coverage difference (model A minus model B), i.e. the
        Horvitz-Thompson estimator
        :math:`\\sum_h N_h \\bar d_h / \\sum_h N_h`.
    :param float mean_width_diff: estimated population mean of the
        paired width difference (model A minus model B).
    :param se_coverage_diff: finite-population standard error of
        ``mean_coverage_diff``. It is NaN if at least one non-census
        stratum has a single sampled unit (its variance cannot be
        estimated); see ``unestimable_strata``.
    :param se_width_diff: finite-population standard error of
        ``mean_width_diff`` (NaN under the same condition).
    :param list unestimable_strata: labels of strata whose standard
        error is not estimable.
    :param list strata: per-stratum :class:`StratumAuditResult`.
    :param dict records: actual sampled record details, keyed by
        stratum. Each entry holds the retained ``X`` / ``y`` /
        ``source`` arrays together with per-record predictions and
        paired coverage / width differences.
    :param dict model_names: mapping ``{"A": name_a, "B": name_b}``.
    """

    alpha: Optional[float]
    overall_coverage: Dict[str, float]
    overall_width: Dict[str, float]
    mean_coverage_diff: float
    mean_width_diff: float
    se_coverage_diff: float
    se_width_diff: float
    unestimable_strata: List[Any]
    strata: List[StratumAuditResult] = field(default_factory=list)
    records: Dict[Any, Dict[str, np.ndarray]] = field(default_factory=dict)
    model_names: Dict[str, str] = field(default_factory=dict)

    # --------------------------------------------------------------- #
    # Convenience views
    # --------------------------------------------------------------- #
    def strata_dataframe(self):
        """Return the per-stratum summary as a pandas DataFrame.

        The per-model columns keep the stable ``coverage_A`` /
        ``coverage_B`` / ``width_A`` / ``width_B`` names; the actual
        model display names are available through :attr:`model_names`.

        :raises ImportError: if pandas is not installed.
        """
        import pandas  # pylint: disable=import-outside-toplevel

        name_a = self.model_names.get("A", "A")
        name_b = self.model_names.get("B", "B")
        return pandas.DataFrame(
            [
                {
                    "stratum": s.stratum,
                    "N": s.N,
                    "n": s.n,
                    "census": s.census,
                    "se_estimable": s.se_estimable,
                    "weight": s.weight,
                    "coverage_A": s.coverage[name_a],
                    "coverage_B": s.coverage[name_b],
                    "width_A": s.width[name_a],
                    "width_B": s.width[name_b],
                    "mean_coverage_diff": s.mean_coverage_diff,
                    "mean_width_diff": s.mean_width_diff,
                    "se_coverage_diff": s.se_coverage_diff,
                    "se_width_diff": s.se_width_diff,
                }
                for s in self.strata
            ]
        )

    def records_dataframe(self):
        """Flatten all retained-record details into a pandas DataFrame.

        :raises ImportError: if pandas is not installed.
        """
        import pandas  # pylint: disable=import-outside-toplevel

        frames = []
        for stratum, payload in self.records.items():
            frame = pandas.DataFrame(
                {
                    "stratum": stratum,
                    "source": list(payload["source"]),
                    "y": list(payload["y"]),
                    "y_pred_A": list(payload["y_pred_A"]),
                    "lower_A": list(payload["lower_A"]),
                    "upper_A": list(payload["upper_A"]),
                    "y_pred_B": list(payload["y_pred_B"]),
                    "lower_B": list(payload["lower_B"]),
                    "upper_B": list(payload["upper_B"]),
                    "coverage_A": list(payload["coverage_A"]),
                    "coverage_B": list(payload["coverage_B"]),
                    "coverage_diff": list(payload["coverage_diff"]),
                    "width_A": list(payload["width_A"]),
                    "width_B": list(payload["width_B"]),
                    "width_diff": list(payload["width_diff"]),
                    "sampling_weight": list(payload["sampling_weight"]),
                }
            )
            frames.append(frame)
        if not frames:
            return pandas.DataFrame()
        return pandas.concat(frames, ignore_index=True)


class StratifiedStreamingPairedAudit:
    """Streaming stratified paired audit of two interval models.

    Two *already fitted* scalar interval models are compared on streamed
    :math:`(X, y)` records. Within each stratum, at most ``K`` records
    are kept as a uniform sample without replacement (reservoir
    sampling) and the exact population size :math:`N_h` is counted. The
    models only run on the retained records, at audit time, on the exact
    same batch of records (paired design).

    :param K: maximum number of records retained per stratum. Either a
        positive integer applied to every stratum, or a mapping
        ``{stratum_label: capacity}`` (strata absent from the mapping
        fall back to ``default_K``).
    :param str default_source: source id attached to ingested records
        when no explicit ``sources`` argument is given.
    :param default_K: fallback capacity for strata missing from a
        dict-valued ``K``. Defaults to ``K`` itself when ``K`` is an
        int; required when ``K`` is a mapping.
    :param random_state: seed or :class:`numpy.random.Generator` used
        for the reservoir randomness (reproducible streaming and
        merges). The generator state is persisted with the audit state.

    Example::

        from deel.puncc.streaming_audit import StratifiedStreamingPairedAudit

        audit = StratifiedStreamingPairedAudit(K=50, random_state=0)
        for X_batch, y_batch, strata_batch, source_batch in stream:
            audit.update(X_batch, y_batch, strata_batch, sources=source_batch)

        result = audit.compute_audit(model_a, model_b, alpha=0.1)
        print(result.mean_coverage_diff, result.se_coverage_diff)
        audit.save("audit_run/")           # persist and continue later
        audit2 = StratifiedStreamingPairedAudit.load("audit_run/")
    """

    def __init__(
        self,
        K: Union[int, Dict[Any, int]] = 100,
        *,
        default_source: str = "default",
        default_K: Optional[int] = None,
        random_state: Union[None, int, np.random.Generator] = None,
    ) -> None:
        self._capacity_map: Optional[Dict[Any, int]]
        if isinstance(K, dict):
            if default_K is None:
                raise ValueError(
                    "`default_K` is required when `K` is provided as a per-stratum mapping."
                )
            self._capacity_map = {k: int(v) for k, v in K.items()}
            self._default_capacity = int(default_K)
        else:
            self._capacity_map = None
            self._default_capacity = int(K) if default_K is None else int(default_K)
        if self._default_capacity <= 0:
            raise ValueError("Per-stratum capacity K must be positive.")
        for cap in (self._capacity_map or {}).values():
            if cap <= 0:
                raise ValueError("Per-stratum capacity K must be positive.")

        self.default_source = str(default_source)
        if isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(random_state)

        self._strata: Dict[Any, _StratumState] = {}

    # ------------------------------------------------------------------ #
    # Properties / introspection
    # ------------------------------------------------------------------ #
    @property
    def strata(self) -> Tuple[Any, ...]:
        """Labels of strata observed so far."""
        return tuple(self._strata.keys())

    @property
    def rng(self) -> np.random.Generator:
        """Underlying random generator (state is persisted on save)."""
        return self._rng

    def capacity(self, stratum: Any) -> int:
        """Return the reservoir capacity configured for ``stratum``."""
        if self._capacity_map is not None and stratum in self._capacity_map:
            return self._capacity_map[stratum]
        return self._default_capacity

    def population_size(self, stratum: Any) -> int:
        """Exact population count :math:`N_h` of ``stratum``."""
        return self._strata[stratum].population_count if stratum in self._strata else 0

    def sample_size(self, stratum: Any) -> int:
        """Number of records currently retained for ``stratum``."""
        return self._strata[stratum].sampled_count if stratum in self._strata else 0

    def source_counts(self, stratum: Optional[Any] = None) -> Dict[Any, int]:
        """Exact per-source population counts.

        :param stratum: if provided, counts for that stratum only;
            otherwise counts aggregated over all strata.
        """
        aggregated: Dict[Any, int] = {}
        targets = [self._strata[stratum]] if stratum is not None else self._strata.values()
        for state in targets:
            for src, cnt in state.source_counts.items():
                aggregated[src] = aggregated.get(src, 0) + cnt
        return aggregated

    @property
    def sources(self) -> Tuple[Any, ...]:
        """All source ids registered in this audit."""
        return tuple(self.source_counts().keys())

    def retained_records(self, stratum: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the retained ``(X, y, source)`` arrays for a stratum.

        Only the retained records (at most ``K``) are ever materialized;
        the full stream history is not stored.
        """
        if stratum not in self._strata:
            raise KeyError(f"Unknown stratum {stratum!r}.")
        return self._strata[stratum].to_arrays()

    # ------------------------------------------------------------------ #
    # Streaming update
    # ------------------------------------------------------------------ #
    def update(
        self,
        X: Sequence[Any],
        y: Sequence[Any],
        strata: Union[Any, Sequence[Any]],
        *,
        sources: Union[None, str, Sequence[Any]] = None,
    ) -> "StratifiedStreamingPairedAudit":
        """Ingest one batch of streamed :math:`(X, y)` records.

        :param X: record features, one entry per record (array-like).
        :param y: record labels, one entry per record.
        :param strata: stratum label per record, or a single scalar label
            applying to the whole batch.
        :param sources: optional source id per record, a single source id
            for the whole batch, or ``None`` to use ``default_source``.

        :returns: ``self``, so calls can be chained.
        """
        X_arr = _records_of(X)
        y_arr = _as_1d_array(np.asarray(y, dtype=object), "y")
        n_records = X_arr.shape[0]
        if y_arr.shape[0] != n_records:
            raise ValueError(
                "`X` and `y` must contain the same number of records "
                f"({n_records} vs {y_arr.shape[0]})."
            )
        labels = _coerce_strata(strata, n_records)
        src_arr = _coerce_sources(sources, n_records, self.default_source)

        for i in range(n_records):
            label = labels[i]
            if label not in self._strata:
                self._strata[label] = _StratumState(self.capacity(label))
            self._strata[label].add(X_arr[i], y_arr[i], src_arr[i], self._rng)
        return self

    # ------------------------------------------------------------------ #
    # Merging states from different sources
    # ------------------------------------------------------------------ #
    def merged_sources(self, other: "StratifiedStreamingPairedAudit") -> Tuple[Any, ...]:
        """Return source ids present in both audit states."""
        return tuple(
            sorted(
                set(self.source_counts()).intersection(other.source_counts()),
                key=str,
            )
        )

    def merge(
        self,
        other: "StratifiedStreamingPairedAudit",
    ) -> "StratifiedStreamingPairedAudit":
        """Merge ``other`` into this audit (in place) and return ``self``.

        The two states must represent **disjoint sources**: any shared
        source id raises :class:`SourceOverlapError`. The two-phase
        stratified merge (population-proportional allocation followed by
        a uniform sub-draw per source) gives a uniform without-replacement
        sample of the pooled population, so unequal source sizes are
        respected — records are never concatenated in equal numbers per
        source.
        """
        overlap = self.merged_sources(other)
        if overlap:
            raise SourceOverlapError(
                "Refusing to merge audits with overlapping sources "
                f"(would double-count population records): {list(overlap)}."
            )

        for label, other_state in other._strata.items():
            if label not in self._strata:
                self._strata[label] = _StratumState(max(self.capacity(label), other_state.capacity))
            self._strata[label].absorb(other_state, self._rng)
        return self

    @classmethod
    def merge_states(
        cls,
        audits: Sequence["StratifiedStreamingPairedAudit"],
        *,
        random_state: Union[None, int, np.random.Generator] = None,
    ) -> "StratifiedStreamingPairedAudit":
        """Merge several audits of disjoint sources into a new audit.

        :param audits: audits to merge. Their source ids must be pairwise
            disjoint, otherwise :class:`SourceOverlapError` is raised.
        :param random_state: generator for the merge sub-sampling. If
            ``None``, a fresh non-deterministic generator is used.
        """
        if not audits:
            raise ValueError("At least one audit is required to merge.")
        first = audits[0]
        # Build the merged capacity map from all participants.
        if any(a._capacity_map is not None for a in audits):
            cap_map: Dict[Any, int] = {}
            defaults = [a._default_capacity for a in audits]
            for audit in audits:
                if audit._capacity_map is not None:
                    cap_map.update(audit._capacity_map)
            merged = cls(
                K=cap_map,
                default_K=max(defaults),
                default_source=first.default_source,
                random_state=random_state,
            )
        else:
            merged = cls(
                K=max(a._default_capacity for a in audits),
                default_source=first.default_source,
                random_state=random_state,
            )
        # Absorb every input audit. The first absorption starts from an
        # empty state, so the first audit's reservoir is carried through
        # the same weighted-reservoir mechanism (with its N_s/n_s keys).
        for audit in audits:
            merged.merge(audit)
        return merged

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, "os.PathLike[str]"]) -> None:
        """Persist the audit state under directory ``path``.

        The format is a JSON metadata file plus an NPZ data file; only
        the retained records (at most ``K`` per stratum), the population
        counts and the source metadata are stored. The directory is
        created if missing.
        """
        out_dir = os.fspath(path)
        os.makedirs(out_dir, exist_ok=True)

        labels = [self._jsonable(label) for label in self._strata.keys()]
        meta = {
            "version": _STATE_VERSION,
            "default_source": self.default_source,
            "default_capacity": self._default_capacity,
            "capacity_map": (
                {self._jsonable(k): v for k, v in self._capacity_map.items()}
                if self._capacity_map is not None
                else None
            ),
            "strata": labels,
            "rng_bit_generator": self._rng.bit_generator.__class__.__name__,
            "rng_state": self._rng.bit_generator.state,
            "population_counts": {
                lab: self._strata[label].population_count
                for lab, label in zip(labels, self._strata.keys())
            },
            "source_counts": {
                lab: {
                    self._jsonable(src): cnt
                    for src, cnt in self._strata[label].source_counts.items()
                }
                for lab, label in zip(labels, self._strata.keys())
            },
            "capacities": {
                lab: self._strata[label].capacity for lab, label in zip(labels, self._strata.keys())
            },
        }
        with open(os.path.join(out_dir, _META_FILENAME), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, cls=_AuditStateEncoder)

        arrays: Dict[str, np.ndarray] = {}
        for lab, label in zip(labels, self._strata.keys()):
            x_arr, y_arr, s_arr = self._strata[label].to_arrays()
            # Re-wrap as 1D object arrays explicitly: np.savez would
            # otherwise stack uniform-shaped records into a 2D array and
            # lose the one-entry-per-record reservoir semantics on reload.
            x_obj = np.empty(x_arr.shape[0], dtype=object)
            x_obj[:] = list(x_arr)
            arrays[f"X__{lab}"] = x_obj
            arrays[f"y__{lab}"] = y_arr
            arrays[f"source__{lab}"] = s_arr
        np.savez(os.path.join(out_dir, _DATA_FILENAME), **arrays)

    @classmethod
    def load(cls, path: Union[str, "os.PathLike[str]"]) -> "StratifiedStreamingPairedAudit":
        """Reopen a previously saved audit and continue streaming.

        :raises AuditStateError: if the directory does not contain a
            valid saved state or its version is unsupported.
        """
        in_dir = os.fspath(path)
        meta_path = os.path.join(in_dir, _META_FILENAME)
        data_path = os.path.join(in_dir, _DATA_FILENAME)
        if not os.path.isfile(meta_path) or not os.path.isfile(data_path):
            raise AuditStateError(f"No valid audit state found in {in_dir!r} (missing files).")
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh, object_hook=_audit_state_hook)
        if meta.get("version") != _STATE_VERSION:
            raise AuditStateError(
                f"Unsupported audit state version {meta.get('version')!r}; "
                f"expected {_STATE_VERSION}."
            )
        try:
            rng = np.random.Generator(np.random.PCG64())
            state = meta["rng_state"]
            # The custom JSON hooks restore the PCG64 integer arrays with
            # their original uint64 dtype; only flag fields need coercion.
            if "has_uint32" in state:
                state["has_uint32"] = int(state["has_uint32"])
            if "uinteger" in state:
                state["uinteger"] = int(state["uinteger"])
            rng.bit_generator.state = state
        except (ValueError, KeyError, OverflowError, TypeError) as exc:
            raise AuditStateError(f"Corrupted RNG state: {exc}") from exc

        capacity_map = meta.get("capacity_map")
        audit = cls(
            K=capacity_map if capacity_map is not None else meta["default_capacity"],
            default_source=meta["default_source"],
            default_K=meta["default_capacity"],
            random_state=rng,
        )
        with np.load(data_path, allow_pickle=True) as data:
            for lab in meta["strata"]:
                label = cls._dejsonable(lab)
                state_obj = _StratumState(meta["capacities"][lab])
                state_obj.population_count = int(meta["population_counts"][lab])
                state_obj.source_counts = {
                    cls._dejsonable(src): int(cnt)
                    for src, cnt in meta["source_counts"][lab].items()
                }
                state_obj.replace_arrays(
                    data[f"X__{lab}"],
                    data[f"y__{lab}"],
                    data[f"source__{lab}"],
                )
                audit._strata[label] = state_obj
        return audit

    @staticmethod
    def _jsonable(value: Any) -> str:
        """Stable JSON key representation of a (possibly non-string) label."""
        if isinstance(value, str):
            # Strings are stored verbatim, unless they collide with the
            # tagged representation used for non-string labels.
            return value
        if isinstance(value, np.generic):
            # Cast numpy scalars to native python types first, otherwise
            # json.dumps would serialize them through default=str and lose
            # the numeric type.
            value = value.item()
        return "tag:" + json.dumps(value)

    @staticmethod
    def _dejsonable(value: str) -> Any:
        if isinstance(value, str) and value.startswith("tag:"):
            return json.loads(value[4:])
        return value

    # ------------------------------------------------------------------ #
    # Paired audit on retained records
    # ------------------------------------------------------------------ #
    def compute_audit(
        self,
        model_a: Any,
        model_b: Any,
        *,
        alpha: Optional[float] = None,
        name_a: str = "A",
        name_b: str = "B",
        predict_kwargs: Optional[Dict[str, Any]] = None,
    ) -> PairedAuditResult:
        """Run both fitted models on the retained records and audit them.

        Both models are called on the **same** retained batch per
        stratum (paired design). Each model must expose

        ``predict(X, alpha=...) -> (y_pred, y_pred_lower, y_pred_upper)``

        as puncc's conformal regressors do; a model whose ``predict``
        takes no ``alpha`` is supported automatically. The native puncc
        metrics :func:`~deel.puncc.metrics.regression_mean_coverage` and
        :func:`~deel.puncc.metrics.regression_sharpness` are invoked on
        every retained batch.

        :param model_a: first fitted scalar interval model.
        :param model_b: second fitted scalar interval model.
        :param alpha: significance level forwarded to ``predict``.
        :param name_a: display name of model A.
        :param name_b: display name of model B.
        :param predict_kwargs: optional extra keyword arguments forwarded
            to both models' ``predict`` calls.

        :returns: population-level and per-stratum paired statistics,
            plus the full retained-record details.
        """
        if not self._strata:
            raise AuditStateError("No records have been ingested; nothing to audit.")
        predict_kwargs = predict_kwargs or {}

        # Population totals N = sum_h N_h
        total_N = sum(s.population_count for s in self._strata.values())

        stratum_results: List[StratumAuditResult] = []
        record_details: Dict[Any, Dict[str, np.ndarray]] = {}
        unestimable: List[Any] = []

        # Horvitz-Thompson accumulators for the population means.
        ht_cov_a = ht_cov_b = 0.0
        ht_width_a = ht_width_b = 0.0
        ht_cov_diff = ht_width_diff = 0.0
        # Variance accumulators of the *paired* differences.
        var_cov_num = 0.0  # sum_h N_h^2 * (1-f_h) s_d,h^2 / n_h
        var_width_num = 0.0

        for label, state in self._strata.items():
            X_ret, y_ret, src_ret = state.to_arrays()
            n_h = state.sampled_count
            N_h = state.population_count
            if n_h == 0:
                # Empty stratum: nothing observed, cannot contribute.
                continue

            lower_a, upper_a, yhat_a = self._intervals(model_a, X_ret, alpha, predict_kwargs)
            lower_b, upper_b, yhat_b = self._intervals(model_b, X_ret, alpha, predict_kwargs)
            y_true = np.asarray(y_ret, dtype=float).reshape(-1)

            # Native puncc metrics on the shared retained batch.
            cov_a = float(regression_mean_coverage(y_true, lower_a, upper_a))
            cov_b = float(regression_mean_coverage(y_true, lower_b, upper_b))
            width_a_native = float(regression_sharpness(lower_a, upper_a))
            width_b_native = float(regression_sharpness(lower_b, upper_b))

            # Per-record paired indicators (same elementwise formulas as
            # the native metrics, needed for the paired-difference design).
            rec_cov_a = ((y_true >= lower_a) & (y_true <= upper_a)).astype(float)
            rec_cov_b = ((y_true >= lower_b) & (y_true <= upper_b)).astype(float)
            rec_width_a = np.abs(upper_a - lower_a)
            rec_width_b = np.abs(upper_b - lower_b)
            cov_diff = rec_cov_a - rec_cov_b
            width_diff = rec_width_a - rec_width_b

            mean_cov_diff = float(cov_diff.mean())
            mean_width_diff = float(width_diff.mean())
            weight_h = N_h / n_h
            sampling_weights = np.full(n_h, weight_h)

            census = n_h >= N_h
            if census:
                # Full census: no sampling uncertainty on the stratum mean.
                se_cov = 0.0
                se_width = 0.0
                se_ok = True
            elif n_h == 1:
                # One single sampled unit, non-census: variance cannot be
                # estimated. Mark it explicitly rather than pretending 0.
                se_cov = float("nan")
                se_width = float("nan")
                se_ok = False
                unestimable.append(label)
            else:
                # SRSWOR variance of the sample mean, with FPC (1 - n/N):
                # Var(mean_d) = (1 - n/N) * s_d^2 / n, with the usual
                # unbiased sample variance s_d^2 of the *paired* diffs.
                fpc = 1.0 - n_h / N_h
                s2_cov = float(cov_diff.var(ddof=1))
                s2_width = float(width_diff.var(ddof=1))
                se_cov = float(np.sqrt(fpc * s2_cov / n_h))
                se_width = float(np.sqrt(fpc * s2_width / n_h))
                se_ok = True
                var_cov_num += (N_h**2) * fpc * s2_cov / n_h
                var_width_num += (N_h**2) * fpc * s2_width / n_h

            stratum_results.append(
                StratumAuditResult(
                    stratum=label,
                    N=N_h,
                    n=n_h,
                    census=census,
                    se_estimable=se_ok,
                    weight=weight_h,
                    coverage={name_a: cov_a, name_b: cov_b},
                    width={name_a: width_a_native, name_b: width_b_native},
                    mean_coverage_diff=mean_cov_diff,
                    mean_width_diff=mean_width_diff,
                    se_coverage_diff=se_cov,
                    se_width_diff=se_width,
                )
            )

            # HT-style weighted accumulation.
            ht_cov_a += N_h * cov_a
            ht_cov_b += N_h * cov_b
            ht_width_a += N_h * width_a_native
            ht_width_b += N_h * width_b_native
            ht_cov_diff += N_h * mean_cov_diff
            ht_width_diff += N_h * mean_width_diff

            record_details[label] = {
                "X": X_ret,
                "y": y_true,
                "source": np.asarray(src_ret, dtype=object),
                "y_pred_A": np.asarray(yhat_a, dtype=float).reshape(-1),
                "lower_A": np.asarray(lower_a, dtype=float).reshape(-1),
                "upper_A": np.asarray(upper_a, dtype=float).reshape(-1),
                "y_pred_B": np.asarray(yhat_b, dtype=float).reshape(-1),
                "lower_B": np.asarray(lower_b, dtype=float).reshape(-1),
                "upper_B": np.asarray(upper_b, dtype=float).reshape(-1),
                "coverage_A": rec_cov_a,
                "coverage_B": rec_cov_b,
                "coverage_diff": cov_diff,
                "width_A": rec_width_a,
                "width_B": rec_width_b,
                "width_diff": width_diff,
                "sampling_weight": sampling_weights,
            }

        overall_cov = {
            name_a: ht_cov_a / total_N,
            name_b: ht_cov_b / total_N,
        }
        overall_width = {
            name_a: ht_width_a / total_N,
            name_b: ht_width_b / total_N,
        }
        est_cov_diff = ht_cov_diff / total_N
        est_width_diff = ht_width_diff / total_N

        if unestimable:
            warnings.warn(
                "Standard error not estimable for non-census strata with a "
                f"single sampled record: {unestimable}. The overall standard "
                "errors are returned as NaN; collect more records or census these strata.",
                RuntimeWarning,
                stacklevel=2,
            )
            overall_se_cov = float("nan")
            overall_se_width = float("nan")
        else:
            overall_se_cov = float(np.sqrt(var_cov_num) / total_N)
            overall_se_width = float(np.sqrt(var_width_num) / total_N)

        return PairedAuditResult(
            alpha=alpha,
            overall_coverage=overall_cov,
            overall_width=overall_width,
            mean_coverage_diff=est_cov_diff,
            mean_width_diff=est_width_diff,
            se_coverage_diff=overall_se_cov,
            se_width_diff=overall_se_width,
            unestimable_strata=unestimable,
            strata=stratum_results,
            records=record_details,
            model_names={"A": name_a, "B": name_b},
        )

    @staticmethod
    def _intervals(
        model: Any,
        X: np.ndarray,
        alpha: Optional[float],
        predict_kwargs: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Call a fitted interval model and normalize its outputs."""
        X_batch = _stack_records(X)
        try:
            prediction = model.predict(X_batch, alpha=alpha, **predict_kwargs)
        except TypeError:
            # Model whose predict does not take alpha (e.g. pre-baked
            # interval predictor).
            prediction = model.predict(X_batch, **predict_kwargs)
        try:
            y_pred, lower, upper = prediction
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Interval model `predict` must return (y_pred, y_pred_lower, y_pred_upper)."
            ) from exc
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if lower.shape != upper.shape or lower.shape[0] != len(X):
            raise ValueError(
                "Model interval outputs have inconsistent shapes: "
                f"{lower.shape} vs {upper.shape} for {len(X)} records."
            )
        return lower, upper, y_pred
