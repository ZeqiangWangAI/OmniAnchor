from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from vlanchor.calibration import fit_reference, to_matrix, transform
from vlanchor.errors import IncompatibleMeasurement, MissingScores
from vlanchor.types import ScoreTable


def make_scores(values, *, split="train", prefix="s"):
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[:, :, None]
    ns, na, nb = values.shape
    anchors = [{"id": f"a{j}", "surface": f"anchor {j}", "language": "en"} for j in range(na)]
    bridges = [{"id": f"b{k}", "prefix": f"bridge {k}: ", "relation": "associated_with",
                "language": "en"} for k in range(nb)]
    samples = [{"id": f"{prefix}{i}", "metadata": {"split": split}} for i in range(ns)]
    rows = [{"sample_id": samples[i]["id"], "anchor_id": anchors[j]["id"],
             "anchor_surface": anchors[j]["surface"], "bridge_id": bridges[k]["id"],
             "bridge_prefix": bridges[k]["prefix"], "relation": "associated_with",
             "event": "token_prefix", "raw_logp": values[i, j, k], "token_count": j + 1,
             "mean_token_logp": values[i, j, k] / (j + 1), "status": "ok",
             "model_id": "toy", "model_revision": "v1", "measurement_id": "m"}
            for i in range(ns) for j in range(na) for k in range(nb)]
    return ScoreTable(pd.DataFrame(rows), {"instrument": {"model": "toy", "version": "v1"},
                                         "anchors": anchors, "bridges": bridges, "samples": samples,
                                         "event": "token_prefix", "measurement_id": "m"})


def test_mixture_geometric_reference_and_sample_sd():
    reference = fit_reference(make_scores(np.log([[0.1], [0.9]])))
    row = reference.statistics.iloc[0]
    assert row.log_mean_probability == pytest.approx(np.log(0.5))
    assert row.mean_logp == pytest.approx(np.log(0.3))
    assert row.std_logp == pytest.approx(np.std(np.log([0.1, 0.9]), ddof=1))
    result = transform(make_scores(np.log([[0.4]]), split="test", prefix="test"), reference)
    assert result.frame.reference_log_ratio.iloc[0] < 0
    assert result.frame.raw_logp.iloc[0] - row.mean_logp > 0


def test_logmeanexp_is_stable_and_raw_is_preserved():
    table = make_scores([[-10000], [-10001]])
    original = table.frame.copy(deep=True)
    result = transform(table, fit_reference(table), variant="reference_log_ratio")
    assert np.isfinite(result.frame.reference_log_ratio).all()
    assert "reference_z" not in result.frame
    pd.testing.assert_frame_equal(table.frame, original)


@pytest.mark.parametrize("split", ["test", "validation", "dev"])
def test_reference_rejects_known_evaluation_split_even_with_override(split):
    with pytest.raises(IncompatibleMeasurement):
        fit_reference(make_scores([[-1], [-2]], split=split), split="train")


@pytest.mark.parametrize("mutation", ["instrument", "surface", "bridge", "event", "revision"])
def test_strict_reference_provenance(mutation):
    table = make_scores([[-1], [-2]])
    reference = fit_reference(table)
    changed = deepcopy(table)
    if mutation == "instrument":
        changed.manifest["instrument"]["version"] = "v2"
    elif mutation == "surface":
        changed.frame["anchor_surface"] = "changed"
        changed.manifest["anchors"][0]["surface"] = "changed"
    elif mutation == "bridge":
        changed.frame["bridge_prefix"] = "changed"
        changed.manifest["bridges"][0]["prefix"] = "changed"
    elif mutation == "event":
        changed.frame["event"] = "turn_terminated"
        changed.manifest["event"] = "turn_terminated"
    else:
        changed.frame["model_revision"] = "v2"
    with pytest.raises(IncompatibleMeasurement):
        transform(changed, reference)


def test_reference_z_affine_and_token_length_identities():
    values = np.array([[-2., -10.], [-3., -12.], [-6., -16.], [-8., -20.]])
    table = make_scores(values)
    baseline = to_matrix(transform(table, fit_reference(table)), variant="reference_z").values
    for modified in [values - np.array([2, 4]), values / np.array([1, 2]), values * [2, 4] - [5, 3]]:
        other = make_scores(modified)
        actual = to_matrix(transform(other, fit_reference(other)), variant="reference_z").values
        np.testing.assert_allclose(actual, baseline)


def test_per_anchor_ranking_invariance_does_not_imply_within_sample_invariance():
    values = np.array([[-1, -2], [-1.5, -2.2], [-3, -2], [-4, -1.4]])
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    shifted = values - np.array([-6, -1])
    for j in range(2):
        assert average_precision_score(labels[:, j], values[:, j]) == pytest.approx(
            average_precision_score(labels[:, j], shifted[:, j]))
        assert roc_auc_score(labels[:, j], values[:, j]) == pytest.approx(
            roc_auc_score(labels[:, j], shifted[:, j]))
        assert spearmanr(values[:, j], shifted[:, j]).statistic == pytest.approx(1)
    before = np.mean([average_precision_score(y, s) for y, s in zip(labels, values)])
    after = np.mean([average_precision_score(y, s) for y, s in zip(labels, shifted)])
    assert before == 1
    assert after < before


def test_fixed_bridges_and_explicit_missing_policies():
    table = make_scores(np.arange(18).reshape(3, 2, 3) * -0.1)
    ragged = ScoreTable(table.frame.drop(index=0), table.manifest)
    with pytest.raises(MissingScores):
        to_matrix(ragged)
    samples = to_matrix(ragged, missing="drop_samples")
    assert samples.sample_ids == ("s1", "s2")
    assert samples.manifest["coverage"]["dropped_samples"] == ["s0"]
    anchors = to_matrix(ragged, missing="drop_anchors")
    assert anchors.anchor_ids == ("a1",)
    np.testing.assert_allclose(anchors.values, table.frame.query("anchor_id == 'a1'")
                               .groupby("sample_id").raw_logp.mean().to_numpy()[:, None])
    with pytest.raises(MissingScores):
        fit_reference(ragged)


def test_constant_insufficient_failed_and_duplicate_observations():
    table = make_scores([[-2, -1], [-2, -3]])
    transformed = transform(table, fit_reference(table))
    assert transformed.frame.query("anchor_id == 'a0'").reference_z.isna().all()
    with pytest.raises(MissingScores):
        to_matrix(transformed, variant="reference_z")
    assert to_matrix(transformed, variant="reference_z", missing="drop_anchors").anchor_ids == ("a1",)
    one = make_scores([[-2]])
    assert fit_reference(one).statistics.z_status.iloc[0] == "undefined_insufficient"
    failed = deepcopy(table)
    failed.frame.loc[0, "status"] = "error"
    with pytest.raises(MissingScores):
        to_matrix(failed)
    duplicate = ScoreTable(pd.concat([table.frame, table.frame.iloc[[0]]]), table.manifest)
    with pytest.raises(MissingScores):
        to_matrix(duplicate)


def test_mixed_relations_and_undeclared_coordinates_fail():
    table = make_scores(np.ones((2, 1, 2)) * -1)
    table.frame.loc[0, "relation"] = "endorses"
    with pytest.raises(IncompatibleMeasurement):
        to_matrix(table)


def test_split_mapping_and_corrupt_artifact_are_not_ignored():
    table = make_scores([[-1], [-2]])
    table.manifest["splits"] = {"s0": "test", "s1": "train"}
    with pytest.raises(IncompatibleMeasurement):
        fit_reference(table)
    table = make_scores([[-1], [-2]])
    reference = fit_reference(table)
    reference.statistics.loc[0, "std_logp"] = -1
    with pytest.raises(IncompatibleMeasurement):
        transform(table, reference)


def test_imported_bridge_language_and_instrument_model_consistency():
    table = make_scores(np.ones((2, 1, 2)) * -1)
    table.manifest["bridges"][0]["language"] = "zh"
    with pytest.raises(IncompatibleMeasurement):
        to_matrix(table)
    table = make_scores([[-1], [-2]])
    table.manifest["model"] = {"id": "some-other-model"}
    with pytest.raises(IncompatibleMeasurement):
        to_matrix(table)
    table = make_scores([[-1], [-2]])
    table.frame.loc[0, "sample_id"] = "unknown"
    with pytest.raises(IncompatibleMeasurement):
        to_matrix(table)
