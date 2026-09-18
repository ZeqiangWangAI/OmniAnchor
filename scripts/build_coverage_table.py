"""Build an explicit evidence-backed coverage ledger; no absence-to-success inference."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    studies = [
        ('ValueEval', 'text', 'English', 1576, 'runs/cpu-valueeval-evidence-20260911-01/runs/analysis-45871/analysis/metrics.csv'),
        ('EmoBank', 'text', 'English', 1000, 'runs/cpu-affect-controls-evidence-20260911-01/runs/analysis-45859/analysis/metrics.csv'),
        ('Chinese EmoBank', 'text', 'Chinese', 528, 'runs/cpu-affect-controls-evidence-20260911-01/runs/analysis-45857/analysis/metrics.csv'),
        ('OASIS direct', 'image', 'English anchors', 201, 'runs/oasis-direct-final201-analysis-20260910-02/direct-correlations.csv'),
        ('OASIS predictive', 'image', 'English anchors', 201, 'runs/oasis12-final-metrics-20260911-01/runs/analysis-47435/analysis/metrics.csv'),
        ('VIVA', 'image + action text', 'English', 216, 'runs/final-results-update-20260911-01/runs/analysis-45872/analysis/metrics.csv')]
    hashes = {}
    for study, modality, language, n, name in studies:
        path = root/name; data = pd.read_csv(path)
        if set(data.n) != {n}:
            raise ValueError('Unexpected final sample coverage: '+study)
        for method in data.method.unique():
            rows.append(dict(study=study, modality=modality, language=language, method=method,
                n=n, unit='materials', status='completed final analysis', evidence=name))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    name = 'runs/fmat-matched-validity-45045-01/matched-correlations.csv'
    data = pd.read_csv(root/name)
    for (study, method), values in data[data.subset == 'all50'].groupby(['study', 'method']):
        assert len(values) == 1 and values.n.iloc[0] == 50
        rows.append(dict(study='FMAT '+study, modality='text', language='English', method=method,
            n=50, unit='criterion items', status='completed matched external criterion', evidence=name))
    hashes[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    name = 'runs/final-results-update-20260911-01/runs/analysis-46173/analysis/change-validity.csv'
    data = pd.read_csv(root/name)
    assert set(data.n_targets) == {9}
    for method in data.method.unique():
        rows.append(dict(study='DWUG', modality='context text', language='English', method=method,
            n=9, unit='target words; 1789 usages', status='completed final analysis', evidence=name))
    hashes[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    name = 'runs/video-demo-verification-20260911-01/costs.csv';data = pd.read_csv(root/name)
    assert len(data) == 6 and set(data.videos) == {16} and set(data.texts) == {16}
    for row in data.itertuples():
        rows.append(dict(study=f'Video demo {row.frames} frames', modality='video and text separately', language='English/Chinese caption paths', method=row.method,
            n=32, unit='16 video + 16 text inputs', status='capability only; no validity claim', evidence=name))
    hashes[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    pd.DataFrame(rows).to_csv(output/'coverage.csv', index=False)
    (output/'manifest.json').write_text(json.dumps({'status':'partial_scope_ledger', 'source_sha256':hashes,
        'scope':'Completed primary outcomes including OASIS predictive evaluation and video capabilities. E3 bounded sensitivity is separate; not every software smoke test is represented.',
        'missing_is_not_failure':'Unlisted combinations were not established by this ledger; no implied zero performance.'},indent=2))


if __name__ == '__main__':
    main()
