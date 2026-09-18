import json

import numpy as np
import pytest

from scripts.merge_vatex_reranking import merge_shards
from omnianchor.provenance import file_hash


def test_full_top100_merge_rejects_incomplete_duplicate_and_changed_support(tmp_path):
    # 101 candidates exercises the genuinely unscored tail beyond top100.
    videos = [f'v{i:03d}' for i in range(101)]
    queries = ['q0', 'q1']
    scores = np.stack([np.linspace(.9, -.9, 101), np.linspace(-.9, .9, 101)])
    path = tmp_path/'embedding.npz'
    np.savez(path, scores=scores, query_ids=queries, candidate_ids=videos)
    selected = np.ones_like(scores, dtype=bool)
    selected[0, -1] = selected[1, 0] = False
    probabilities = np.where(selected, .25, np.nan)
    runs = []
    for number, columns in enumerate([list(range(50)), list(range(50, 101))]):
        run = tmp_path/str(number)
        run.mkdir()
        runs.append(run)
        ids = [videos[i] for i in columns]
        manifest = dict(pilot=False, stage_one_sha256=file_hash(path), query_ids=queries,
            video_ids=videos, owned_video_ids=ids, model_id='pinned', revision='fixed', instruction='same',
            resources={}, source_sha256={}, vision_reuse={}, media_preprocessing_reuse=False,
            resource_spec_sha256='same', samples_sha256='same')
        (run/'manifest.json').write_text(json.dumps(manifest))
        (run/'events.jsonl').write_text('{"status":"completed"}\n')
        (run/'cost.json').write_text('{"gpu":"NVIDIA RTX 5000 Ada Generation"}')
        np.savez(run/'shard.npz', probabilities=probabilities[:, columns], selected=selected[:, columns],
            query_ids=queries, candidate_ids=ids)
    result = merge_shards(path, runs[::-1])
    np.testing.assert_equal(result['probabilities'], probabilities)
    np.testing.assert_equal(result['scores'], np.where(selected, 2.25, scores))
    with pytest.raises(ValueError, match='Incomplete gallery'):
        merge_shards(path, runs[:1])
    with pytest.raises(ValueError, match='Duplicate'):
        merge_shards(path, runs+runs[:1])
    damaged = runs[0]/'shard.npz'
    with np.load(damaged) as data:
        contents = dict(data)
    contents['selected'][1, 0] = True
    np.savez(damaged, **contents)
    with pytest.raises(ValueError, match='support differs'):
        merge_shards(path, runs)
