"""Assemble complete reranking support without replacing missing scores or reading gold."""
import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_vatex import two_stage_priority
else:
    from analyze_vatex import two_stage_priority
from omnianchor.campaign import create_run, append_event
from omnianchor.io import read_json
from omnianchor.provenance import file_hash


def merge_shards(embedding_path, runs):
    with np.load(embedding_path, allow_pickle=False) as data:
        queries, videos = data['query_ids'].tolist(), data['candidate_ids'].tolist()
        embedding = data['scores'].astype(float)
    if (queries != sorted(set(queries)) or videos != sorted(set(videos))
            or embedding.shape != (len(queries), len(videos)) or not queries or not videos):
        raise ValueError('Require the complete lexicographically ordered embedding matrix.')
    selected = np.zeros_like(embedding, dtype=bool)
    np.put_along_axis(selected, np.argsort(-embedding, axis=1, kind='stable')[:, :min(100, len(videos))], True, axis=1)
    probabilities = np.full_like(embedding, np.nan)
    seen, identities = set(), []
    stage_hash = file_hash(embedding_path)
    for run in runs:
        run = Path(run)
        manifest = read_json(run/'manifest.json')
        events = (run/'events.jsonl').read_text().splitlines()
        if not events or json.loads(events[-1])['status'] != 'completed':
            raise ValueError('Shard is incomplete or failed.')
        if (manifest['pilot'] or manifest['stage_one_sha256'] != stage_hash
                or manifest['query_ids'] != queries or manifest['video_ids'] != videos):
            raise ValueError('Shard population or stage-one matrix differs.')
        identities.append({k: manifest[k] for k in ['model_id', 'revision', 'instruction', 'resources',
            'source_sha256', 'vision_reuse', 'media_preprocessing_reuse', 'resource_spec_sha256', 'samples_sha256']})
        if '5000 Ada' not in read_json(run/'cost.json')['gpu']:
            raise ValueError('Require matched frozen Ada hardware.')
        if identities[-1] != identities[0]:
            raise ValueError('Shard scoring identities differ.')
        with np.load(run/'shard.npz', allow_pickle=False) as data:
            ids = data['candidate_ids'].tolist()
            if (not ids or len(set(ids)) != len(ids) or not set(ids) <= set(videos)
                    or seen & set(ids) or ids != manifest['owned_video_ids']
                    or data['query_ids'].tolist() != queries):
                raise ValueError('Duplicate, unknown or misaligned shard IDs.')
            columns = [videos.index(i) for i in ids]
            values = data['probabilities']
            if values.shape != (len(queries), len(ids)) or not np.array_equal(data['selected'], selected[:, columns]):
                raise ValueError('Candidate support differs from frozen top100.')
            probabilities[:, columns] = values
            seen.update(ids)
    if seen != set(videos):
        raise ValueError('Incomplete gallery coverage; no missing-score substitution.')
    scores = two_stage_priority(embedding, probabilities, selected)
    return dict(scores=scores, probabilities=probabilities, selected=selected,
                embedding_scores=embedding, query_ids=np.array(queries), candidate_ids=np.array(videos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding-scores', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.embedding_scores, Path(__file__)]
    paths += [r/n for r in args.runs for n in ['manifest.json', 'events.jsonl', 'shard.npz', 'cost.json']]
    create_run(args.output, dict(purpose='Full top100 reranking assembly, no relevance-label access',
        inputs_sha256={str(p): file_hash(p) for p in paths}))
    try:
        result = merge_shards(args.embedding_scores, args.runs)
        np.savez_compressed(args.output/'qwen-direct-reranker.npz', **result)
        append_event(args.output, 'completed')
    except BaseException as exc:
        append_event(args.output, 'failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
