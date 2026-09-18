"""Evaluate both VATEX languages under one frozen paired-comparison family."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .analyze_vatex import align_scores, retrieval_rows
else:
    from analyze_vatex import align_scores, retrieval_rows
from omnianchor.analysis import bh_fdr
from omnianchor.campaign import append_event, create_run
from omnianchor.evaluation import group_bootstrap
from omnianchor.io import load_samples, read_json
from omnianchor.provenance import file_hash


def paired_interval(values, groups):
    """The shared group-bootstrap CI and its centered two-sided tail probability."""
    values = np.asarray(values, dtype=float)
    draws = []

    def statistic(index):
        value = float(values[index].mean())
        draws.append(value)
        return value

    result = group_bootstrap(statistic, groups)
    # The first callback is the original population, followed by bootstrap draws.
    boot = np.asarray(draws[1:])
    result['centered_bootstrap_p'] = float((1 + (np.abs(boot-result['estimate']) >= abs(result['estimate'])).sum()) / (1+len(boot)))
    return result


def validate_family(family, methods):
    keys = [(r['language'], r['method_a'], r['method_b'], r['direction'], r['metric']) for r in family]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Frozen comparison family is empty or duplicated.')
    for language, a, b, direction, metric in keys:
        if language not in {'en', 'zh'} or a == b or not {a, b} <= set(methods):
            raise ValueError('Comparison family has unknown language or methods.')
        if direction != 'caption_to_video' or metric not in {'recall@1', 'recall@5', 'recall@10'}:
            raise ValueError('Primary family is caption-to-video recall at 1/5/10.')
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.frozen_evaluation)
    if (contract['status'] != 'frozen' or contract['analysis_sha256'] != file_hash(Path(__file__))
            or contract['retrieval_helper_sha256'] != file_hash(Path(__file__).with_name('analyze_vatex.py'))):
        raise ValueError('Final analysis changed after freeze.')
    for name, digest in contract['source_sha256'].items():
        if file_hash(Path(name)) != digest:
            raise ValueError('Frozen analysis dependency changed: '+name)
    plan = read_json(Path(contract['analysis_plan']))
    if file_hash(Path(contract['analysis_plan'])) != contract['analysis_plan_sha256'] or plan['status'] != 'frozen':
        raise ValueError('Earlier analysis plan changed.')
    for name in ['primary_method', 'methods', 'caption_only_methods', 'primary_family']:
        if contract[name] != plan[name]:
            raise ValueError('Final admission differs from earlier analysis plan: '+name)
    required_sources = {'src/omnianchor/evaluation.py', 'src/omnianchor/analysis.py'}
    if not required_sources <= set(contract['source_sha256']):
        raise ValueError('Missing frozen statistical dependencies.')
    methods = contract['methods']
    if len(set(methods)) != len(methods) or contract['primary_method'] != 'native-raw_logp':
        raise ValueError('Require fixed native raw geometry and unique methods.')
    if not set(contract['caption_only_methods']) <= set(methods):
        raise ValueError('Unknown caption-only method.')
    family = validate_family(contract['primary_family'], methods)
    if set(contract['languages']) != {'en', 'zh'}:
        raise ValueError('Both languages are required for joint multiplicity control.')
    for name, digest in contract['input_sha256'].items():
        if file_hash(Path(name)) != digest:
            raise ValueError('Frozen evaluation input changed: '+name)
    create_run(args.output, dict(purpose='Frozen VATEX final evaluation, both languages and joint comparison family',
        test_used=True, final_contract_sha256=file_hash(args.frozen_evaluation),
        input_sha256=contract['input_sha256'], primary_family=contract['primary_family'],
        uncertainty='1000 YouTube source-group draws; fixed gallery; 95% percentile intervals; seed42',
        tie_policy='stable lexicographic gallery sample ID', excluded_queries=[],
        limitations='Published pairs may omit valid alternative matches; source bootstrap is conditional on this available gallery. Secondary directions/geometries are descriptive.'))
    try:
        results, per_query, contributions, groups_by_direction = [], [], {}, {}
        shared_video_ids = None
        for language, inputs in contract['languages'].items():
            paths = [inputs['videos'], inputs['captions'], *inputs['scores'].values()]
            if not set(paths) <= set(contract['input_sha256']) or set(inputs['scores']) != set(methods):
                raise ValueError('Unhashed inputs or missing methods in final evaluation.')
            videos = sorted(load_samples(Path(inputs['videos'])), key=lambda s: s.id)
            captions = sorted(load_samples(Path(inputs['captions'])), key=lambda s: s.id)
            vi, ci = [s.id for s in videos], [s.id for s in captions]
            if shared_video_ids is not None and vi != shared_video_ids:
                raise ValueError('Languages must share the same gallery.')
            shared_video_ids = vi
            if len(videos) != 407 or len(captions) != 4070 or len(set(vi+ci)) != len(vi+ci):
                raise ValueError('Require all 407 protected videos and 4070 captions.')
            if any(s.metadata['split'] != 'test' for s in videos+captions):
                raise ValueError('Final population contains non-test material.')
            if any(len(s.parts) != 1 or s.parts[0].type != 'video' for s in videos):
                raise ValueError('Gallery videos must not include captions.')
            if any(len(s.parts) != 1 or s.parts[0].type != 'text' or s.language != language for s in captions):
                raise ValueError('Caption role or language mismatch.')
            sources = {s.id: s.group_id for s in videos}
            if any(not g for g in sources.values()) or any(s.group_id != sources.get(s.pair_id) for s in captions):
                raise ValueError('Require aligned YouTube source groups.')
            relevance = np.equal.outer([s.pair_id for s in captions], vi)
            if not np.all(relevance.sum(axis=0) == 10) or not np.all(relevance.sum(axis=1) == 1):
                raise ValueError('Every video needs all ten paired captions.')
            for method in methods:
                scores = align_scores(Path(inputs['scores'][method]), ci, vi)
                directions = [('caption_to_video', relevance, scores, captions)]
                if method not in contract['caption_only_methods']:
                    directions.append(('video_to_caption', relevance.T, scores.T, videos))
                for direction, gold, values, samples in directions:
                    groups = [s.group_id for s in samples]
                    groups_by_direction[language, direction] = groups
                    for metric, rows in retrieval_rows(gold, values).items():
                        contributions[language, method, direction, metric] = rows
                        interval = group_bootstrap(lambda index: rows[index].mean(), groups)
                        results.append(dict(language=language, method=method, direction=direction, metric=metric, **interval))
                        per_query.extend(dict(language=language, method=method, direction=direction, metric=metric,
                            sample_id=s.id, source_group=s.group_id, value=float(v)) for s, v in zip(samples, rows))
        paired = []
        for language, a, b, direction, metric in family:
            delta = contributions[language, a, direction, metric] - contributions[language, b, direction, metric]
            paired.append(dict(language=language, method_a=a, method_b=b, direction=direction,
                metric=metric, **paired_interval(delta, groups_by_direction[language, direction])))
        paired = pd.DataFrame(paired)
        paired['primary_family_bh_q'] = bh_fdr(paired.centered_bootstrap_p.to_numpy())
        pd.DataFrame(results).to_csv(args.output/'metrics.csv', index=False)
        pd.DataFrame(per_query).to_csv(args.output/'per-query.csv', index=False)
        paired.to_csv(args.output/'paired-differences.csv', index=False)
        append_event(args.output, 'completed')
    except BaseException as exc:
        append_event(args.output, 'failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
