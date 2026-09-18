"""Verify and plot the prespecified32-material anchor-count cost sweep."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 -- registers the requested scientific styles
import numpy as np
import pandas as pd

from omnianchor.campaign import create_run, append_event
from omnianchor.io import read_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = read_json(args.runs)
    if set(runs) != {'16', '64', '128', '256'}:
        raise ValueError('Require all four predefined anchor counts.')
    rows, identities, hashes = [], [], {}
    all_anchors = None
    for count in [256, 128, 64, 16]:
        folder = Path(runs[str(count)])/'measurement'
        m, cost, hardware = [read_json(folder/name) for name in ['manifest.json', 'cost.json', 'hardware.json']]
        events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        completed = [e for e in events if e['status'] == 'sample_completed']
        ids = [e['sample_id'] for e in completed]
        if ((folder.parent/'exit_code.txt').read_text().strip() != '0' or events[-1]['status'] != 'completed'
                or len(ids) != 32 or len(set(ids)) != 32 or cost['samples'] != 32 or cost['anchors'] != count
                or cost['bridges'] != 3 or cost['score_items'] != 32*count*3 or m['shared_prefill']
                or any(e['execution']['failed_items'] or e['execution']['score_items'] != count*3 for e in completed)
                or read_json(folder/'native-verification.json')['status'] != 'passed'):
            raise ValueError('Incomplete or different cost experiment.')
        if '5000 Ada' not in hardware['gpu']:
            raise ValueError('Cost sweep requires matched Ada hardware.')
        spec = m['spec'].copy()
        anchors = spec.pop('anchors')
        spec.pop('name')
        if all_anchors is None:
            all_anchors = anchors
        if anchors != all_anchors[:count]:
            raise ValueError('Require the frozen nested anchor sequence.')
        identities.append(dict(spec=spec, ids=ids, hardware=hardware, samples=m['samples_sha256'],
            batch=m['batch_size'], core={k:m['source_hashes'][k] for k in
                ['src/omnianchor/engine.py','src/omnianchor/backends/hf.py','src/omnianchor/backends/media.py']}))
        if identities[-1] != identities[0]:
            raise ValueError('Cost runs changed inputs, core scoring, hardware or non-anchor configuration.')
        hashes.update({str(p):file_hash(p) for p in [folder/n for n in
            ['manifest.json','cost.json','hardware.json','events.jsonl','native-verification.json']]})
        parts = sorted(folder.glob('part-*.parquet'))
        if len(parts) != 32:
            raise ValueError('Raw score coverage is incomplete.')
        scores = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if (len(scores) != 32*count*3 or set(scores.sample_id) != set(ids)
                or not scores.status.eq('ok').all() or not np.isfinite(scores.raw_logp).all()
                or scores.duplicated(['sample_id','anchor_id','bridge_id']).any()
                or not scores.groupby('sample_id').size().eq(count*3).all()):
            raise ValueError('Raw scores contradict successful measurement coverage.')
        hashes.update({str(p):file_hash(p) for p in parts})
        rows.append(dict(anchors=count, **{k:v for k,v in cost.items() if k!='anchors'},
            seconds_per_material=cost['measurement_seconds']/32,
            seconds_per_score=cost['measurement_seconds']/(32*count*3),
            mean_event_tokens=float(scores.token_count.mean()),
            min_event_tokens=int(scores.token_count.min()), max_event_tokens=int(scores.token_count.max()),
            gpu=hardware['gpu']))
    table = pd.DataFrame(rows).sort_values('anchors')
    create_run(args.output, dict(purpose='Observed N16/64/128/256 efficiency, no human-label analysis',
        inputs_sha256=hashes, script_sha256=file_hash(Path(__file__)),
        limitation='One run per N on32matched materials; no timing confidence interval. Nested anchor identities/token lengths change with N; not an isolated causal anchor-count effect. Measurement wall time differs from Slurm allocation time.'))
    try:
        table.to_csv(args.output/'costs.csv', index=False)
        plt.style.use(['science', 'no-latex', 'bright'])
        plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42,
            'axes.spines.top':False,'axes.spines.right':False})
        fig, axes = plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
        for key,label,style in [('measurement_seconds','Measurement wall time','-o'),('forward_seconds','Timed forwards','--s')]:
            axes[0].plot(table.anchors,table[key]/32,style,label=label,color='#176B87' if style=='-o' else '#B16A36')
        axes[0].set(xlabel='Anchors',ylabel='Seconds per material',xticks=[16,64,128,256])
        axes[0].legend(frameon=False,fontsize=8)
        axes[1].plot(table.anchors,table.peak_allocated_bytes/2**30,'-o',color='#176B87',label='Allocated')
        axes[1].plot(table.anchors,table.peak_reserved_bytes/2**30,'--s',color='#B16A36',label='Reserved')
        axes[1].set(xlabel='Anchors',ylabel='Peak GPU memory (GiB)',xticks=[16,64,128,256],ylim=(0,10))
        axes[1].legend(frameon=False,fontsize=8)
        fig.suptitle('32 fixed DWUG materials · three bridges · Qwen3.5-4B · RTX 5000 Ada',fontsize=10)
        for extension in ['pdf','png']:
            fig.savefig(args.output/f'anchor-efficiency.{extension}',dpi=180)
        plt.close(fig)
        if not np.isfinite(table.seconds_per_material).all():
            raise ValueError('Invalid measured time.')
        append_event(args.output,'completed')
    except BaseException as exc:
        append_event(args.output,'failed',error=repr(exc))
        raise


if __name__ == '__main__':
    main()
