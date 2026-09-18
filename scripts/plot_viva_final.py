"""Plot frozen VIVA final utility and paired contrasts with SciencePlots."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import numpy as np
import pandas as pd

from omnianchor.campaign import create_run, append_event
from omnianchor.provenance import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=[args.analysis/n for n in ['metrics.csv','paired-differences.csv','per-recipient.csv']]
    metrics,paired,per=[pd.read_csv(p) for p in paths]
    recalculated=per.groupby(['method','condition','metric']).value.mean()
    if len(metrics)!=18 or not metrics.n.eq(216).all():
        raise ValueError('Require all216recipients,3methods,3conditions and2metrics.')
    for r in metrics.itertuples():
        if not np.isclose(recalculated.loc[r.method,r.condition,r.metric],r.value,atol=1e-12,rtol=0):
            raise ValueError('Per-recipient values disagree with reported metrics.')
    create_run(args.output,dict(purpose='VIVA frozen final216 utility, no tuning',
        input_sha256={str(p):file_hash(p) for p in paths},script_sha256=file_hash(Path(__file__)),
        style='SciencePlots2.2.2:science/no-latex/bright;STIXGeneral',
        inference='Saved1000recipient-image-group bootstrap95%intervals;4primaryAP contrasts and BHq'))
    plt.style.use(['science','no-latex','bright'])
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix',
        'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.top':False,'ytick.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9,3.9),layout='constrained',gridspec_kw={'width_ratios':[1,1.15]})
    conditions=['action_only','image_action','mismatched_image_action']
    colors=['#176B87','#929DA6','#B16A36']
    for index,(method,label) in enumerate([('native','OmniAnchor'),('qwen-embedding','Embedding'),('qwen-reranker','Reranker')]):
        d=metrics[(metrics.method==method)&(metrics.metric=='AP')].set_index('condition').loc[conditions]
        x=np.arange(3)+(index-1)*.12
        axes[0].errorbar(x,d.value,yerr=[d.value-d.ci_lower,d.ci_upper-d.value],fmt=['o','s','^'][index],
            color=colors[index],capsize=3,ms=5,label=label,linestyle='-',linewidth=1)
    axes[0].set(xticks=range(3),xticklabels=['Text only','Image + text','Wrong image\n+ text'],ylabel='Mean average precision',ylim=(.64,.91),title='A  Value ranking on 216 images')
    axes[0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.10),ncol=3,fontsize=8)
    axes[0].set_title('A  Value ranking on 216 images',pad=34)
    contrasts=paired[paired.metric=='AP'].reset_index(drop=True)
    labels=['Image − text only','Correct − wrong image','OmniAnchor − embedding','OmniAnchor − reranker']
    for i,r in enumerate(contrasts.itertuples()):
        axes[1].errorbar(r.difference,i,xerr=[[r.difference-r.ci_lower],[r.ci_upper-r.difference]],fmt='o',color='#176B87',capsize=3,ms=5)
        axes[1].text(.235,i,f'q={r.primary_family_bh_q:.3f}',va='center',ha='right',fontsize=9)
    axes[1].axvline(0,color='#888888',ls='--',lw=.8)
    axes[1].set(yticks=range(4),yticklabels=labels,xlabel='Paired difference in average precision',xlim=(-.055,.245),title='B  Prespecified paired contrasts')
    axes[1].invert_yaxis()
    axes[1].set_title('B  Prespecified paired contrasts',pad=34)
    for extension in ['pdf','png']:
        fig.savefig(args.output/f'viva-final.{extension}',dpi=200)
    plt.close(fig)
    append_event(args.output,'completed')


if __name__=='__main__':
    main()
