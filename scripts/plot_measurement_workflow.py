"""Render the declared measurement and validation workflow; no result data are invented."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 -- registers the requested scientific styles
from matplotlib.patches import FancyBboxPatch

from omnianchor.campaign import create_run, append_event
from omnianchor.provenance import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    create_run(args.output,dict(purpose='Method schematic, not empirical evidence',
        source_sha256={p:file_hash(root/p) for p in ['SPEC.md','DERIVATION_PACKAGE.md','docs/HANDOFF_SPEC.md']},
        script_sha256=file_hash(Path(__file__)),matplotlib=matplotlib.__version__))
    plt.style.use(['science', 'no-latex', 'bright'])
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(10,4.8))
    ax.set(xlim=(0,10),ylim=(1.2,5.8))
    ax.axis('off')

    def box(x,y,w,h,title,body,color='#EDF5F7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.08',
            facecolor=color,edgecolor='#637883',linewidth=1))
        ax.text(x+w/2,y+h-.24,title,ha='center',va='top',fontweight='bold',fontsize=12)
        ax.text(x+w/2,y+h-.64,body,ha='center',va='top',fontsize=12,linespacing=1.5)

    def arrow(start,end,**kwargs):
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',color='#50666F',lw=1.3,**kwargs))

    ax.text(.05,5.62,'OmniAnchor measurement and validation',fontsize=13,fontweight='bold',va='top')
    top=[('Material','Text / image / video\nContent hashes\nProcessing records'),
         ('Fixed specification','Model + anchors\nExact bridge relation'),
         ('Event scores','Token log likelihood\nTeacher forcing\nUncached decoder'),
         ('Coordinates','Mean across bridges\nRaw / log-ratio / z\nTrain-only reference')]
    for index,(title,body) in enumerate(top):
        x=.05+index*2.52
        box(x,3.7,2.3,1.6,title,body)
        if index<3:
            arrow((x+2.3,4.52),(x+2.49,4.52))
    box(.05,1.46,4.2,1.65,'Protected evaluation protocol',
        'Train: search, references, predictor\nDev: fixed hyperparameter grid\nTest: one frozen assessment',color='#F2F3F4')
    box(4.7,1.46,2.43,1.65,'Direct validity',
        'Declared scores\nHuman criterion\nNo learned label map')
    box(7.56,1.46,2.39,1.65,'Predictive validity',
        'Frozen predictor\nHuman criterion\nFeatures + model')
    arrow((8.7,3.7),(5.92,3.14),connectionstyle='arc3,rad=0')
    arrow((8.83,3.7),(8.83,3.14))
    try:
        fig.savefig(args.output/'measurement-workflow.pdf',bbox_inches='tight')
        fig.savefig(args.output/'measurement-workflow.png',dpi=200,bbox_inches='tight')
        append_event(args.output,'completed')
    except BaseException as exc:
        append_event(args.output,'failed',error=repr(exc))
        raise
    finally:
        plt.close(fig)


if __name__=='__main__':
    main()
