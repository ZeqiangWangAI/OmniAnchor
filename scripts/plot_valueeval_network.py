"""Plot the frozen human-network comparison, including its weak agreement and stability."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 -- registers the requested scientific styles
import networkx as nx
import numpy as np
import pandas as pd

from omnianchor.campaign import append_event, create_run
from omnianchor.evaluation import compare_networks
from omnianchor.io import read_json
from omnianchor.provenance import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=[args.analysis/p for p in ['human-all-nodes-adjacency.csv','native-raw_logp-all-nodes-adjacency.csv',
        'native-raw_logp/edges.csv','native-raw_logp/summary.json','manifest.json']]
    human,native=[pd.read_csv(p,index_col=0) for p in paths[:2]]
    nodes=human.index.tolist()
    if len(nodes)!=20 or set(native.index)!=set(nodes) or set(human.columns)!=set(nodes) or set(native.columns)!=set(nodes):
        raise ValueError('Require the exact20shared concepts.')
    gold=human.loc[nodes,nodes].to_numpy()
    predicted=native.loc[nodes,nodes].to_numpy()
    point=compare_networks(predicted,gold,top_k=20)
    summary=read_json(paths[3])
    for key in ['edge_weight_spearman','top_k_edge_jaccard']:
        if not np.isclose(point[key],summary['point'][key],atol=1e-12,rtol=0):
            raise ValueError('Figure inputs do not reproduce the frozen comparison.')
    edge_table=pd.read_csv(paths[2])
    edges={frozenset([r.source,r.target]):r for r in edge_table.itertuples()}
    upper=np.triu_indices(20,1)
    if len(edges)!=190:
        raise ValueError('The full edge evidence must remain available.')
    pairs=list(zip(*upper))
    for i,j in pairs:
        if not np.isclose(edges[frozenset([nodes[i],nodes[j]])].weight,predicted[i,j],atol=1e-12,rtol=0):
            raise ValueError('Edge table and adjacency disagree.')
    create_run(args.output,dict(purpose='Final ValueEval network evidence, including weak external alignment',
        source_sha256={str(p):file_hash(p) for p in paths},script_sha256=file_hash(Path(__file__)),
        layout='Identical fixed circular node positions in original coordinate order; no embedding or community fit',
        graph_display='Exactly20strongest nonzero absolute edges pernetwork; all190edges used in comparison',
        opacity='Native edge opacity encodes saved source-bootstrap top20selection frequency; not a posterior probability',
        node_ids=nodes,figure_claim='No human-network recovery is established by these results'))
    plt.style.use(['science', 'no-latex', 'bright'])
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.8,7.5),layout='constrained')
    positions={i:(np.cos(np.pi/2-2*np.pi*i/20),np.sin(np.pi/2-2*np.pi*i/20)) for i in range(20)}
    selected_native=None
    for ax,matrix,title,is_native in [(axes[0,0],gold,'Human label network',False),(axes[0,1],predicted,'Native raw-score network',True)]:
        graph=nx.Graph()
        graph.add_nodes_from(range(20))
        order=[int(k) for k in np.argsort(-np.abs(matrix[upper]),kind='stable') if matrix[upper][k]!=0][:20]
        if len(order)!=20:
            raise ValueError('Insufficient defined nonzero edges for the frozen density.')
        if is_native:
            selected_native=order
        for k in order:
            i,j=pairs[k]
            weight=matrix[i,j]
            frequency=edges[frozenset([nodes[i],nodes[j]])].top20_bootstrap_frequency if is_native else 1
            graph.add_edge(i,j)
            nx.draw_networkx_edges(graph,positions,edgelist=[(i,j)],ax=ax,
                edge_color='#176B87' if weight>=0 else '#B16A36',alpha=.15+.85*frequency,width=.6+1.5*abs(weight))
        nx.draw_networkx_nodes(graph,positions,ax=ax,node_color='#F2F3F4',edgecolors='#647580',node_size=155,linewidths=.7)
        nx.draw_networkx_labels(graph,positions,labels={i:str(i+1) for i in range(20)},ax=ax,font_size=8)
        ax.set_title(title,fontsize=11)
        ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15))
        ax.set_aspect('equal')
        ax.axis('off')
    ax=axes[1,0]
    ax.scatter(gold[upper],predicted[upper],s=13,alpha=.55,color='#176B87',edgecolors='none')
    lo,hi=summary['edge_weight_spearman_ci']
    ax.set(xlabel='Human edge weight (Pearson r)',ylabel='Native edge weight (Pearson r)',title='All 190 shared edges')
    ax.text(.03,.97,f"Edge Spearman = {point['edge_weight_spearman']:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]\nTop 20 Jaccard = {point['top_k_edge_jaccard']:.3f}",
        transform=ax.transAxes,ha='left',va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    ax=axes[1,1]
    records=[edges[frozenset([nodes[pairs[k][0]],nodes[pairs[k][1]]])] for k in selected_native]
    labels=[f'{pairs[k][0]+1}–{pairs[k][1]+1}' for k in selected_native]
    ax.barh(range(20),[r.top20_bootstrap_frequency for r in records],color='#176B87')
    ax.set_yticks(range(20),labels,fontsize=7)
    ax.invert_yaxis()
    ax.set(xlim=(0,1),xlabel='Source-bootstrap selection frequency',title='Stability of displayed native edges')
    fig.suptitle('ValueEval protected network comparison: 1,576 materials, 105 source groups',fontsize=11)
    try:
        pd.DataFrame({'node_number':range(1,21),'concept':nodes}).to_csv(args.output/'node-key.csv',index=False)
        fig.savefig(args.output/'valueeval-network.pdf',bbox_inches='tight')
        fig.savefig(args.output/'valueeval-network.png',dpi=200,bbox_inches='tight')
        append_event(args.output,'completed')
    except BaseException as exc:
        append_event(args.output,'failed',error=repr(exc))
        raise
    finally:
        plt.close(fig)


if __name__=='__main__':
    main()
