"""Static group effect figure for the saved validation replication report."""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np


def plot_groups(report: dict, root: Path) -> None:
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.mpl-cache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(root / '.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'matching': '#2166ac', 'attributes': '#b35806', 'din': '#238b45'}
    markers = {'matching': 'o', 'attributes': 's', 'din': '^'}
    axes_spec = [('item_type', ['new-in-period', 'warm-tail', 'warm-head'], 'Item groups'),
                 ('history_diversity', ['1-2', '3-5', '6-10', '11+ / homogeneous', '11+ / mixed', '11+ / unknown categories'], 'Prior user history')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout='constrained')
    names = list(report['models'])
    for ax, (axis, order, title) in zip(axes, axes_spec):
        groups = report['models'][names[0]]['groups'][axis]
        labels = [x for x in order if x in groups]
        y = np.arange(len(labels))
        for j, name in enumerate(names):
            ci = [report['models'][name]['groups'][axis][g]['delta_vs_r1_user_bootstrap'] for g in labels]
            d = np.array([x['diff'] for x in ci]); lo=np.array([x['ci_low'] for x in ci]); hi=np.array([x['ci_high'] for x in ci])
            yy = y + (j - (len(names)-1)/2)*0.22
            ax.errorbar(d, yy, xerr=[np.maximum(0,d-lo),np.maximum(0,hi-d)], fmt=markers[name],
                        capsize=3, markersize=5, color=colors[name], label=name)
        ax.axvline(0, color='0.45', linewidth=1)
        ax.set_yticks(y, [f"{g}\n{groups[g]['n_instances']:,} cases" for g in labels])
        ax.invert_yaxis(); ax.set_title(title); ax.set_xlabel('Change in macro nDCG@10 vs retriever order')
        ax.grid(axis='x', alpha=0.15); ax.spines[['top','right']].set_visible(False)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,-3))
    seeds = '; '.join(f"{n}: {report['models'][n]['seeds']}" for n in names)
    complete = all(len(report['models'][n]['seeds']) == 3 for n in names)
    fig.suptitle(('Three-seed validation' if complete else 'Provisional validation') + ' — who gains from reranking?\n' + seeds)
    axes[-1].legend(loc='lower left', frameon=False)
    fig.supxlabel('Points: mean across completed ranker seeds. Bars: exploratory paired-user 95% CI; seed uncertainty is separate.', fontsize=9)
    fig.savefig(root / 'group_gains.png', dpi=180)
    fig.savefig(root / 'group_gains.svg')
    plt.close(fig)
