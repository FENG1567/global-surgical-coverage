"""Compact bold supplementary layouts; original quantitative artists retained.
SPDX-License-Identifier: MIT
"""
from pathlib import Path
import argparse,json
import matplotlib.pyplot as plt
import figure_core_supplement as core
from layout_style import bold_all,label,remove_old_panel_labels,signature,export

def old_legend(fig):
    items=[]
    for leg in list(fig.legends):
        items.extend(zip(leg.legend_handles,[t.get_text() for t in leg.get_texts()]))
        leg.remove()
    return items

def decorate(ax,title):
    ax.set_title(title,loc='left',fontsize=10.0,fontweight='bold',pad=9)
    ax.tick_params(pad=3)
    ax.xaxis.labelpad=6

def information(data,repo,out):
    fig,audit=core.build_s2(data,repo);baseline=signature(fig)
    remove_old_panel_labels(fig);items=old_legend(fig)
    bold_all(fig);fig.set_size_inches(183/25.4,104/25.4)
    for ax,x,l in zip(fig.axes,[.115,.415,.715],'ABC'):
        ax.set_position([x,.265,.265,.585])
        ax.title.set_text(ax.title.get_text().replace('Selected / total = ', 'Selected/total: '))
        ax.title.set_fontsize(9.5)
        ax.xaxis.labelpad=5
        ax.tick_params(pad=2)
        label(fig,l,x-.065,.94)
    fig.axes[0].set_ylabel('Mean remaining coverage width (pp)',fontsize=9,labelpad=7)
    if fig._suptitle:fig._suptitle.set_position((.5,.993));fig._suptitle.set_fontsize(11)
    leg=fig.legend([x[0] for x in items],[x[1] for x in items],loc='lower center',bbox_to_anchor=(.52,.06),
               ncol=2,fontsize=8.5,columnspacing=1.2,handlelength=2.1,handletextpad=.5,labelspacing=.55)
    for h in leg.legend_handles[2:]:h.set_dashes([3,2])
    return export(fig,out,'Figure_S1',104,baseline),audit

def validation_mexico(data,out):
    fig,audit=core.build_s1(data);baseline=signature(fig)
    remove_old_panel_labels(fig);items=old_legend(fig);bold_all(fig)
    fig.set_size_inches(183/25.4,133/25.4)
    a,b=fig.axes;a.set_position([.29,.56,.68,.38]);b.set_position([.29,.14,.68,.28])
    decorate(a,'Brazil: strict cholecystectomy');decorate(b,'Mexico: two services')
    a.set_xlabel('Incremental AUC (activity model - baseline)',fontsize=9.5)
    b.set_xlabel('30-minute population coverage (%)',fontsize=9.5)
    label(fig,'A',.02,.99);label(fig,'B',.02,.47)
    fig.legend([x[0] for x in items],[x[1] for x in items],loc='lower center',bbox_to_anchor=(.51,.005),
               ncol=3,fontsize=8.5,columnspacing=.9,handletextpad=.4)
    return export(fig,out,'Figure_S2',133,baseline),audit

def brazil(data,out):
    fig,audit=core.build_s3(data);baseline=signature(fig)
    remove_old_panel_labels(fig);items=old_legend(fig);bold_all(fig)
    a,b=fig.axes;a.set_position([.21,.28,.35,.59]);b.set_position([.66,.28,.32,.59])
    decorate(a,'Brazil: national joint-\ncomposition counts')
    decorate(b,'Repeated spatial-\nblock validation')
    b.tick_params(axis='y',labelleft=False)
    a.set_xlabel('30-minute population\ncoverage (%)',fontsize=9.5)
    b.set_xlabel('Incremental AUC\n(activity - baseline)',fontsize=9.5)
    label(fig,'A',.02,.99);label(fig,'B',.60,.99)
    fig.legend([x[0] for x in items],[x[1] for x in items],loc='lower center',bbox_to_anchor=(.51,.02),
               ncol=3,fontsize=8.5,columnspacing=.8,handletextpad=.4)
    return export(fig,out,'Figure_S3',92,baseline),audit

def state(data,out):
    fig,audit=core.build_s4(data);baseline=signature(fig)
    remove_old_panel_labels(fig);bold_all(fig)
    a,b=fig.axes;items=[]
    for ax in [a,b]:
        if ax.legend_:
            items.extend(zip(ax.legend_.legend_handles,[t.get_text() for t in ax.legend_.get_texts()]));ax.legend_.remove()
    a.set_position([.23,.29,.30,.59]);b.set_position([.75,.29,.23,.59])
    decorate(a,'Constructed coverage\nbounds');decorate(b,'Three validation\ndesigns')
    a.set_xlabel('30-minute population\ncoverage (%)',fontsize=9.5)
    b.set_xlabel('Incremental AUC\n(activity - baseline)',fontsize=9.5)
    label(fig,'A',.02,.99);label(fig,'B',.59,.99)
    fig.legend([x[0] for x in items],[x[1] for x in items],loc='lower center',bbox_to_anchor=(.51,.005),
               ncol=3,fontsize=8.5,columnspacing=1.0,handletextpad=.4,labelspacing=.65)
    return export(fig,out,'Figure_S4',105,baseline),audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    audits={};values={}
    for name,fn in [('Figure_S1',lambda:information(args.data,args.repo,args.out)),('Figure_S2',lambda:validation_mexico(args.data,args.out)),('Figure_S3',lambda:brazil(args.data,args.out)),('Figure_S4',lambda:state(args.data,args.out))]:
        audits[name],values[name]=fn()
    (args.out/'supplement_layout_values.json').write_text(json.dumps(values,indent=2))
    (args.out/'supplement_layout_summary.json').write_text(json.dumps(audits,indent=2))
if __name__=='__main__':main()


