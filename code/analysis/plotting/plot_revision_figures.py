"""Revision figures from aggregate source tables, Python-only rendering.

Figure contract: quantitative grids; show conditional coverage uncertainty,
its dependence on budgets/composition and the empirical consequences of
missing labels. All plotted ranges are constructed bounds unless certified.
Existing Figures 1 and 3 retain their original data and vector exports.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent/'submission_editor_20261113'
UPGRADE = ROOT.parent/'global_surg_access_upgrade_20261112'
OUT = ROOT/'figures'
NAVY, TEAL, GOLD, GRAY = '#355C7D', '#547A73', '#B07D3C', '#78818A'
NAMES = {'chole':'Cholecystectomy','cesarean':'Cesarean delivery','appendectomy':'Appendectomy'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7.5,'axes.titlesize':8,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,
    'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    for extension in ['pdf','svg','png']:
        fig.savefig(OUT/f'{stem}.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def panel(ax, letter, title):
    ax.set_title(title,loc='left',pad=10)
    ax.text(-.12,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)

def figure2():
    old = pd.read_csv(PREVIOUS/'figures/main/Figure_2_source_data.csv')
    old = old.loc[old.scenario_label.notna()].copy()
    assert len(old)==4
    pairs = pd.read_csv(ROOT/'global/global_budget_paired_spreads.csv')
    exact = pd.read_csv(ROOT/'global/country_exact_composition_spreads.csv')
    fig, axes = plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained',gridspec_kw={'wspace':.13,'hspace':.2})
    ax=axes[0,0]
    for y,row in enumerate(old.itertuples()):
        color=TEAL if row.scenario in ['rand','cluster'] else GRAY
        ax.scatter(row.coverage_30_percent,y,color=color,s=33)
        ax.text(row.coverage_30_percent+.25,y,f'{row.coverage_30_percent:.2f}%',va='center',fontsize=7)
    ax.set_yticks(range(4),old.scenario_label)
    ax.invert_yaxis();ax.set_xlim(76,88);ax.set_xlabel('30-minute coverage (%)')
    panel(ax,'a','Original central-budget assignments')
    ax.text(.98,.14,'Random–cluster spread: 5.58 pp',transform=ax.transAxes,ha='right',fontsize=6.8,color=TEAL)
    for ax,threshold,letter in [(axes[0,1],30,'b'),(axes[1,0],120,'c')]:
        data=pairs.loc[pairs.population_group.eq('all') & pairs.threshold_minutes.eq(threshold)].set_index('budget').loc[['lower','central','upper']]
        for y,(budget,row) in enumerate(data.iterrows()):
            ax.plot([row.cluster_coverage_percent,row.random_coverage_percent],[y,y],color=TEAL,lw=2)
            ax.scatter([row.cluster_coverage_percent,row.random_coverage_percent],[y,y],color=[GOLD,NAVY],s=24,zorder=3)
            ax.annotate(f'{row.constructed_spread_pp:.2f} pp',(max(row.cluster_coverage_percent,row.random_coverage_percent),y),xytext=(5,0),textcoords='offset points',va='center',fontsize=6.8)
        ax.set_yticks(range(3),[f'{b.title()} (k={int(data.loc[b,"selected_facilities"]):,})' for b in data.index]);ax.invert_yaxis()
        ax.set_xlabel(f'{threshold}-minute coverage (%)')
        ax.set_xlim((67,91) if threshold==30 else (95.1,98.1))
        panel(ax,letter,f'Paired budget sensitivity: {threshold} min')
    ax=axes[1,1]
    data=exact.loc[exact.population_group.eq('all')].sort_values('threshold_minutes')
    for y,row in enumerate(data.itertuples()):
        ax.plot([row.country_composition_cluster_percent,row.random_coverage_percent],[y,y],color=TEAL,lw=2)
        ax.scatter([row.country_composition_cluster_percent,row.random_coverage_percent],[y,y],color=[GOLD,NAVY],s=24,zorder=3)
        ax.text(row.random_coverage_percent+.45,y,f'{row.constructed_spread_pp:.2f} pp',va='center',fontsize=7)
    ax.set_yticks(range(2),[f'{int(t)} min' for t in data.threshold_minutes]);ax.invert_yaxis()
    ax.set_xlim(77,102);ax.set_xlabel('Population coverage (%)')
    panel(ax,'d','Exact country-by-composition counts')
    fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=GOLD,label='Clustered'),Line2D([],[],marker='o',lw=0,color=NAVY,label='Random')],loc='outside lower center',ncol=2,fontsize=7)
    save(fig,'Figure_2')
    old.to_csv(OUT/'Figure_2_original_assignments.csv',index=False)
    pairs.to_csv(OUT/'Figure_2_budget_sensitivity.csv',index=False)
    exact.to_csv(OUT/'Figure_2_country_exact_composition.csv',index=False)

def reuse_unchanged():
    OUT.mkdir(exist_ok=True)
    for number in [1,3]:
        for source in (PREVIOUS/'figures/main').glob(f'Figure_{number}.*'):
            if source.suffix in ['.pdf','.svg','.png']:
                shutil.copy2(source,OUT/source.name)
        shutil.copy2(PREVIOUS/f'figures/main/Figure_{number}_source_data.csv',OUT/f'Figure_{number}_source_data.csv')
    supplementary=OUT/'supplement'
    supplementary.mkdir(exist_ok=True)
    for source in (PREVIOUS/'figures/supplement').iterdir():
        if source.suffix in ['.pdf','.png','.svg','.csv']:
            shutil.copy2(source,supplementary/source.name)

def figure4():
    from finalize_revision import combined_bounds
    bounds=pd.read_csv(ROOT/'clinical/results/m2_bounds.csv')
    bounds=combined_bounds(bounds)
    old=pd.read_csv(UPGRADE/'final_tables/stronger_constraint_widths.csv')
    rows=[]
    for service in NAMES:
        first=bounds[bounds.service.eq(service)].iloc[0]
        for threshold in [30,120]:
            oldrow=old.loc[old.country.eq('BRA') & old.service.eq(service) & old.threshold_minutes.eq(threshold) & old.population_group.eq('all')].iloc[0]
            rows.append({'label':f'Brazil {NAMES[service]}: original','threshold':threshold,
                'low':oldrow.low_percent,'high':oldrow.high_percent,
                'width':oldrow.width_lower_bound_pp,'color':GRAY})
        for r in bounds[bounds.service.eq(service)].to_dict('records'):
            for threshold in [30,120]:
                rows.append({'label':f'Brazil {NAMES[service]}: structure','threshold':threshold,
                    'low':r[f'minimum_{threshold}']*100,'high':r[f'maximum_{threshold}']*100,
                    'width':r[f'lower_bound_width_{threshold}_pp'],'color':TEAL})
    for service in ['chole','cesarean']:
        for threshold in [30,120]:
            r=old.loc[old.country.eq('MEX') & old.folder.eq('mexico_state_constraints') & old.service.eq(service) & old.threshold_minutes.eq(threshold) & old.population_group.eq('all')].iloc[0]
            rows.append({'label':f'Mexico {NAMES[service]}','threshold':threshold,'low':r.low_percent,'high':r.high_percent,'width':r.width_lower_bound_pp,'color':GOLD})
    data=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,2,figsize=(7.2,5),layout='constrained',sharey=True)
    for ax,threshold,letter in zip(axes,[30,120],['a','b']):
        part=data[data.threshold.eq(threshold)].reset_index(drop=True)
        for y,r in part.iterrows():
            ax.plot([r.low,r.high],[y,y],color=r.color,lw=1.7)
            ax.scatter([r.low,r.high],[y,y],color=r.color,s=19,zorder=3)
            ax.annotate(f'{r.width:.2f}',(r.high,y),xytext=(4,0),textcoords='offset points',va='center',fontsize=6.3)
        ax.set_yticks(range(len(part)),part.label,fontsize=6.8)
        ax.set_xlim(35 if threshold==30 else 78,104)
        ax.set_xlabel('Population coverage (%)')
        panel(ax,letter,f'{threshold}-minute feasible constructions')
        ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis()
    save(fig,'Figure_4')
    data.drop(columns='color').to_csv(OUT/'Figure_4_source_data.csv',index=False)

def figure5(prediction):
    merged=ROOT/'prediction/results/root_completion/flexible_n1_fixed/denominator_spatial_increment_summary_merged.csv'
    delta=pd.read_csv(merged if merged.exists() else prediction/'denominator_spatial_increment_summary.csv')
    temporal=pd.read_csv(prediction/'temporal_denominator_absolute_performance.csv')
    comparisons=[('population_urban_activity minus population_urban','Population + urban',GRAY),
                 ('strong_baseline_plus_activity minus strong_baseline','Strong',NAVY),
                 ('flexible_strong_baseline_plus_activity minus flexible_strong_baseline','Flexible strong',GOLD)]
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.6),layout='constrained')
    for ax,metric,letter,label in [(axes[0,0],'auc','a','Activity increment in record-weighted AUC'),(axes[0,1],'log_loss','b','Activity improvement in log loss')]:
        for i,service in enumerate(NAMES):
            for k,(comparison,name,color) in enumerate(comparisons):
                r=delta.loc[delta.service.eq(service)&delta.eligibility_threshold_resident_AIHS.eq(1)&delta.comparison.eq(comparison)&delta.metric.eq(metric)].iloc[0]
                sign=1 if metric=='auc' else -1
                value=r.mean_delta*sign
                lo,hi=sorted([r.repartition_p025*sign,r.repartition_p975*sign])
                y=i+(k-1)*.2
                ax.plot([lo,hi],[y,y],color=color,lw=1.3)
                ax.scatter(value,y,color=color,s=22,zorder=3)
        ax.axvline(0,color='gray',lw=.7,ls='--')
        ax.set_yticks(range(3),list(NAMES.values()),fontsize=6.8);ax.invert_yaxis()
        ax.set_xlabel('Extended − baseline AUC' if metric=='auc' else 'Baseline − extended log loss')
        panel(ax,letter,label)
    ax=axes[1,0]
    for service,color in zip(NAMES,[NAVY,TEAL,GOLD]):
        p=delta.loc[delta.service.eq(service)&delta.comparison.eq('strong_baseline_plus_activity minus strong_baseline')&delta.metric.eq('log_loss')].sort_values('eligibility_threshold_resident_AIHS')
        ax.plot(p.eligibility_threshold_resident_AIHS,-p.mean_delta,'o-',color=color,label=NAMES[service],ms=3.5,lw=1)
    ax.axhline(0,color='gray',ls='--',lw=.7);ax.set_xticks([1,5,10,20]);ax.set_xlabel('Minimum resident AIH records');ax.set_ylabel('Log-loss improvement')
    ax.legend(fontsize=6,loc='best');panel(ax,'c','Strong baseline: inclusion sensitivity')
    ax=axes[1,1]
    for i,service in enumerate(NAMES):
        p=temporal.loc[temporal.service.eq(service)&temporal.eligibility_threshold_resident_AIHS.eq(1)].set_index('model')
        a=float(p.loc['strong_baseline','auc']);b=float(p.loc['strong_baseline_plus_activity','auc'])
        ax.plot([a,b],[i,i],color=TEAL,lw=2);ax.scatter(a,i,facecolors='white',edgecolors=TEAL,s=28,zorder=3);ax.scatter(b,i,color=TEAL,s=20,zorder=3)
        ax.annotate(f'Δ {b-a:+.4f}',(max(a,b),i),xytext=(4,0),textcoords='offset points',va='center',fontsize=6.4)
    ax.set_yticks(range(3),list(NAMES.values()),fontsize=6.8);ax.invert_yaxis();ax.set_xlabel('Record-weighted AUC')
    lo,hi=ax.get_xlim();ax.set_xlim(lo,hi+(hi-lo)*.5)
    panel(ax,'d','2023→2024: strong baseline')
    ax.legend(handles=[Line2D([],[],marker='o',lw=0,markerfacecolor='white',markeredgecolor=TEAL,label='Strong'),Line2D([],[],marker='o',lw=0,color=TEAL,label='+Activity')],fontsize=6,loc='lower right')
    fig.legend(handles=[Line2D([],[],marker='o',color=c,lw=0,label=n) for _,n,c in comparisons],loc='outside lower center',ncol=3,fontsize=6.8)
    save(fig,'Figure_5')
    delta.to_csv(OUT/'Figure_5_spatial_source_data.csv',index=False)
    temporal.to_csv(OUT/'Figure_5_temporal_source_data.csv',index=False)

def figure6():
    metrics=pd.read_csv(ROOT/'clinical/results/m1_scenario_metrics.csv')
    ranks=pd.read_csv(ROOT/'clinical/results/m1_state_rank_change.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),layout='constrained')
    sets=['full_original','B_structure_positive_or_unknown']
    titles=['Original candidates','Surgical-centre candidates (A = B)']
    rules=['random','high_local_population_priority','low_local_population_priority']
    for ax,metric,letter,title in [(axes[0,0],'national_abs_error_pp','a','National coverage error'),(axes[0,1],'mean_absolute_state_error_pp','b','State coverage error')]:
        for i,candidate in enumerate(sets):
            for k,(rule,color) in enumerate(zip(rules,[GRAY,NAVY,GOLD])):
                part=metrics.loc[metrics.candidate_set.eq(candidate)&metrics.rule.eq(rule)]
                jitter=np.linspace(-.05,.05,len(part)) if len(part)>1 else np.zeros(1)
                ax.scatter(part[metric],i+(k-1)*.2+jitter,color=color,s=15,alpha=.75)
        ax.set_yticks(range(2),['Original','Structure']);ax.invert_yaxis();ax.set_xlabel('Absolute error (percentage points)')
        panel(ax,letter,title)
    for ax,candidate,title,letter in zip(axes[1],sets,titles,['c','d']):
        part=ranks.loc[ranks.candidate_set.eq(candidate)]
        for field,name,color in [('high_priority_rank_lowest_first','High local-pop.',NAVY),('low_priority_rank_lowest_first','Low local-pop.',GOLD)]:
            ax.scatter(part.observed_rank_lowest_first,part[field],s=20,color=color,label=name,alpha=.8)
        ax.plot([1,27],[1,27],color=GRAY,lw=.8,ls='--')
        ax.set(xlim=(0,28),ylim=(0,28),xlabel='Observed state rank',ylabel='Assigned state rank')
        ax.set_xticks([1,5,15,27]);ax.set_yticks([1,5,15,27]);panel(ax,letter,title)
    fig.legend(handles=[Line2D([],[],marker='o',color=c,lw=0,label=n) for c,n in [(GRAY,'Random (12)'),(NAVY,'High local population'),(GOLD,'Low local population')]],loc='outside lower center',ncol=3,fontsize=6.8)
    save(fig,'Figure_6')
    metrics.to_csv(OUT/'Figure_6_errors_source_data.csv',index=False)
    ranks.to_csv(OUT/'Figure_6_rank_source_data.csv',index=False)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',type=Path);args=parser.parse_args()
    reuse_unchanged()
    figure2()
    if (ROOT/'clinical/results/m2_bounds.csv').exists():
        figure4();figure6()
    if args.prediction:
        figure5(args.prediction)
