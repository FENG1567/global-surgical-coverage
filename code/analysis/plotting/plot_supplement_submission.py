"""Source-linked supplementary figures for the completed extension analyses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('__AUTHOR_OUTPUTS__/global_surg_access_upgrade_20261112')
DATA=ROOT/'server_results/derived'
OUT=Path('__AUTHOR_OUTPUTS__/submission_editor_20261113/figures/supplement');OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'pdf.fonttype':42,'svg.fonttype':'none','xtick.labelsize':6.5,'ytick.labelsize':6.5})
blue='#357596';gold='#BC8E45';gray='#676767'

def save(fig,name):
    name={'supplement_validation_and_mexico':'Figure_S1','supplement_information_value':'Figure_S2','supplement_brazil_multiservice':'Figure_S3','supplement_stronger_constraints':'Figure_S4'}[name]
    # Freeze the constrained layout before switching between export renderers.
    fig.set_dpi(600)
    fig.canvas.draw()
    fig.set_layout_engine('none')
    fig.savefig(OUT/f'{name}.png',dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    from PIL import Image
    with Image.open(OUT/f'{name}.png') as raster:
        raster.convert('RGB').save(OUT/f'{name}.tiff',compression='tiff_lzw',dpi=(600,600))
    plt.close(fig)

comparisons=pd.DataFrame(json.loads((DATA/'strict_chole_validation_extensions_v2/summary.json').read_text(encoding='utf8'))['comparisons'])
mexico=pd.read_csv(DATA/'mexico_joint_search/coverage_results.csv')
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,3.55),gridspec_kw={'width_ratios':[1.3,1]},layout='constrained')
labels=[];source=[]
for i,(scheme,comparison,label) in enumerate([
 ('spatial_blocks','population_urban_activity minus population_urban','Spatial blocks'),
 ('leave_region_out','population_urban_activity minus population_urban','Leave region out'),
 ('leave_region_out_buffered','population_urban_activity minus population_urban','Buffered region holdout'),
 ('spatial_blocks','prior_service_plus_activity minus prior_service_baseline','Prior-service baseline, blocks'),
 ('leave_region_out','prior_service_plus_activity minus prior_service_baseline','Prior-service baseline, regions'),
 ('leave_region_out_buffered','prior_service_plus_activity minus prior_service_baseline','Prior-service baseline, buffered'),
]):
    row=comparisons[comparisons.scheme.eq(scheme)&comparisons.comparison.eq(comparison)].iloc[0]
    color=blue if i<3 else gray
    value=float(row.mean_delta_auc)
    if row.n_repartitions>1:
        ax.plot([row.repartition_lo,row.repartition_hi],[i,i],color=color,lw=1.5)
    ax.scatter(value,i,s=25,color=color,zorder=3)
    labels.append(label);source.append(row.to_dict())
ax.axvline(0,color='#aaaaaa',lw=.7)
ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlabel('Incremental AUC from activity-based access')
ax.set_xlim(-.001,.018);ax.set_title('Brazil, strict cholecystectomy\n4,169 municipalities',loc='left',fontsize=8)
ax.text(-.08,1.08,'a',transform=ax.transAxes,fontweight='bold',fontsize=10)
for i,(service,label) in enumerate([('chole','Cholecystectomy'),('cesarean','Cesarean delivery')]):
    q=mexico[mexico.service.eq(service)&mexico.scenario.str.contains('observed|jointmatched|jointsearch')]
    lo,hi=q.all_30.min()*100,q.all_30.max()*100
    bx.plot([lo,hi],[i,i],color=blue,lw=2)
    for scenario,marker,color in [('observed','o',blue),('random_jointmatched','s',gray),('cluster_minimum','D',gold)]:
        if scenario=='cluster_minimum':
            cluster=q[q.scenario.str.contains('cluster_jointmatched|jointsearch')]
            value=cluster.all_30.min()*100
        else:
            value=mexico.loc[mexico.scenario.eq(service+'_'+scenario),'all_30'].iloc[0]*100
        bx.scatter(value,i,marker=marker,s=28,color=color,zorder=3,
            label={'observed':'Observed','random_jointmatched':'Random, fixed composition','cluster_minimum':'Clustered minimum'}[scenario] if i==0 else None)
    bx.annotate(f'At least {hi-lo:.2f} pp',(lo,i),xytext=(0,-22),textcoords='offset points',fontsize=7)
bx.set_yticks([0,1],['Cholecystectomy','Cesarean delivery']);bx.set_ylim(1.65,-.55)
bx.set_xlim(65,95);bx.set_xlabel('30-minute population coverage (%)')
bx.set_title('Mexico, fixed count\nand joint facility composition',loc='left',fontsize=8)
bx.legend(loc='lower right',fontsize=6,frameon=False)
bx.text(-.08,1.08,'b',transform=bx.transAxes,fontweight='bold',fontsize=10)
save(fig,'supplement_validation_and_mexico')
pd.DataFrame(source).to_csv(OUT/'validation_panel_source.csv',index=False)
mexico.to_csv(OUT/'mexico_panel_source.csv',index=False)

sim=pd.read_csv(DATA/'information_value_v2/audit_simulations.csv')
endpoints=pd.read_csv(DATA/'information_value_v2/exhaustive_endpoints.csv').set_index('iso')
fig,axes=plt.subplots(1,3,figsize=(7.2,2.75),sharey=True,layout='constrained')
for index,(ax,iso) in enumerate(zip(axes,['BLZ','DJI','GNB'])):
    sub=sim[sim.iso.eq(iso)]
    for prior,linestyle in [('count_only','-'),('count_and_exact_q','--')]:
        for strategy,color in [('random',blue),('largest_catchment',gold)]:
            data=sub[sub.prior.eq(prior)&sub.strategy.eq(strategy)]
            group=data.groupby('fraction').remaining_width
            mean=group.mean()*100
            ax.plot(mean.index*100,mean.values,color=color,ls=linestyle,lw=1.4,
                label=('Random' if strategy=='random' else 'Largest catchment')+(' + exact composition' if prior.endswith('exact_q') else ''))
            if prior=='count_only':
                low=group.quantile(.25)*100;high=group.quantile(.75)*100
                ax.fill_between(mean.index*100,low.values,high.values,color=color,alpha=.10,lw=0)
    row=endpoints.loc[iso]
    ax.set_title(f'{iso}, {int(row.k)}/{int(row.n)} facilities',loc='left',fontsize=8)
    ax.set_xticks([0,20,40,60]);ax.set_xlabel('Facility labels revealed (%)')
    ax.set_ylim(0,21);ax.text(-.1,1.08,'abc'[index],transform=ax.transAxes,fontweight='bold',fontsize=10)
axes[0].set_ylabel('Mean remaining coverage width (pp)')
axes[1].legend(loc='upper right',fontsize=5.5,frameon=False)
save(fig,'supplement_information_value')
sim.to_csv(OUT/'information_value_panel_source.csv',index=False)

bra_path=DATA/'brazil_spatial_v1/coverage_results.csv'
validation_path=DATA/'multiservice_validation/paired_increment_summary.csv'
if bra_path.exists() and validation_path.exists():
    bra=pd.read_csv(bra_path);validation=pd.read_csv(validation_path)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,3.25),layout='constrained')
    for i,(service,label) in enumerate([('chole','Cholecystectomy'),('cesarean','Cesarean delivery'),('appendectomy','Appendectomy')]):
        selected=bra[bra.service.eq(service)&bra.scenario.str.contains('observed|jointmatched|jointsearch')]
        lo,hi=selected.all_30.min()*100,selected.all_30.max()*100
        ax.plot([lo,hi],[i,i],color=blue,lw=2)
        for scenario,marker,color in [('observed','o',blue),('random_jointmatched','s',gray),('cluster_minimum','D',gold)]:
            if scenario=='cluster_minimum':
                clustered=selected[selected.scenario.str.contains('cluster_jointmatched|jointsearch')]
                value=clustered.all_30.min()*100
            else:
                value=bra.loc[bra.scenario.eq(service+'_'+scenario),'all_30'].iloc[0]*100
            ax.scatter(value,i,s=27,marker=marker,color=color,zorder=3,
                label={'observed':'Observed','random_jointmatched':'Random, fixed composition','cluster_minimum':'Clustered minimum'}[scenario] if i==0 else None)
        row=validation[validation.service.eq(service)&validation.comparison.eq('+activity - baseline')].iloc[0]
        bx.plot([row.lo,row.hi],[i,i],color=blue,lw=1.5);bx.scatter(row['mean'],i,color=blue,s=27)
        ax.annotate(f'{hi-lo:.2f} pp',(lo,i),xytext=(0,-15),textcoords='offset points',fontsize=7)
    for panel in [ax,bx]:
        panel.set_yticks(range(3),['Cholecystectomy','Cesarean delivery','Appendectomy']);panel.set_ylim(2.55,-.55)
    ax.set_xlabel('30-minute population coverage (%)');ax.legend(loc='lower right',fontsize=6,frameon=False)
    ax.set_title('Fixed facility count\nand joint facility composition',loc='left',fontsize=8)
    bx.axvline(0,color='#aaaaaa',lw=.7);bx.set_xlabel('Incremental AUC from 2017–2019 activity')
    bx.set_xticks([0,.005,.01,.015,.02])
    bx.set_title('2024 outcomes, repeated spatial blocks',loc='left',fontsize=8)
    for panel,label in [(ax,'a'),(bx,'b')]:panel.text(-.08,1.08,label,transform=panel.transAxes,fontweight='bold',fontsize=10)
    save(fig,'supplement_brazil_multiservice')
    bra.to_csv(OUT/'brazil_panel_source.csv',index=False)
    validation.to_csv(OUT/'multiservice_validation_panel_source.csv',index=False)
print('UPGRADE_FIGURES_EXPORTED',flush=True)

state=pd.read_csv(ROOT/'final_tables/stronger_constraint_widths.csv')
state=state[state.threshold_minutes.eq(30)&state.population_group.eq('all')&state.folder.str.contains('state_constraints')]
prior=pd.read_csv(ROOT/'final_tables/prior_service_adjusted_validation.csv')
simple=pd.read_csv(ROOT/'final_tables/multiservice_spatial_validation.csv')
temporal=pd.read_csv(ROOT/'final_tables/temporal_validation.csv')
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained')
order=[('BRA','chole'),('BRA','cesarean'),('BRA','appendectomy'),('MEX','chole'),('MEX','cesarean')]
service_labels={'chole':'Cholecystectomy','cesarean':'Cesarean delivery','appendectomy':'Appendectomy'}
for i,(country,service) in enumerate(order):
    row=state[state.country.eq(country)&state.service.eq(service)].iloc[0]
    ax.plot([row.low_percent,row.high_percent],[i,i],color=blue,lw=1.8)
    ax.scatter([row.low_percent,row.high_percent],[i,i],color=[gold,blue],marker='o',s=19,zorder=3)
    ax.text(row.low_percent-1,i,f'{row.width_lower_bound_pp:.2f}',ha='right',va='center',fontsize=6.5)
ax.set_yticks(range(5),[country+' '+service_labels[service] for country,service in order])
ax.set_ylim(4.6,-.6);ax.set_xlim(40,95);ax.set_xlabel('30-minute population coverage (%)')
ax.set_title('Exact state and joint-composition quotas\nConstructed width lower bounds (pp)',loc='left',fontsize=8)
for i,service in enumerate(['chole','cesarean','appendectomy']):
    raw=simple[simple.service.eq(service)].iloc[0]
    strong=prior[prior.service.eq(service)&prior.scheme.eq('spatial_blocks')].iloc[0]
    time=temporal[temporal.service.eq(service)].iloc[0]
    for offset,value,low,high,color,marker,label in [
        (-.18,raw['mean'],raw.lo,raw.hi,blue,'o','Population + urban fraction'),
        (0,strong.mean_delta_auc,strong.repartition_lo,strong.repartition_hi,gray,'s','Plus prior local service'),
        (.18,time.activity_delta_auc,None,None,gold,'D','2023 to 2024 time holdout'),
    ]:
        if low is not None:bx.plot([low,high],[i+offset,i+offset],color=color,lw=1.3)
        bx.scatter(value,i+offset,s=19,color=color,marker=marker,label=label if i==0 else None,zorder=3)
bx.axvline(0,color='#aaaaaa',lw=.7)
bx.set_xticks([0,.005,.01,.015,.02])
bx.set_yticks(range(3),[service_labels[s] for s in ['chole','cesarean','appendectomy']])
bx.set_ylim(2.85,-.55);bx.set_xlabel('Incremental AUC from prior activity access')
bx.set_title('Brazil, baseline and validation dependence',loc='left',fontsize=8)
bx.legend(loc='lower right',fontsize=5.8,frameon=False)
for panel,label in [(ax,'a'),(bx,'b')]:panel.text(-.08,1.08,label,transform=panel.transAxes,fontweight='bold',fontsize=10)
save(fig,'supplement_stronger_constraints')
state.to_csv(OUT/'state_constraint_panel_source.csv',index=False)
prior.to_csv(OUT/'strong_baseline_panel_source.csv',index=False)
