#!/usr/bin/env python3
"""Redesign Figures 1–6 from released aggregate CSVs; input tables are read-only.
SPDX-License-Identifier: MIT
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image

MM = 1 / 25.4
width_mm = 183
PNG_DPI = 300
TIFF_DPI = 600
DEFAULT_DATA = Path(__file__).resolve().parents[3] / "data" / "figure_sources"
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "artifacts" / "figures"
C = {"ink":"#20262C","navy":"#355C7D","teal":"#547A73","ochre":"#B07D3C",
     "plum":"#75618B","slate":"#58636E","gray":"#8B9298","light":"#D8DDE1"}
SERVICE = {"chole":("Cholecystectomy",C["navy"],"o"),
           "cesarean":("Cesarean delivery",C["teal"],"s"),
           "appendectomy":("Appendectomy",C["ochre"],"^")}
GROUP = {"all":"All","urban":"Urban","rural":"Rural",
         "measured_ratio_countries":"Measured ratio",
         "prior_or_unmatched_countries":"Prior/unmatched"}
BUDGET = {"lower":"Lower","central":"Central","upper":"Upper"}
COUNTRY = {"AND":"Andorra","BHS":"Bahamas","BLZ":"Belize","BRB":"Barbados","BRN":"Brunei",
 "COM":"Comoros","CUW":"Curaçao","DJI":"Djibouti","DMA":"Dominica","ERI":"Eritrea",
 "FSM":"Micronesia","GNB":"Guinea-Bissau","GNQ":"Equatorial Guinea","GRL":"Greenland",
 "ISL":"Iceland","LCA":"Saint Lucia","LUX":"Luxembourg","MLT":"Malta",
 "NCL":"New Caledonia","PYF":"French Polynesia","SLB":"Solomon Islands",
 "STP":"São Tomé and Príncipe","SUR":"Suriname","SYC":"Seychelles",
 "TON":"Tonga","VUT":"Vanuatu","WSM":"Samoa","CAF":"Central African Republic","SWZ":"Eswatini"}
COMP = {
 "population_urban_OSM minus population_urban":"Population + OSM",
 "population_urban_activity minus population_urban":"Population + activity",
 "flexible_population_urban_plus_activity minus flexible_population_urban":"Flexible population + activity",
 "strong_baseline_plus_activity minus strong_baseline":"Strong baseline + activity",
 "flexible_strong_baseline_plus_activity minus flexible_strong_baseline":"Flexible strong + activity"}
MODELS = {"population_urban":"Population + urban",
 "population_urban_OSM":"Population + urban + OSM",
 "population_urban_activity":"Population + urban + activity",
 "strong_baseline":"Strong baseline",
 "strong_baseline_plus_activity":"Strong baseline + activity"}
MODEL_MARKER = {"population_urban":"o","population_urban_OSM":"s",
 "population_urban_activity":"^","strong_baseline":"D","strong_baseline_plus_activity":"v"}

mpl.rcParams.update({
 "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans","sans-serif"],
 "font.size":7.5,"axes.labelsize":7.5,"axes.titlesize":8.0,"axes.titleweight":"bold",
 "axes.linewidth":.7,"axes.spines.top":False,"axes.spines.right":False,"axes.axisbelow":True,
 "xtick.labelsize":6.7,"ytick.labelsize":6.7,"xtick.major.width":.6,"ytick.major.width":.6,
 "xtick.major.size":2.6,"ytick.major.size":2.6,"legend.frameon":False,
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})

def style(ax, grid="x"):
    ax.spines["left"].set_color("#737B82"); ax.spines["bottom"].set_color("#737B82")
    ax.tick_params(colors=C["ink"], pad=2)
    if grid: ax.grid(axis=grid,color="#E7E9EB",linewidth=.45,zorder=0)

def panel(ax, letter, title):
    ax.set_title(title,loc="left",pad=7,color=C["ink"])
    ax.text(-.105,1.045,letter,transform=ax.transAxes,fontsize=8.4,fontweight="bold",
            ha="left",va="bottom",color=C["ink"],clip_on=False)

def need(df,name,cols):
    missing=sorted(set(cols)-set(df.columns))
    if missing: raise ValueError(f"{name} missing columns: {missing}")

def savefig(fig,out,name,height_mm):
    fig.set_size_inches(width_mm*MM,height_mm*MM,forward=True)
    for ext in ("pdf","svg"):
        fig.savefig(out/f"{name}.{ext}",format=ext,facecolor="white")
    fig.savefig(out/f"{name}.png",format="png",dpi=PNG_DPI,facecolor="white")
    tif=out/f"{name}.tiff"
    fig.savefig(tif,format="tiff",dpi=TIFF_DPI,facecolor="white",
                pil_kwargs={"compression":"tiff_lzw"})
    with Image.open(tif) as image:
        rgb=image.convert("RGB"); temp=out/f".{name}.rgbtmp.tiff"
        rgb.save(temp,format="TIFF",compression="tiff_lzw",dpi=(TIFF_DPI,TIFF_DPI))
    temp.replace(tif); plt.close(fig)

def read_sources(folder):
    names=["Figure_1_source_data","Figure_2_original_assignments","Figure_2_budget_sensitivity",
      "Figure_2_country_exact_composition","Figure_3_source_data","Figure_4_source_data",
      "Figure_5_spatial_source_data","Figure_5_temporal_source_data",
      "Figure_6_errors_source_data","Figure_6_rank_source_data"]
    return {n:pd.read_csv(folder/f"{n}.csv") for n in names}

def file_hash(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def figure1(d,out):
    data=d["Figure_1_source_data"]
    need(data,"Figure 1",["defn","thr_min","all","urban","rural","D1_urban_minus_rural_pp"])
    labels={"D0u":"Any health facility","D1":"OSM hospitals","D2meas_corr":"Budget + local population","D1walk":"Walking mode"}
    colors={"D0u":C["navy"],"D1":C["teal"],"D2meas_corr":C["ochre"],"D1walk":C["plum"]}
    markers={"D0u":"o","D1":"s","D2meas_corr":"^","D1walk":"D"}
    if set(data.defn.unique())!=set(labels): raise ValueError("F1 definitions changed.")
    fig,(a,b)=plt.subplots(2,1,figsize=(width_mm*MM,124*MM),gridspec_kw={"height_ratios":[1.08,1]})
    handles=[]
    for key,label in labels.items():
        part=data.loc[data.defn.eq(key)].sort_values("thr_min")
        a.plot(part.thr_min,part["all"]*100,color=colors[key],marker=markers[key],ms=3.1,lw=1.25,zorder=3)
        handles.append(Line2D([],[],color=colors[key],marker=markers[key],lw=1.2,markersize=3,label=label))
    for ax in (a,b):
        ax.set_xlim(8,188); ax.set_ylim(0,104); ax.set_xticks([15,30,45,60,90,120,180])
        ax.set_yticks([0,20,40,60,80,100]); ax.set_xlabel("Travel-time threshold (min)")
        ax.set_ylabel("Population coverage (%)"); style(ax)
    a.axvline(120,color=C["slate"],lw=.75,ls="--",zorder=1)
    panel(a,"a","Population coverage across facility definitions and modes")
    a.legend(handles=handles,loc="lower left",bbox_to_anchor=(.015,.025),ncol=2,fontsize=6.1,
             handlelength=1.5,columnspacing=.9,labelspacing=.25)
    d1=data.loc[data.defn.eq("D1")].sort_values("thr_min")
    b.plot(d1.thr_min,d1.urban*100,color=C["navy"],marker="o",lw=1.35,ms=3.2,label="Urban",zorder=3)
    b.plot(d1.thr_min,d1.rural*100,color=C["ochre"],marker="s",lw=1.35,ms=3.2,label="Rural",zorder=3)
    gaps={}; offsets={15:(10,-28),30:(30,-28),120:(8,10)}
    for minute in (15,30,120):
        row=d1.loc[d1.thr_min.eq(minute)]
        if len(row)!=1: raise ValueError(f"Expected one D1 value at {minute} min.")
        urban=float(row.urban.iloc[0]*100); rural=float(row.rural.iloc[0]*100); gap=urban-rural
        gaps[str(minute)]=round(gap,6)
        b.vlines(minute,rural,urban,color=C["slate"],lw=.9,linestyle=(0,(2,1)),zorder=2)
        b.annotate(f"{minute} min: {gap:.1f} pp",(minute,(urban+rural)/2),
          xytext=offsets[minute],textcoords="offset points",fontsize=6,color=C["ink"],
          arrowprops={"arrowstyle":"-","lw":.45,"color":C["slate"]},
          bbox={"boxstyle":"round,pad=.15","fc":"white","ec":"none","alpha":.94})
    panel(b,"b","OSM hospitals: urban and rural coverage")
    b.legend(loc="lower right",ncol=2,fontsize=6.4,handlelength=1.5,columnspacing=1)
    fig.subplots_adjust(left=.105,right=.98,top=.95,bottom=.075,hspace=.60)
    savefig(fig,out,"Figure_1",124)
    return {"source_rows":len(data),"panel_a_points":len(data),"panel_b_points":len(d1)*2,"d1_gap_pp":gaps}

def paired_panel(ax,part,xcluster,xrandom,ylabels,xlabel,title,letter):
    values=np.r_[part[xcluster].to_numpy(float),part[xrandom].to_numpy(float)]
    lo,hi=float(values.min()),float(values.max()); span=max(hi-lo,1.0)
    ax.set_xlim(lo-span*.055,hi+span*.18)
    for y,(_,row) in enumerate(part.iterrows()):
        cluster=float(row[xcluster]); random=float(row[xrandom])
        ax.plot([cluster,random],[y,y],color="#BBC1C6",lw=1.1,zorder=1)
        ax.scatter(cluster,y,s=20,color=C["ochre"],edgecolor="white",lw=.3,zorder=3)
        ax.scatter(random,y,s=20,color=C["navy"],edgecolor="white",lw=.3,zorder=3)
        ax.annotate(f"{row.constructed_spread_pp:.2f}",(max(cluster,random),y),
                    xytext=(4,0),textcoords="offset points",fontsize=5.9,color=C["ink"],
                    ha="left",va="center",clip_on=False)
    ax.set_yticks(np.arange(len(part)),ylabels,fontsize=5.9); ax.invert_yaxis()
    ax.set_xlabel(xlabel); style(ax); panel(ax,letter,title)


def figure2(d,out):
    old=d["Figure_2_original_assignments"].copy(); pairs=d["Figure_2_budget_sensitivity"].copy()
    exact=d["Figure_2_country_exact_composition"].copy()
    need(old,"F2 original",["scenario_label","coverage_30_percent"])
    need(pairs,"F2 budget",["budget","population_group","threshold_minutes","selected_facilities","random_coverage_percent","cluster_coverage_percent","constructed_spread_pp"])
    need(exact,"F2 exact",["population_group","threshold_minutes","random_coverage_percent","country_composition_cluster_percent","constructed_spread_pp"])
    old=old.loc[old.scenario_label.notna()].copy()
    order=["Random","High local-pop.","Low local-pop.","Clustered"]
    old=old.set_index("scenario_label").loc[order].reset_index()
    random=float(old.loc[old.scenario_label.eq("Random"),"coverage_30_percent"].iloc[0])
    cluster=float(old.loc[old.scenario_label.eq("Clustered"),"coverage_30_percent"].iloc[0])
    original_gap=random-cluster
    if len(old)!=4 or not np.isclose(original_gap,5.58,atol=.01): raise ValueError("F2 original difference changed.")
    fig,axes=plt.subplots(2,2,figsize=(width_mm*MM,148*MM))
    colors={"Random":C["navy"],"Clustered":C["ochre"],"High local-pop.":C["teal"],"Low local-pop.":C["gray"]}
    markers={"Random":"o","Clustered":"o","High local-pop.":"D","Low local-pop.":"s"}
    ax=axes[0,0]; panel(ax,"a","Original central-budget assignments")
    for y,row in old.iterrows():
        value=float(row.coverage_30_percent)
        ax.scatter(value,y,s=30,color=colors[row.scenario_label],marker=markers[row.scenario_label],edgecolor="white",lw=.35,zorder=3)
        ax.annotate(f"{value:.2f}%",(value,y),xytext=(4,0),textcoords="offset points",fontsize=6.3,va="center",color=C["ink"])
    ax.set_yticks(np.arange(4),order,fontsize=6.4); ax.invert_yaxis(); ax.set_xlim(76,91)
    ax.set_xlabel("30-minute coverage (%)"); style(ax)
    group=pairs.loc[pairs.population_group.eq("all")]
    for ax,threshold,letter in [(axes[0,1],30,"b"),(axes[1,0],120,"c")]:
        part=group.loc[group.threshold_minutes.eq(threshold)].copy()
        part=part.set_index("budget").loc[["lower","central","upper"]].reset_index()
        labels=[BUDGET[b].lower() for b in part.budget]
        paired_panel(ax,part,"cluster_coverage_percent","random_coverage_percent",labels,
          f"{threshold}-minute population coverage (%)",f"Paired budget sensitivity: {threshold} min",letter)
    part=exact.loc[(exact.population_group.eq("all"))&exact.threshold_minutes.isin([30,120])].copy()
    part=part.sort_values("threshold_minutes").reset_index(drop=True)
    labels=[f"{int(v)} min" for v in part.threshold_minutes]
    paired_panel(axes[1,1],part,"country_composition_cluster_percent","random_coverage_percent",labels,
      "Population coverage (%)","Exact country-composition pairs","d")
    handles=[Line2D([],[],color=colors[n],marker=markers[n],lw=0,markersize=4.2,label=n) for n in order]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.145),ncol=4,fontsize=6.3,columnspacing=1,handletextpad=.35)
    fig.text(.5,.055,"Panel a is the original assignment realization (random − clustered = 5.58 pp); panels b–d are separate paired constructions.",
      ha="center",va="center",fontsize=6.1,color=C["slate"])
    fig.subplots_adjust(left=.24,right=.985,top=.91,bottom=.23,wspace=.42,hspace=.56)
    savefig(fig,out,"Figure_2",148)
    return {"source_rows":{"original":len(old),"budget":len(pairs),"exact_composition":len(exact)},
      "plotted_points":{"a":4,"b_pairs":3,"c_pairs":3,"d_pairs":len(part)},
      "subgroups_remain_in_source_tables":True,"original_random_minus_clustered_pp":round(original_gap,8),
      "budget_spreads_match":bool(np.allclose(group.constructed_spread_pp,group.random_coverage_percent-group.cluster_coverage_percent,atol=1e-8)),
      "exact_spreads_match":bool(np.allclose(exact.loc[exact.population_group.eq("all")].constructed_spread_pp,
       exact.loc[exact.population_group.eq("all")].random_coverage_percent-exact.loc[exact.population_group.eq("all")].country_composition_cluster_percent,atol=1e-8))}


def figure3(d,out):
    raw=d["Figure_3_source_data"]
    need(raw,"Figure 3",["iso","iso3","width","width_pp","all_facility_coverage_percent","contrast_pp","rank_contrast_ascending","C_rural","C_urban"])
    certified=raw.loc[raw.iso.notna()].copy().sort_values("width_pp",ascending=False).reset_index(drop=True)
    signed=raw.loc[raw.iso3.notna()].copy().sort_values("contrast_pp").reset_index(drop=True)
    if len(certified)!=27 or len(signed)!=162: raise ValueError("F3 requires all 27 and 162 rows.")
    if not np.allclose(certified.width_pp,certified.width*100,atol=1e-7): raise ValueError("F3 width values changed.")
    if not np.allclose(signed.contrast_pp,(signed.C_rural-signed.C_urban)*100,atol=1e-6): raise ValueError("F3 contrasts changed.")
    certified["country_name"]=certified.iso.map(lambda x:COUNTRY.get(str(x),str(x)))
    certified["country_label"]=certified.apply(lambda r:f"{r.country_name} ({r.iso})",axis=1)
    signed["rank_percent"]=signed.rank_contrast_ascending/len(signed)*100
    median=float(certified.width_pp.median()); p90=float(certified.width_pp.quantile(.90))
    rho=float(certified.all_facility_coverage_percent.corr(certified.width_pp,method="spearman"))
    fig=plt.figure(figsize=(width_mm*MM,164*MM))
    grid=fig.add_gridspec(2,2,height_ratios=[1.72,1],hspace=.48,wspace=.30)
    a=fig.add_subplot(grid[0,:]); b=fig.add_subplot(grid[1,0]); c=fig.add_subplot(grid[1,1])
    panel(a,"a","Certified full-population country widths")
    y=np.arange(len(certified)); colors=[C["ochre"] if iso=="COM" else C["navy"] for iso in certified.iso]
    a.barh(y,certified.width_pp,color=colors,height=.68,edgecolor="white",linewidth=.25,zorder=2)
    a.set_yticks(y,certified.country_label,fontsize=6.2); a.invert_yaxis()
    a.set_xlim(0,46); a.set_xticks([0,10,20,30,40]); a.set_xlabel("Certified country width (percentage points)")
    a.axvline(median,color=C["ochre"],ls="--",lw=.9,zorder=1); a.axvline(p90,color=C["plum"],ls=":",lw=.95,zorder=1)
    a.text(.99,.985,f"Median {median:.2f} pp · p90 {p90:.2f} pp",transform=a.transAxes,ha="right",va="top",fontsize=6.2,color=C["slate"])
    style(a); a.grid(axis="y",visible=False)
    panel(b,"b","Coverage and width")
    b.text(.98,.98,f"Spearman ρ = {rho:.2f}",transform=b.transAxes,ha="right",va="top",fontsize=6.0,color=C["slate"])
    b.scatter(certified.all_facility_coverage_percent,certified.width_pp,s=23,c=colors,edgecolor="white",linewidth=.35,zorder=3)
    b.set_xlim(20,103); b.set_ylim(-1,max(45,float(certified.width_pp.max())+2))
    b.set_xticks([30,50,70,90,100]); b.set_xlabel("All-facility coverage (%)"); b.set_ylabel("Certified width (pp)"); style(b)
    panel(c,"c","Allocation contrast (162 countries)")
    point_colors=np.full(len(signed),C["gray"],dtype=object); point_colors[0]=C["plum"]; point_colors[-1]=C["navy"]
    c.scatter(signed.contrast_pp,signed.rank_percent,s=13,c=point_colors,edgecolor="white",linewidth=.22,zorder=3)
    c.axvline(0,color=C["ink"],lw=.7,zorder=1); c.set_ylim(0,103); c.set_yticks([0,25,50,75,100])
    c.set_xlim(min(-31,float(signed.contrast_pp.min())-2),max(31,float(signed.contrast_pp.max())+2))
    c.set_xlabel("C(u=0) − C(u=1) (pp)"); c.set_ylabel("Sorted country rank (%)")
    low=signed.iloc[0]; high=signed.iloc[-1]
    for row,where,label in [(low,(-28,18),f"Central African Republic\n(CAF) {low.contrast_pp:+.1f} pp"),
                             (high,(11,73),f"Eswatini (SWZ)\n{high.contrast_pp:+.1f} pp")]:
        c.annotate(label,(row.contrast_pp,row.rank_percent),
          xytext=where,textcoords="data",ha="left",va="center",fontsize=6.0,color=C["ink"],
          arrowprops={"arrowstyle":"-","lw":.45,"color":C["slate"]})
    style(c); c.grid(axis="y",color="#EEF0F2",linewidth=.35)
    fig.subplots_adjust(left=.265,right=.98,top=.955,bottom=.11)
    savefig(fig,out,"Figure_3",164)
    return {"source_rows":len(raw),"certified_countries":len(certified),"signed_contrast_countries":len(signed),
      "median_width_pp":round(median,6),"p90_width_pp":round(p90,6),"spearman_rho":round(rho,6),
      "rank_range_percent":[round(float(signed.rank_percent.min()),4),round(float(signed.rank_percent.max()),4)],
      "contrast_definition":"C(u=0) - C(u=1)",
      "extreme_contrasts":[{"iso3":str(low.iso3),"name":COUNTRY.get(str(low.iso3),str(low.iso3)),"pp":round(float(low.contrast_pp),6)},
                           {"iso3":str(high.iso3),"name":COUNTRY.get(str(high.iso3),str(high.iso3)),"pp":round(float(high.contrast_pp),6)}]}


def figure4(d,out):
    data=d["Figure_4_source_data"].copy()
    need(data,"Figure 4",["label","threshold","low","high","width"])
    if len(data)!=16 or not np.allclose(data.width,data.high-data.low,atol=1e-7): raise ValueError("F4 source endpoints changed.")
    def classify(label):
        if label.startswith("Mexico"):
            service=label.replace("Mexico ","").replace("Cholecystectomy","Chole").replace("Cesarean delivery","Cesarean")
            return f"Mexico · {service}","Mexico state constraint"
        service=label.replace("Brazil ","").split(":")[0].replace("Cholecystectomy","Chole").replace("Cesarean delivery","Cesarean")
        strategy="original" if label.endswith("original") else "screened"
        return f"{service} · {strategy}",("Brazil original" if strategy=="original" else "Brazil structural screen")
    colors={"Brazil original":C["gray"],"Brazil structural screen":C["teal"],"Mexico state constraint":C["ochre"]}
    data["display"],data["group"]=zip(*data.label.map(classify))
    order=["Chole · original","Chole · screened","Cesarean · original","Cesarean · screened",
      "Appendectomy · original","Appendectomy · screened","Mexico · Chole","Mexico · Cesarean"]
    fig,axes=plt.subplots(1,2,figsize=(width_mm*MM,132*MM))
    for ax,threshold,letter in zip(axes,[30,120],["a","b"]):
        part=data.loc[data.threshold.eq(threshold)].copy()
        part["_order"]=part.display.map({v:i for i,v in enumerate(order)})
        part=part.sort_values("_order").reset_index(drop=True)
        if len(part)!=8: raise ValueError(f"F4 needs 8 rows at {threshold} min.")
        panel(ax,letter,f"{threshold}-minute coverage")
        for y,row in part.iterrows():
            color=colors[row.group]; mid=(row.low+row.high)/2
            ax.plot([row.low,row.high],[y,y],color=color,lw=1.55,zorder=2)
            ax.scatter([row.low,row.high],[y,y],color=color,s=19,edgecolor="white",linewidth=.35,zorder=3)
            ax.text(mid,y-.23,f"{row.width:.2f} pp",ha="center",va="bottom",fontsize=5.8,color=C["ink"],clip_on=False)
        ax.set_yticks(np.arange(8),part.display,fontsize=6.3); ax.invert_yaxis()
        ax.set_xlabel("Constructed population coverage (%)"); ax.set_xlim(34 if threshold==30 else 77,100); style(ax)
    handles=[Line2D([],[],color=C["gray"],lw=2,label="Brazil original"),
       Line2D([],[],color=C["teal"],lw=2,label="Brazil screened (A = B)"),
       Line2D([],[],color=C["ochre"],lw=2,label="Mexico state constraint")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.025),ncol=3,fontsize=6.2,handlelength=1.5,columnspacing=.8)
    fig.subplots_adjust(left=.245,right=.985,top=.91,bottom=.19,wspace=.55)
    savefig(fig,out,"Figure_4",132)
    return {"source_rows":len(data),"rows_by_threshold":{str(int(k)):int(v) for k,v in data.groupby("threshold").size().items()},
       "endpoint_spread_relation_holds":True,"structural_screen_note":"A and B sets combined over 17 unique configurations"}


def figure5(d,out):
    spatial=d["Figure_5_spatial_source_data"].copy(); temporal=d["Figure_5_temporal_source_data"].copy()
    need(spatial,"F5 spatial",["service","eligibility_threshold_resident_AIHS","comparison","metric","mean_delta","repartition_p025","repartition_p975","n_repartitions"])
    need(temporal,"F5 temporal",["service","eligibility_threshold_resident_AIHS","model","train_year","test_year","auc"])
    if len(spatial)!=192 or len(temporal)!=30: raise ValueError("F5 source row counts changed.")
    comps=["population_urban_activity minus population_urban",
      "strong_baseline_plus_activity minus strong_baseline",
      "flexible_strong_baseline_plus_activity minus flexible_strong_baseline"]
    comp_labels=["Population + urban","Strong baseline","Flexible strong baseline"]
    comp_colors=[C["gray"],C["navy"],C["ochre"]]; services=list(SERVICE)
    fig,axes=plt.subplots(2,2,figsize=(width_mm*MM,142*MM)); plot_values=[]
    expected={
      ("chole",comps[0]):.00090,("cesarean",comps[0]):-.00003,("appendectomy",comps[0]):.00130,
      ("chole",comps[1]):.00724,("cesarean",comps[1]):.00127,("appendectomy",comps[1]):.00160,
      ("chole",comps[2]):.00195,("cesarean",comps[2]):.00036,("appendectomy",comps[2]):.00069}
    auc1=spatial.loc[spatial.eligibility_threshold_resident_AIHS.eq(1)&spatial.metric.eq("auc")]
    for service in services:
        for comp in comps:
            row=auc1.loc[auc1.service.eq(service)&auc1.comparison.eq(comp)]
            if len(row)!=1: raise ValueError(f"F5 AUC comparison mismatch for {service}/{comp}.")
            actual=float(row.mean_delta.iloc[0])
            if not np.isclose(actual,expected[(service,comp)],atol=.0002):
                raise ValueError(f"F5 AUC sanity value mismatch for {service}/{comp}: {actual:.8f}.")
            plot_values.append({"panel":"a","service":service,"comparison":comp,"metric":"auc",
                "threshold":1,"source_mean_delta":actual,"displayed_mean_delta":actual,
                "q025":float(row.repartition_p025.iloc[0]),"q975":float(row.repartition_p975.iloc[0])})
    offsets=[-.22,0,.22]
    for ax,metric,letter,title in [
      (axes[0,0],"auc","a","AUC increment at threshold 1"),
      (axes[0,1],"log_loss","b","Log-loss improvement at threshold 1")]:
        panel(ax,letter,title)
        part=spatial.loc[spatial.eligibility_threshold_resident_AIHS.eq(1)&spatial.metric.eq(metric)&spatial.comparison.isin(comps)]
        if len(part)!=9: raise ValueError(f"F5 {metric} expected nine original points, found {len(part)}.")
        for si,service in enumerate(services):
            for ci,comp in enumerate(comps):
                row=part.loc[part.service.eq(service)&part.comparison.eq(comp)]
                if len(row)!=1: raise ValueError("F5 baseline/service row missing or duplicated.")
                r=row.iloc[0]; raw=float(r.mean_delta); lo=float(r.repartition_p025); hi=float(r.repartition_p975)
                value=raw
                if metric=="log_loss": value,lo,hi=-raw,-hi,-lo
                lo=min(lo,value); hi=max(hi,value); y=si+offsets[ci]
                ax.errorbar(value,y,xerr=np.array([[value-lo],[hi-value]]),fmt="o",color=comp_colors[ci],
                    ecolor=comp_colors[ci],elinewidth=.85,capsize=1.7,markersize=3.0,
                    markeredgecolor="white",markeredgewidth=.3,zorder=3)
                if metric=="log_loss":
                    plot_values.append({"panel":"b","service":service,"comparison":comp,"metric":"log_loss",
                       "threshold":1,"source_mean_delta":raw,"displayed_mean_delta":value,
                       "q025":-float(r.repartition_p975),"q975":-float(r.repartition_p025)})
        ax.axvline(0,color=C["ink"],lw=.65,ls="--",zorder=1)
        ax.set_yticks(np.arange(3),[SERVICE[s][0] for s in services],fontsize=6.5); ax.invert_yaxis()
        ax.set_xlabel("Δ AUC" if metric=="auc" else "Baseline − extended log loss")
        vals=[]
        for _,r in part.iterrows():
            lo=float(r.repartition_p025); hi=float(r.repartition_p975); value=float(r.mean_delta)
            if metric=="log_loss": lo,hi,value=-hi,-lo,-value
            vals.extend([lo,hi,value])
        low=min(vals); high=max(vals); pad=max((high-low)*.12,.00025); ax.set_xlim(low-pad,high+pad)
        ax.set_xticks([0,.002,.004,.006,.008]); ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:.3f}"))
        style(ax)
    panel(axes[1,0],"c","Strong-baseline log-loss sensitivity")
    strong="strong_baseline_plus_activity minus strong_baseline"
    sens=spatial.loc[spatial.comparison.eq(strong)&spatial.metric.eq("log_loss")]
    if len(sens)!=12 or set(sens.eligibility_threshold_resident_AIHS.unique())!={1,5,10,20}:
        raise ValueError("F5 sensitivity must retain 12 source means across 4 thresholds.")
    service_handles=[]
    for service in services:
        part=sens.loc[sens.service.eq(service)].sort_values("eligibility_threshold_resident_AIHS")
        axes[1,0].plot(part.eligibility_threshold_resident_AIHS,-part.mean_delta,
            marker=SERVICE[service][2],color=SERVICE[service][1],lw=1.0,ms=3.2)
        service_handles.append(Line2D([],[],color=SERVICE[service][1],marker=SERVICE[service][2],
            lw=1,markersize=3.5,label=SERVICE[service][0]))
        for _,r in part.iterrows():
            plot_values.append({"panel":"c","service":service,"comparison":strong,"metric":"log_loss",
              "threshold":int(r.eligibility_threshold_resident_AIHS),"source_mean_delta":float(r.mean_delta),
              "displayed_mean_delta":float(-r.mean_delta),"q025":None,"q975":None})
    axes[1,0].set_xticks([1,5,10,20]); axes[1,0].set_xlim(.5,20.8)
    axes[1,0].set_xlabel("Minimum resident AIH records"); axes[1,0].set_ylabel("Baseline − extended log loss")
    axes[1,0].set_ylim(0,.0051); axes[1,0].axhline(0,color=C["slate"],lw=.6,ls="--")
    style(axes[1,0])

    panel(axes[1,1],"d","Temporal AUC at threshold 1")
    part=temporal.loc[temporal.eligibility_threshold_resident_AIHS.eq(1)]
    if len(part)!=15 or set(part.train_year.unique())!={2023} or set(part.test_year.unique())!={2024}:
        raise ValueError("F5 temporal panel must retain threshold-1 2023/2024 rows.")
    deltas={}
    for si,service in enumerate(services):
        service_rows=part.loc[part.service.eq(service)].set_index("model")
        base=float(service_rows.loc["strong_baseline","auc"])
        extended=float(service_rows.loc["strong_baseline_plus_activity","auc"])
        delta=extended-base; deltas[service]=round(delta,8)
        axes[1,1].plot([base,extended],[si,si],color="#AEB5BA",lw=1.4,zorder=1)
        axes[1,1].scatter(base,si,s=25,color=C["gray"],edgecolor="white",linewidth=.35,zorder=3)
        axes[1,1].scatter(extended,si,s=25,color=C["teal"],edgecolor="white",linewidth=.35,zorder=3)
        axes[1,1].annotate(f"Δ {delta:+.5f}",(max(base,extended),si),xytext=(4,0),textcoords="offset points",
            fontsize=6.0,color=C["ink"],va="center",clip_on=False)
        for model in ["strong_baseline","strong_baseline_plus_activity"]:
            plot_values.append({"panel":"d","service":service,"comparison":model,"metric":"auc",
              "threshold":1,"source_mean_delta":float(service_rows.loc[model,"auc"]),
              "displayed_mean_delta":float(service_rows.loc[model,"auc"]),"q025":None,"q975":None})
    axes[1,1].set_yticks(np.arange(3),[SERVICE[s][0] for s in services],fontsize=6.5)
    axes[1,1].invert_yaxis(); axes[1,1].set_xlabel("AUC in 2024 holdout")
    selected_temp=part.loc[part.model.isin(["strong_baseline","strong_baseline_plus_activity"])]
    axes[1,1].set_xlim(min(selected_temp.auc)-.004,max(selected_temp.auc)+.028); style(axes[1,1])
    comp_handles=[Line2D([],[],color=comp_colors[i],marker="o",lw=0,markersize=3.8,label=comp_labels[i]) for i in range(3)]
    temporal_handles=[Line2D([],[],color=C["gray"],marker="o",lw=0,markersize=3.8,label="Strong baseline"),
                      Line2D([],[],color=C["teal"],marker="o",lw=0,markersize=3.8,label="Strong + activity")]
    fig.legend(handles=comp_handles,loc="lower center",bbox_to_anchor=(.5,.025),ncol=3,fontsize=6.1,columnspacing=.9,handletextpad=.3,
        title="Baseline in panels a and b",title_fontsize=6.1)
    axes[1,0].legend(handles=service_handles,loc="center right",bbox_to_anchor=(.99,.57),ncol=1,fontsize=6.1,handletextpad=.4)
    axes[1,1].legend(handles=temporal_handles,loc="lower left",ncol=1,fontsize=6.0,handletextpad=.35)
    fig.subplots_adjust(left=.225,right=.98,top=.92,bottom=.16,wspace=.36,hspace=.78)
    savefig(fig,out,"Figure_5",142)
    pd.DataFrame(plot_values).to_csv(out/"Figure_5_plot_values.csv",index=False)
    return {"source_rows":{"spatial":len(spatial),"temporal":len(temporal)},
      "panel_points":{"a":9,"b":9,"c":12,"d":6},"comparisons_in_a_b":comps,
      "spatial_repartitions":sorted(int(x) for x in spatial.n_repartitions.unique() if pd.notna(x)),
      "temporal_threshold":1,"temporal_delta_auc_strong_activity_minus_strong":deltas,
      "expected_n1_auc_increment_checks":"passed","plot_values_csv":"Figure_5_plot_values.csv"}


def figure6(d,out):
    errors=d["Figure_6_errors_source_data"].copy(); ranks=d["Figure_6_rank_source_data"].copy()
    need(errors,"F6 errors",["candidate_set","rule","scenario","national_abs_error_pp","mean_absolute_state_error_pp"])
    need(ranks,"F6 ranks",["candidate_set","state_code","state","observed_rank_lowest_first","high_priority_rank_lowest_first","low_priority_rank_lowest_first"])
    candidates=["full_original","B_structure_positive_or_unknown"]
    rules=["random","high_local_population_priority","low_local_population_priority"]
    counts=errors.groupby(["candidate_set","rule"]).size().to_dict()
    for candidate in candidates:
        if [counts.get((candidate,r),0) for r in rules]!=[12,1,1]: raise ValueError(f"F6 point counts changed for {candidate}.")
    if len(errors)!=28 or len(ranks)!=54: raise ValueError("F6 source row counts changed.")
    fig,axes=plt.subplots(2,2,figsize=(width_mm*MM,142*MM))
    rule_name={"random":"random (12)","high_local_population_priority":"high priority","low_local_population_priority":"low priority"}
    rule_color={"random":C["gray"],"high_local_population_priority":C["navy"],"low_local_population_priority":C["ochre"]}
    rule_marker={"random":"o","high_local_population_priority":"^","low_local_population_priority":"s"}
    order=[(c,r) for c in candidates for r in rules]; ymap={v:i for i,v in enumerate(order)}
    ylabels=[f"{'Full' if c=='full_original' else 'B'} · {rule_name[r]}" for c,r in order]
    for ax,metric,letter,title,xlabel in [
      (axes[0,0],"national_abs_error_pp","a","National coverage error","Absolute error (pp)"),
      (axes[0,1],"mean_absolute_state_error_pp","b","Mean state error","Mean absolute error (pp)")]:
        panel(ax,letter,title)
        for candidate,rule in order:
            sub=errors.loc[errors.candidate_set.eq(candidate)&errors.rule.eq(rule)].sort_values("scenario")
            y0=ymap[(candidate,rule)]; jitter=np.linspace(-.12,.12,len(sub)) if rule=="random" else np.zeros(len(sub))
            ax.scatter(sub[metric].to_numpy(float),y0+jitter,s=16,color=rule_color[rule],marker=rule_marker[rule],
              alpha=.88,edgecolor="white",linewidth=.25,zorder=3)
        ax.set_yticks(np.arange(len(ylabels)),ylabels,fontsize=6.1); ax.invert_yaxis()
        ax.set_xlabel(xlabel); ax.xaxis.set_major_locator(MaxNLocator(4)); style(ax)
    for ax,candidate,letter,title in [
      (axes[1,0],"full_original","c","Full candidate set"),
      (axes[1,1],"B_structure_positive_or_unknown","d","Structural screen (B)")]:
        panel(ax,letter,title); sub=ranks.loc[ranks.candidate_set.eq(candidate)]
        ax.plot([1,27],[1,27],color=C["gray"],lw=.8,ls="--",zorder=1)
        ax.scatter(sub.observed_rank_lowest_first,sub.high_priority_rank_lowest_first,s=20,color=C["navy"],
          marker="^",edgecolor="white",linewidth=.3,zorder=3)
        ax.scatter(sub.observed_rank_lowest_first,sub.low_priority_rank_lowest_first,s=20,color=C["ochre"],
          marker="s",edgecolor="white",linewidth=.3,zorder=3)
        ax.set_xlim(.2,27.8); ax.set_ylim(.2,27.8); ax.set_xticks([1,5,10,15,20,27]); ax.set_yticks([1,5,10,15,20,27])
        ax.set_xlabel("Observed state rank"); ax.set_ylabel("Assigned state rank")
        ax.set_aspect("equal",adjustable="box"); style(ax); ax.grid(axis="y",color="#EEF0F2",linewidth=.35)
    handles=[Line2D([],[],color=C["gray"],marker="o",lw=0,markersize=4,label="Random constructions (12)"),
      Line2D([],[],color=C["navy"],marker="^",lw=0,markersize=4,label="High-local-population priority"),
      Line2D([],[],color=C["ochre"],marker="s",lw=0,markersize=4,label="Low-local-population priority")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.14),ncol=3,fontsize=6.1,columnspacing=.9,handletextpad=.35)
    fig.subplots_adjust(left=.245,right=.98,top=.94,bottom=.25,wspace=.45,hspace=.58)
    savefig(fig,out,"Figure_6",142)
    return {"source_rows":{"errors":len(errors),"ranks":len(ranks)},
      "random_points_per_candidate":{c:12 for c in candidates},
      "priority_points_per_candidate":{c:2 for c in candidates},
      "rank_points_per_candidate":{c:27 for c in candidates}}

CAPTIONS={
 "Figure_1":"Figure 1 | Population coverage across travel-time thresholds and the D1 urban–rural contrast. (a) Four overall-population coverage curves across facility definitions and travel modes; the dashed line marks 120 min. D2meas_corr denotes D2meas AND D1_p85; walking uses a separate friction surface. (b) D1 urban and rural coverage; labels give their difference at 15, 30 and 120 min.",
 "Figure_2":"Figure 2 | Constructed coverage under assignment, budget and country-composition scenarios. (a) Original central-budget assignments at 30 min; random-minus-clustered difference is 5.58 pp. (b,c) Overall-population paired random and clustered coverage over lower, central and upper budgets at 30 and 120 min. (d) Overall-population country-composition pairs at both thresholds. Panel a and panels b–d are separate constructions.",
 "Figure_3":"Figure 3 | Certified country widths and signed allocation contrasts. (a) Twenty-seven certified widths, ordered by width; country names and ISO3 codes are shown. (b) All-facility coverage versus certified width for the same countries. (c) Signed C(u=0) − C(u=1) endpoint contrast for 162 countries, ordered by contrast rank; this is not set width.",
 "Figure_4":"Figure 4 | Constructed feasible coverage endpoints across services and constraint scenarios. (a,b) Low and high constructed coverage endpoints at 30 and 120 min. Adjacent numbers show spread (pp); Brazil structural-screen endpoints combine A and B candidate sets across 17 unique configurations. Endpoints are constructed coverage, not optimized true width.",
 "Figure_5":"Figure 5 | Procedure-activity increments across baselines and validation designs. (a,b) Mean AUC change and log-loss improvement for three baseline comparisons at resident-AIHS threshold 1 across three services. Ranges show 2.5th–97.5th percentiles across 10 held-out-block repartitions, not population confidence intervals. (c) Mean log-loss improvement for the strong baseline across thresholds 1, 5, 10 and 20. (d) Strong-baseline and strong-baseline-plus-activity AUC in 2024 after training on 2023 data at threshold 1; labels show the difference to five decimal places.",
 "Figure_6":"Figure 6 | Coverage error and state ranking across candidate sets. (a,b) Each of 12 random constructions and each priority rule are shown as individual points for the full and B candidate sets. (c,d) Assigned versus observed ranks for 27 states under high- and low-local-population rules. Rank 1 denotes the lowest observed coverage; random points do not imply a confidence interval."
}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,default=DEFAULT_DATA,help="Released figure-source CSV directory.")
    ap.add_argument("--out",type=Path,default=DEFAULT_OUT,help="Staging directory for figures and notes.")
    args=ap.parse_args(); data_dir=args.data.expanduser().resolve(); out=args.out.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    exclusions={f"Figure_{i}":0 for i in range(1,7)}
    data=read_sources(data_dir)
    files=sorted(data_dir/f"{name}.csv" for name in data)
    qa={"status":"pass","backend":"python-matplotlib","figure_width_mm":183,
        "contract_excluded_observations":exclusions,
        "source_files":{p.name:{"rows":len(data[p.stem]),"sha256":file_hash(p)} for p in files},
        "figures":{},"outputs":{}}
    qa["figures"]["Figure_1"]=figure1(data,out)
    qa["figures"]["Figure_2"]=figure2(data,out)
    qa["figures"]["Figure_3"]=figure3(data,out)
    qa["figures"]["Figure_4"]=figure4(data,out)
    qa["figures"]["Figure_5"]=figure5(data,out)
    qa["figures"]["Figure_6"]=figure6(data,out)
    for i in range(1,7):
        name=f"Figure_{i}"
        for ext in ("pdf","png","svg","tiff"):
            path=out/f"{name}.{ext}"
            if not path.exists() or path.stat().st_size<1500: raise ValueError(f"Missing/undersized output: {path}")
        with Image.open(out/f"{name}.png") as png:
            png_dpi=[float(v) for v in png.info.get("dpi",(0,0))]
            if not all(abs(v-PNG_DPI)<1 for v in png_dpi):
                raise ValueError(f"{name} PNG dpi mismatch: {png_dpi}")
        with Image.open(out/f"{name}.tiff") as image:
            qa["outputs"][name]={"png_dpi":png_dpi,"pdf_bytes":(out/f"{name}.pdf").stat().st_size,
              "svg_bytes":(out/f"{name}.svg").stat().st_size,"png_bytes":(out/f"{name}.png").stat().st_size,
              "tiff_bytes":(out/f"{name}.tiff").stat().st_size,"tiff_mode":image.mode,
              "tiff_size_px":list(image.size),"tiff_dpi":[float(v) for v in image.info.get("dpi",(0,0))],
              "tiff_compression":image.info.get("compression")}
            if image.mode!="RGB": raise ValueError(f"{name} TIFF is not RGB.")
    (out/"revision_qa.json").write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding="utf-8")
    notes=["# Main-figure readability revision","",
      "All six figures use the released aggregate source tables without modifying them. Source values are preserved; display transformations are limited to percentage, rank, and improvement-oriented axis labels. No observation in the scoped panels was removed.","",
      "## Figure notes and exact English captions",""]
    for i in range(1,7):
        name=f"Figure_{i}"
        notes += [f"### {name}","",CAPTIONS[name],"",
                  "Source-row and numerical checks: "+json.dumps(qa["figures"][name],ensure_ascii=False),""]
    notes += ["## Export","","All figures are 183 mm wide. PNG is 300 dpi; TIFF is 600 dpi, RGB and LZW-compressed. PDF and SVG retain editable text.",
      "","Input row counts and SHA-256 hashes are recorded in revision_qa.json. Input tables were read-only during generation."]
    (out/"revision_notes.md").write_text("\n".join(notes)+"\n",encoding="utf-8")
    print(json.dumps({"status":"pass","output":str(out),"figures":[f"Figure_{i}" for i in range(1,7)],
                      "qa":str(out/"revision_qa.json")},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
