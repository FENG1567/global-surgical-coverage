"""Shared typography, editable exports and deterministic artist-data checks.
SPDX-License-Identifier: MIT
"""
from pathlib import Path
import os,json,hashlib,itertools
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
FONT_DIR=Path(os.environ.get('FIGURE_FONT_DIR',str(Path('C:/Windows/Fonts') if Path('C:/Windows/Fonts').exists() else ROOT/'assets/fonts')))
BOLD=FONT_DIR/'arialbd.ttf';REGULAR=FONT_DIR/'arial.ttf'
for p in [REGULAR,BOLD]:
    if not p.exists():raise FileNotFoundError(f'Required Arial font missing: {p.name}; configure FIGURE_FONT_DIR')
    font_manager.fontManager.addfont(str(p))
MM=1/25.4;WIDTH_MM=183;DPI=600
mpl.rcParams.update({'font.family':'Arial','font.weight':'bold','font.size':9.0,
    'axes.labelsize':9.5,'axes.labelweight':'bold','axes.titlesize':10.5,'axes.titleweight':'bold',
    'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'legend.frameon':False,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.9})

def bold_all(fig):
    for t in fig.findobj(Text):
        if not t.get_text():continue
        label=t.get_gid() or ''
        size=18.0 if label.startswith('panel_') else max(8.5,float(t.get_fontsize()))
        t.set_fontproperties(font_manager.FontProperties(fname=str(BOLD),size=size,weight='bold'))
    return fig

def label(fig,letter,x,y):
    return fig.text(x,y,letter.upper(),fontsize=18,fontweight='bold',fontfamily='Arial',
                    ha='left',va='top',color='#111111',gid='panel_'+letter.upper())

def remove_old_panel_labels(fig):
    for t in list(fig.findobj(Text)):
        if t.get_text().strip() in list('abcdABCD'):
            try:t.remove()
            except NotImplementedError:t.set_visible(False)

def signature(fig):
    """Coordinate data only: typography and axes-box layout can change freely."""
    h=hashlib.sha256();counts={'lines':0,'collections':0,'patches':0}
    def add(a):
        a=np.asarray(a,dtype=np.float64)
        h.update(str(a.shape).encode());h.update(a.tobytes())
    for ax in fig.axes:
        for item in ax.lines:
            add(item.get_xdata());add(item.get_ydata());counts['lines']+=1
        for item in ax.collections:
            add(item.get_offsets())
            if hasattr(item,'get_segments'):
                for s in item.get_segments():add(s)
            counts['collections']+=1
        for item in ax.patches:
            if hasattr(item,'get_x'):
                add([item.get_x(),item.get_y(),item.get_width(),item.get_height()])
            counts['patches']+=1
    return {'sha256':h.hexdigest(),**counts}

def check_text(fig):
    fig.canvas.draw();r=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    visible=[];outside=[];overlap=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():continue
        bb=t.get_window_extent(r)
        if bb.width<=0 or bb.height<=0:continue
        entry={'text':t.get_text(),'bbox':[round(v,2) for v in bb.extents],
               'size':float(t.get_fontsize()),'gid':t.get_gid()}
        visible.append((t,bb,entry))
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1>w+0.5 or bb.y1>h+0.5:outside.append(entry)
    for (ta,a,ea),(tb,b,eb) in itertools.combinations(visible,2):
        dx=min(a.x1,b.x1)-max(a.x0,b.x0);dy=min(a.y1,b.y1)-max(a.y0,b.y0)
        if dx>1.0 and dy>1.0:
            frac=dx*dy/min(a.width*a.height,b.width*b.height)
            if frac>.12:overlap.append({'a':ea['text'],'b':eb['text'],'fraction':round(frac,3)})
    return {'outside':outside,'overlaps':overlap,'text_count':len(visible),
            'minimum_fontsize':min(e['size'] for _,_,e in visible),'panel_labels':[e for _,_,e in visible if (e['gid'] or '').startswith('panel_')]}

def export(fig,out,name,height_mm,baseline_signature=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig.set_size_inches(WIDTH_MM*MM,height_mm*MM,forward=True)
    bold_all(fig)
    now=signature(fig)
    if baseline_signature is not None:assert now==baseline_signature,(name,'scientific artist coordinates changed',baseline_signature,now)
    audit=check_text(fig)
    audit.update({'figure':name,'width_mm':WIDTH_MM,'height_mm':height_mm,'artist_signature':now,
                  'font_family':'Arial','all_text_bold':True,'panel_label_points':18,'tiff_dpi':DPI})
    fig.savefig(out/f'{name}.pdf',facecolor='white')
    fig.savefig(out/f'{name}.svg',facecolor='white')
    fig.savefig(out/f'{name}.png',dpi=300,facecolor='white')
    fig.savefig(out/f'{name}.tiff',dpi=DPI,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    tif=out/f'{name}.tiff';temp=out/f'.{name}.rgb.tiff'
    with Image.open(tif) as im:im.convert('RGB').save(temp,format='TIFF',compression='tiff_lzw',dpi=(DPI,DPI))
    temp.replace(tif)
    (out/f'{name}_layout_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    plt.close(fig)
    print('EXPORTED',name,'outside',len(audit['outside']),'text_overlaps',len(audit['overlaps']),flush=True)
    return audit

