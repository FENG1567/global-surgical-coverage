from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import layout_supplement as previous
import layout_style as shared
from matplotlib.text import Text

def export(fig,out,name,height,baseline):
 if name=='Figure_S1':
  for ax in fig.axes:ax.title.set_text(ax.title.get_text().replace('Selected/total:','Facilities selected:'))
 if name=='Figure_S2':
  a,b=fig.axes;b.set_position([.29,.18,.68,.24])
  fig.legends[0].set_title('Panel B: coverage configurations')
 if name=='Figure_S3':
  a,b=fig.axes
  a.set_position([.21,.31,.31,.54]);b.set_position([.73,.31,.25,.54])
  b.tick_params(axis='y',labelleft=True)
  b._left_title.set_text('Repeated spatial-\nblock validation')
  fig.legends[0].set_title('Panel A: coverage configurations')
 if name=='Figure_S4':
  a,b=fig.axes
  a.set_yticks(a.get_yticks(),[t.get_text().replace('BRA','Brazil').replace('MEX','Mexico') for t in a.get_yticklabels()])
  for t in a.texts:
   value=t.get_text()
   try:float(value)
   except ValueError:continue
   x,y=t.get_position()
   segments=[c.get_segments()[0] for c in a.collections if hasattr(c,'get_segments') and len(c.get_segments()) and abs(c.get_segments()[0][0][1]-y)<.001]
   assert len(segments)==1
   middle=float(segments[0][:,0].mean())
   t.set_text('≥ '+value+' pp');t.set_position((middle,y-.25));t.set_ha('center');t.set_va('center')
  b._left_title.set_text('Validation\ncomparisons')
  old=fig.legends[0];handles=old.legend_handles;labels=[t.get_text() for t in old.get_texts()];old.remove()
  a.set_position([.23,.34,.30,.54]);b.set_position([.75,.34,.23,.54])
  fig.legend(handles[:2],labels[:2],loc='lower center',bbox_to_anchor=(.29,.01),ncol=1,title='Panel A: endpoints',fontsize=8.5,title_fontsize=8.5,handletextpad=.4,labelspacing=.35)
  fig.legend(handles[2:],labels[2:],loc='lower center',bbox_to_anchor=(.73,.01),ncol=1,title='Panel B: comparisons',fontsize=8.5,title_fontsize=8.5,handletextpad=.4,labelspacing=.35)
 return shared.export(fig,out,name,height,baseline)
previous.export=export
if __name__=='__main__':previous.main()
