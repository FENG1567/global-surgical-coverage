from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import layout_main_4_6 as previous

old4,old5,old6=previous.layout_4,previous.layout_5,previous.layout_6
def layout4(fig):
 old4(fig)
 a,b=fig.axes
 a.set_yticks(a.get_yticks(),[t.get_text().replace('Chole','Cholecystectomy') for t in a.get_yticklabels()])
 for leg in fig.legends:
  for t in leg.get_texts():t.set_text(t.get_text().replace('Brazil screened (A = B)','Brazil screened'))
 for ax in fig.axes:ax.set_xlabel('Population coverage (%)')

def layout5(fig):
 old5(fig)
 a,b,c,d=fig.axes
 for ax,title in [(a,'AUC increment\n≥1 resident record'),(b,'Log-loss improvement\n≥1 resident record'),(c,'Strong-baseline sensitivity'),(d,'Temporal AUC\n≥1 resident record')]:ax._left_title.set_text(title)
 fig.subplots_adjust(left=.175,right=.98,top=.87,bottom=.235,wspace=.70,hspace=.57)
 d.tick_params(axis='y',labelleft=True)
 d.set_yticks([0,1,2],['Cholecystectomy','Cesarean delivery','Appendectomy'])
 for t in list(fig.texts):
  if (t.get_gid() or '').startswith('panel_'):t.remove()
 for letter,x,y in [('A',.02,.96),('B',.54,.96),('C',.02,.55),('D',.54,.55)]:previous.shared.label(fig,letter,x,y)
 d.set_xticks([.88,.90,.92,.94,.96,.98])

def layout6(fig):
 old6(fig)
 a,b,c,d=fig.axes
 a.set_yticks(a.get_yticks(),[t.get_text().replace('B ·','Screened ·') for t in a.get_yticklabels()])
 d._left_title.set_text('Structurally screened')
previous.layout_4=layout4;previous.layout_5=layout5;previous.layout_6=layout6
if __name__=='__main__':previous.main()

