import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/root/code/tokenizers-perfetto-results')
ss=json.loads((root/'analysis-all.json').read_text())
cases=['en-whitespace','zh-whitespace','en-bytelevel','zh-bytelevel','code-bytelevel']
bins=['<1K','1K-4K','4K-16K','16K-64K','64K+']
fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
for ax,phase in zip(axs,['prepare','commit']):
 bottom=[0.]*len(cases)
 for label,color in zip(bins,['#173f5f','#20639b','#3caea3','#f6d55c','#ed553b']):
  vals=[]
  for case in cases:
   runs=[s for s in ss if s['case']==case and s['arm']=='coarse' and s['rep']<=3]
   vals.append(100*sum(s[phase]['by_size'][label]['lost_ns'] for s in runs)/sum(s[phase]['lost_ns'] for s in runs))
  ax.bar(range(len(cases)),vals,bottom=bottom,label=label,color=color)
  bottom=[a+b for a,b in zip(bottom,vals)]
 ax.set_xticks(range(len(cases)),['EN / WS','ZH / WS','EN / BL','ZH / BL','Code / BL'],rotation=25)
 ax.set_title(phase.capitalize());ax.set_ylim(0,100)
axs[0].set_ylabel('Share of cumulative task-occupancy gap (%)')
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=5,title='Raw candidate positions per batch')
fig.suptitle('Different loss distributions: 4 workers, 3 traces per workload')
fig.tight_layout(rect=(0,.18,1,.94));fig.savefig(root/'loss-by-batch-size.png',dpi=180)
