"""Executed v3 data only; user's fine-line journal plotting conventions."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'figures'))
import make_figures as old
F=ROOT/'figures';OUT=ROOT/'replication/results_v3'
METHODS=['H-NSGA','H-MVNS-6','H-MVNS-2','RP-NSGA']
C=dict(zip(METHODS,['#0000FF','#FF0000','#009600','#FF00FF']))
MARK=dict(zip(METHODS,['o','^','x','s']))
LABEL=dict(zip(METHODS,['H-NSGA','MVNS-6','MVNS-2','RP-NSGA']))
MANIFEST=[]

def save(fig,name,source):
    old.save(fig,name)
    MANIFEST.append(dict(figure=name,source=str(source.relative_to(ROOT)).replace('\\','/'),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),real_executed=True,
        script='figures/make_upgrade_figures.py',formats=['pdf','svg','png']))

def panels(rows=1,cols=2,height=2.7):
    fig,axs=plt.subplots(rows,cols,figsize=(old.WIDTH,height),squeeze=False)
    fig.subplots_adjust(left=.09,right=.975,bottom=.18 if rows==1 else .105,
                        top=.86 if rows==1 else .91,wspace=.35,hspace=.66)
    return fig,list(axs.flat)

def thin(ax,x,y,method,yerr=None,markevery=1):
    ax.errorbar(x,y,yerr=yerr,color=C[method],marker=MARK[method],mfc='none',mew=.6,
                ms=4,lw=.6,elinewidth=.35,capsize=1.2,label=LABEL[method],markevery=markevery)

def validation():
    raw=[json.loads(line) for line in (ROOT/'qa/theory-upgrade-evidence/single-move-raw.jsonl').read_text(encoding='utf8').splitlines()]
    timing=pd.read_csv(OUT/'prefix-timing.csv');batches=pd.read_csv(OUT/'prefix-timing-batches.csv')
    fig,axs=panels(2,2,4.65)
    values=[[r['proxy_error'] for r in raw if r['scenario']==s] for s in ['E0','E1','E2']]
    old.blue_boxplot(axs[0],values);axs[0].set(xticks=[1,2,3],xticklabels=['E0','E1','E2'],ylabel=r'$\widehat L_{max}-L^\prime_{max}$ / min')
    axs[0].axhline(0,color='gray',lw=.4);old.panel(axs[0],'a','准备负载代理误差')
    values=[[max(r['upper'].values())-max(r['lower'].values()) for r in raw if r['scenario']==s] for s in ['E0','E1','E2']]
    old.blue_boxplot(axs[1],values);axs[1].set(xticks=[1,2,3],xticklabels=['E0','E1','E2'],ylabel=r'$UB_L^\prime-LB_L^\prime$ / min')
    old.panel(axs[1],'b','可靠区间宽度')
    timing['bin']=pd.cut(timing.prefix_fraction,[0,.25,.5,.75,1],include_lowest=True,labels=False)
    paired=timing.groupby(['instance','seed','trial','bin'],observed=True)[['full_ns','prefix_ns']].sum().reset_index()
    paired['ratio']=paired.prefix_ns/paired.full_ns
    old.blue_boxplot(axs[2],[paired[paired.bin==i].ratio for i in range(4)])
    axs[2].set(xticks=range(1,5),xticklabels=['0–.25','.25–.50','.50–.75','.75–1'],xlabel=r'Prefix fraction $c/n$',ylabel='Prefix / full time')
    axs[2].axhline(1,color='gray',ls='--',lw=.45);old.panel(axs[2],'c','同候选交错串行评估')
    for s,m in zip(['E0','E1','E2'],METHODS):
        g=batches[batches.instance==s]
        thin(axs[3],range(1,4),g.ratio,m)
    axs[3].set(xlim=(.8,3.2),xticks=[1,2,3],xlabel='Fixed parent index',ylabel='Batch time ratio')
    axs[3].axhline(1,color='gray',ls='--',lw=.45)
    handles=[Line2D([],[],color=C[m],marker=MARK[m],mfc='none',lw=.6,label=s) for s,m in zip(['E0','E1','E2'],METHODS)]
    axs[3].legend(handles=handles,loc='best',fontsize=8);old.panel(axs[3],'d','9个母体批次的成本比')
    save(fig,'fig3_upgrade_validation',OUT/'prefix-timing.csv')

def oracle():
    df=pd.DataFrame(json.loads((OUT/'oracle/runs.json').read_text(encoding='utf8')))
    fig,axs=panels(height=2.75)
    for ax,col,title,yl in zip(axs,['hv_ratio','exact_coverage'],['相对精确前沿的体积','精确非支配目标覆盖'],['HV / exact HV','Exact-front coverage']):
        lows=[]
        for m in METHODS[:3]:
            g=df[df.method==m].groupby('instance')[col]
            thin(ax,range(1,7),g.mean(),m,g.std())
            lows.extend((g.mean()-g.std()).tolist())
        ax.set(xticks=range(1,7),xticklabels=[f'T{i}' for i in range(1,7)],ylabel=yl,ylim=(-.05,1.08));old.panel(ax,'a' if col=='hv_ratio' else 'b',title)
        if col=='hv_ratio':
            ax.set_ylim(np.floor((min(lows)-.004)*1000)/1000,1.005)
    axs[0].legend(loc='lower left',fontsize=7.9)
    save(fig,'fig4_upgrade_oracle',OUT/'oracle/runs.json')

def confirm(df):
    fig,axs=panels(2,2,4.40)
    means=df[(df.instance!='E0')&(df.budget==4000)].groupby(['instance','method'])[['hv','igdplus']].mean().reset_index()
    for i,(population,data) in enumerate([('E0',df[(df.instance=='E0')&(df.budget==4000)]),('Mk',means)]):
        for j,col in enumerate(['hv','igdplus']):
            ax=axs[i*2+j];old.blue_boxplot(ax,[data[data.method==m][col] for m in METHODS])
            ax.set(xticks=range(1,5),xticklabels=[LABEL[m] for m in METHODS],ylabel='Hypervolume' if col=='hv' else r'IGD$^+$')
            ax.tick_params(axis='x',labelsize=8)
            old.panel(ax,'abcd'[i*2+j],f'{population}，4000评价')
    save(fig,'fig5_upgrade_confirmation',OUT/'metrics.csv')

def budgets(df):
    fig,axs=panels(2,2,4.45);source=[]
    common=df[(df.seed<=450006)&(df.method!='RP-NSGA')]
    for i,population in enumerate(['E0','Mk']):
        data=common[common.instance=='E0'] if i==0 else common[common.instance!='E0'].groupby(['instance','method','budget']).mean(numeric_only=True).reset_index()
        if i==1:
            baseline=data[data.method=='H-NSGA'][['instance','budget','hv','igdplus']].rename(columns={'hv':'base_hv','igdplus':'base_igdplus'})
            data=data.merge(baseline,on=['instance','budget'],validate='many_to_one')
            data['hv']=data.hv-data.base_hv
            data['igdplus']=data.base_igdplus-data.igdplus
        for j,col in enumerate(['hv','igdplus']):
            ax=axs[i*2+j]
            for m in METHODS[:3]:
                g=data[data.method==m].groupby('budget')[col];mean=g.mean();se=g.std()/np.sqrt(g.count())
                thin(ax,mean.index/1000,mean,m,1.96*se)
                source.extend(dict(population=population,method=m,metric=col,budget=int(b),mean=float(mean[b]),se=float(se[b]),units=int(g.count()[b]),comparison='absolute' if i==0 else 'paired benefit vs H-NSGA') for b in mean.index)
            ylabel=('Hypervolume' if col=='hv' else r'IGD$^+$') if i==0 else (r'$\Delta HV$ vs H-NSGA' if col=='hv' else r'IGD$^+$ benefit vs H-NSGA')
            ax.set(xticks=[2,4,8],xlim=(1.6,8.4),xlabel='Independent budget / 1000',ylabel=ylabel)
            if i==1:ax.axhline(0,color='gray',lw=.4)
            old.panel(ax,'abcd'[i*2+j],f'{population}独立预算响应')
    axs[0].legend(loc='best',fontsize=7.7)
    pd.DataFrame(source).to_csv(F/'data/upgrade-budget-plot.csv',index=False,encoding='utf-8-sig')
    save(fig,'fig6_upgrade_budget',OUT/'metrics.csv')

def benchmarks(df):
    data=df[(df.instance!='E0')&(df.budget==4000)].groupby(['instance','method'])[['hv','igdplus']].mean().reset_index()
    fig,axs=panels(2,1,4.20);fig.subplots_adjust(hspace=.62,bottom=.10)
    for ax,col,letter in zip(axs,['hv','igdplus'],['a','b']):
        for m in METHODS:
            g=data[data.method==m].sort_values('instance');thin(ax,range(1,11),g[col],m)
        ax.set(xticks=range(1,11),xticklabels=[f'Mk{i:02d}' for i in range(1,11)],xlim=(.75,10.25),ylabel='Hypervolume' if col=='hv' else r'IGD$^+$')
        old.panel(ax,letter,'固定4000评价的实例均值')
    axs[0].legend(loc='best',ncol=4,fontsize=8.0)
    save(fig,'fig7_upgrade_benchmarks',OUT/'metrics.csv')

def tradeoffs():
    fig,axs=panels(1,3,2.80);fig.subplots_adjust(left=.07,wspace=.39,bottom=.23,top=.79)
    labels=[r'$C_{max}$ / min','TEC / input kWh',r'$L_{max}$ / min']
    data=[]
    for m in METHODS:
        path=OUT/'completed'/f'E0_{m}_450001_4000.json';r=json.loads(path.read_text(encoding='utf8'));data.append(r)
        a=np.unique(np.asarray(r['front']),axis=0)
        for ax,(x,y) in zip(axs,[(0,1),(0,2),(1,2)]):
            b=a[:,[x,y]];keep=[p for p in b if not any(np.all(q<=p+1e-9) and np.any(q<p-1e-9) for q in b)]
            b=np.unique(keep,axis=0);b=b[np.argsort(b[:,0])]
            thin(ax,b[:,0],b[:,1],m,markevery=max(1,len(b)//12))
            ax.set(xlabel=labels[x],ylabel=labels[y]);ax.tick_params(labelsize=7.6)
    for ax,letter in zip(axs,'abc'):old.panel(ax,letter,'同种子终点投影')
    fig.legend(*axs[0].get_legend_handles_labels(),ncol=4,loc='upper center',bbox_to_anchor=(.5,.995),fontsize=8)
    (F/'data/upgrade-tradeoff-runs.json').write_text(json.dumps(data),encoding='utf8')
    save(fig,'fig8_upgrade_tradeoffs',F/'data/upgrade-tradeoff-runs.json')

def schedule():
    path=OUT/'completed/E0_H-MVNS-2_450001_4000.json';r=json.loads(path.read_text(encoding='utf8'));p=r['representative']
    entries=p['entries'];machines=[f'M{i:02d}' for i in range(1,11)]
    fig=plt.figure(figsize=(old.WIDTH,4.95));ax=fig.add_axes([.09,.31,.88,.58])
    stats=old.p1_gantt(ax,entries,machines);old.panel(ax,'a','同一折中计划的加工排程')
    ax.set_xlabel('Time / min')
    b=fig.add_axes([.09,.105,.88,.105]);loads=[sum(e['end']-e['setup_start'] for e in entries if e['machine_id']==m) for m in machines]
    b.bar(range(1,11),loads,width=.58,color=old.BOX_BLUE,edgecolor='gray',linewidth=.45)
    b.set(xticks=range(1,11),xticklabels=machines,ylabel='Load / min',yticks=[0,200,400],ylim=(0,450));b.tick_params(labelsize=8)
    old.panel(b,'b','准备与加工累计占用')
    fig.text(.09,.962,'H-MVNS-2；seed=450001；B=4000',fontsize=8.3)
    fig.text(.97,.962,rf'$C_{{max}}={p["objectives"][0]:g}$  $TEC={p["objectives"][1]:g}$  $L_{{max}}={p["objectives"][2]:g}$',ha='right',fontsize=8.3)
    (F/'data/upgrade-schedule-audit.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    save(fig,'fig9_upgrade_schedule',path)

def main():
    df=pd.read_csv(OUT/'metrics.csv');validation();oracle();confirm(df);budgets(df);benchmarks(df);tradeoffs();schedule()
    (F/'upgrade-data-manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding='utf8')
    (ROOT/'qa/upgrade-figure-render-audit.json').write_text(json.dumps(old.REPORT,ensure_ascii=False,indent=2),encoding='utf8')
    print('Exported',len(MANIFEST),'new data figures in PDF/SVG/450dpi PNG')

if __name__=='__main__':main()
