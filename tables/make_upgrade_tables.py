"""Typeset executed upgrade outputs, with sample units and metric direction."""
import sys,json
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'replication/results_v3';T=ROOT/'tables'
sys.path.insert(0,str(ROOT/'replication'))
from run_budget_study import make_data

def table(name,label,caption,headers,rows,foot='',wide=False):
    environment='table*' if wide else 'table';width=r'0.97\textwidth' if wide else r'\columnwidth'
    content=[rf'\begin{{{environment}}}[!t]',r'\centering\small',rf'\caption{{{caption}}}\label{{{label}}}',
             r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',r'\toprule',
             ' & '.join(headers)+r' \\',r'\midrule']
    content+=[' & '.join(str(x) for x in row)+r' \\' for row in rows]
    content+=[r'\bottomrule',r'\end{tabular}']
    if foot:content += [r'\par\vspace{2pt}\begin{minipage}{'+width+r'}\footnotesize '+foot+r'\end{minipage}']
    content += [rf'\end{{{environment}}}']
    (T/(name+'.tex')).write_text('\n'.join(content)+'\n',encoding='utf8')

def source_tables():
    rows=[]
    for instance in ['E0']+[f'Mk{i:02d}' for i in range(1,11)]:
        d=make_data(instance);eligible={o.machine_id for v in d.operations.values() for o in v.candidates}
        rows.append([instance,len(d.jobs),f'{len(d.machines)}/{len(eligible)}',len(d.operations),sum(len(v.candidates) for v in d.operations.values())])
    table('datasets','tab:data','工程与公开工艺输入规模',['实例','工件','机器d/q','工序','候选弧'],rows,
        'd/q为定义/资格机器数。E0时间为min，能源沿用输入kWh标记、仅累计加工弧；Mk时间为原始单位，附加能源为无量纲构造。')
    table('variants','tab:variants','新版共享框架的确认对照',['方法',r'$\ell$/局部额度','环境选择','重启'],[
        ['H-NSGA','0 / 0','等级-拥挤','关闭'],['H-MVNS-6','6 / 24','等级-拥挤','关闭'],
        ['H-MVNS-2','2 / 8','等级-拥挤','关闭'],['RP-NSGA','0 / 0','参考点','关闭']],
        '共用规则初始化、40个体、遗传表示及完整目标评价。RP仅替换环境选择；H-MVNS两版本仅改变局部试探强度。')

def stats_text(values,digits=5):
    return f'{np.mean(values):.{digits}f} '+r'$\pm$'+f' {np.std(values,ddof=1):.{digits}f}'

def main():
    source_tables();df=pd.read_csv(OUT/'metrics.csv');effects=pd.read_csv(OUT/'effects.csv')
    diagnostics=pd.read_csv(OUT/'theory-diagnostics.csv')
    rows=[]
    for _,r in diagnostics.iterrows():
        rows.append([r.scenario,int(r.moves),f'{int(r.under)}/{int(r.equal)}/{int(r.over)}',f'{r.max_abs_error:.2f}',
                     f'{r.interval_width_mean:.2f}',int(r.guaranteed_improve),int(r.guaranteed_no_strict_improve),f'{r.parent_spearman_mean:.4f}'])
    table('upgrade-diagnostics','tab:upgrade-diagnostics','共同母体单迁移的理论与代理诊断',
        ['情景','迁移数','低估/相等/高估','最大误差',r'区间宽度','保证改善','排除严格改善',r'$\bar\rho_S$'],rows,
        r'各8个固定父体；误差为$\widehat L_{max}-L^\prime_{max}$，容差$10^{-8}$。误差及区间宽度单位min。区间覆盖率均为100\%；排名相关仅将浮点负载圆整至$10^{-9}$后按父体计算再平均。两种判定均仅针对负载，不作多目标剪枝。',True)
    oracle=json.loads((OUT/'oracle/summary.json').read_text(encoding='utf8'));oruns=pd.DataFrame(json.loads((OUT/'oracle/runs.json').read_text(encoding='utf8')))
    rows=[]
    for r in oracle:
        case=r['instance'];ratios=[oruns[(oruns.instance==case)&(oruns.method==m)].hv_ratio.mean() for m in ['H-NSGA','H-MVNS-6','H-MVNS-2']]
        rows.append([case,'3×2' if case in ['T1','T2','T3'] else '2×3',r['orientations'],r['acyclic'],r['encodings'],
                     f'{len(r["exact_front"])}/{len(r["decoder_front"])}',*[f'{v:.5f}' for v in ratios]])
    table('upgrade-oracle','tab:upgrade-oracle','独立枚举与200评价小例求解',
        ['小例','工件×工序','全顺序','无环顺序','OS/MA数','真/可达PF','G体积比','L6体积比','L2体积比'],rows,
        'G为H-NSGA，L6/L2为对应匹配邻域版本。体积比为8次求解HV除以精确前沿HV的均值；全部6例真前沿与可达前沿逐目标相同。人工构造用于精度诊断，不是工厂实测。',True)
    rows=[]
    for population in ['E0','Mk']:
        data=df[(df.budget==4000)&((df.instance=='E0') if population=='E0' else (df.instance!='E0'))]
        if population=='Mk':data=data.groupby(['instance','method']).mean(numeric_only=True).reset_index()
        for m in ['H-NSGA','H-MVNS-6','H-MVNS-2','RP-NSGA']:
            g=data[data.method==m]
            rows.append([population,m,len(g),stats_text(g.hv),stats_text(g.igdplus),f'{g.front_size.mean():.1f}',f'{(g.local_fresh/g.budget*100).mean():.1f}'])
    table('upgrade-confirmation','tab:upgrade-confirmation','固定4000评价的独立确认',
        ['对象','方法','统计单位n','HV',r'IGD$^+$','前沿数',r'局部新评价/\%'],rows,
        r'E0为12配对种子（RP为6）；Mk为10实例的各方法6种子均值。均值$\pm$标准差，E0和Mk的变异来源不同。指标按输入解析界计算，绝对值不与旧探索尺度比较。',True)
    primary=effects[effects.family!='F2'];rows=[]
    for _,r in primary.iterrows():
        metric='HV' if r.metric=='hv' else r'IGD$^+$'
        rows.append([r.family,r.control,metric,int(r.n),f'{r.delta:+.6f}',f'[{r.lo:+.6f}, {r.hi:+.6f}]',f'{r.p_holm:.5f}',f'{int(r.wins)}/{int(r.ties)}/{int(r.n-r.wins-r.ties)}'])
    table('upgrade-effects','tab:upgrade-effects','L2相对各对照的预定配对效应',
        ['检验族','对照','指标','n','改善量',r'95\%区间',r'$p_{Holm}$','胜/平/负'],rows,
        r'改善量定义为HV的L2减对照、IGD$^+$的对照减L2，正值有利L2。区间为20000次配对bootstrap的逐项区间、未经同时覆盖校正。F1-Mk/F1-E0/F3分别在4/4/2项内Holm校正；E0-RP仅描述。',True)
    means=df.groupby(['instance','method','budget']).mean(numeric_only=True).reset_index();mk=means[(means.instance!='E0')&(means.budget==4000)]
    rows=[]
    for instance in sorted(mk.instance.unique()):
        g=mk[mk.instance==instance].set_index('method');cells=[]
        for m in ['H-NSGA','H-MVNS-6','H-MVNS-2','RP-NSGA']:
            h=f'{g.loc[m,"hv"]:.5f}';i=f'{g.loc[m,"igdplus"]:.5f}'
            if abs(g.loc[m,'hv']-g.hv.max())<1e-10:h=r'\textbf{'+h+'}'
            if abs(g.loc[m,'igdplus']-g.igdplus.min())<1e-10:i=r'\textbf{'+i+'}'
            cells.append(h+' / '+i)
        rows.append([instance,*cells])
    table('upgrade-benchmarks','tab:upgrade-benchmarks','Mk逐实例4000评价均值（HV / IGD$^+$）',
          ['实例','H-NSGA','H-MVNS-6','H-MVNS-2','RP-NSGA'],rows,
          r'各单元6种子均值；粗体分别标出该实例最大HV与最小IGD$^+$。逐实例胜负仅描述，确认检验采用10个实例均值。',True)
    rows=[]
    for _,r in effects[effects.family=='F2'].iterrows():
        b='梯形均值' if 'trapezoidal' in str(r.budget) else str(int(float(r.budget)))
        rows.append([b,r.control,'HV' if r.metric=='hv' else r'IGD$^+$',f'{r.delta:+.6f}',f'[{r.lo:+.6f},{r.hi:+.6f}]',f'{r.p_holm:.5f}',int(r.wins)])
    table('upgrade-budget-effects','tab:upgrade-budget-effects','独立预算响应的跨实例配对效应（F2）',
          ['预算','对照','指标','L2改善量',r'95\%区间',r'$p_{Holm}$','胜数/10'],rows,
          r'12项同属F2族；梯形均值权重为2000、4000、8000预算的1/6、1/2、1/3，不是截取8000运行的轨迹面积。正值有利L2。',True)
    rows=[]
    common=df[(df.seed<=450006)&(df.method!='RP-NSGA')]
    for b in [2000,4000,8000]:
        for m in ['H-NSGA','H-MVNS-6','H-MVNS-2']:
            g=common[(common.instance=='E0')&(common.budget==b)&(common.method==m)]
            rows.append([b,m,f'{g.hv.mean():.5f}',f'{g.igdplus.mean():.5f}',f'{(100*g.local_fresh/g.budget).mean():.1f}',f'{g.generations.mean():.1f}',f'{(100*g.reused_operations/(g.budget*58)).mean():.1f}'])
    table('upgrade-intensity','tab:upgrade-intensity','E0共同6种子的预算与计算工作量',
        ['预算','方法','HV',r'IGD$^+$',r'局部评价/\%','完成代数',r'复用工序/\%'],rows,
        '复用比例按每运行预算×58道工序为分母；局部比例按未缓存新完整编码评价计数。代数和复用工序不是完整算法墙钟速度指标。',True)
    print('Updated source tables and seven numerical tables from actual v3 results')

if __name__=='__main__':main()
