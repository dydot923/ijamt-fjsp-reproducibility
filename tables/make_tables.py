"""Journal tables, sourced solely from completed experiment summaries."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'replication'/'results_v2'; T=ROOT/'tables'
METHODS=['R-NSGA','H-NSGA','R-VNS','H-VNS','H-VNS-R','H-MVNS','MOEAD']
def write(name,caption,label,headers,rows,align=None,wide=False,note=''):
    cols=align or 'l'+'r'*(len(headers)-1); env='table*' if wide else 'table'
    s='\\begin{'+env+'}[!t]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'
    s+='\\setlength{\\tabcolsep}{'+('4' if wide else '3')+'pt}\n'
    s+='\\begin{tabular}{'+cols+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
    s+='\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)
    s+='\n\\bottomrule\n\\end{tabular}\n'
    if note:s+='\\par\\vspace{2pt}\\begin{minipage}{'+('0.97\\textwidth' if wide else '\\columnwidth')+'}\\footnotesize '+note+'\\end{minipage}\n'
    s+='\\end{'+env+'}\n';(T/(name+'.tex')).write_text(s,encoding='utf8')
def pm(g,c,d=4):return f'{g[c].mean():.{d}f} $\\pm$ {g[c].std():.{d}f}'
def main():
    manifest=json.loads((ROOT/'replication/data/benchmark_manifest.json').read_text(encoding='utf8'))
    rows=[['E0',20,10,58,347,'min / kWh']]
    rows += [[r['instance'],r['jobs'],r['machines'],r['operations'],r['arcs'],'原始 / 合成'] for r in manifest]
    write('datasets','工程与公开工艺实例规模','tab:data',['实例','工件','机器','工序','候选弧','时间/能耗'],rows,align='lrrrrl',note='E0为实际候选机器数；Mk能耗为无量纲合成值。四个约束情景沿用E0规模。')
    rows=[['R-NSGA','随机','关闭','关闭'],['H-NSGA','规则混合','关闭','关闭'],['R-VNS','随机','原机器邻域','关闭'],['H-VNS','规则混合','原机器邻域','关闭'],['H-VNS-R','规则混合','原机器邻域','开启'],['H-MVNS','规则混合','目标匹配','关闭'],['MOEA/D','随机','分解更新','关闭']]
    write('variants','共享表示的机制对照','tab:variants',['方法','初始化','局部/更新','重启'],rows,align='llll',note='NSGA系列共用非支配环境选择及自适应遗传框架；MOEA/D使用分解更新及固定变异概率。所有方法共用编码、解码和遗传算子类型。')
    if '--scaffold-only' in sys.argv:return
    df=pd.read_csv(OUT/'metrics.csv');main=df[df.instance=='E0'];rows=[]
    for m in METHODS:
        g=main[main.method==m];assert len(g)==30
        rows.append([m.replace('MOEAD','MOEA/D'),pm(g,'hv'),pm(g,'igdplus'),pm(g,'cmax_min',2),pm(g,'tec_min',2),pm(g,'lmax_min',2),f'{g.generations.mean():.1f}' if m!='MOEAD' else '--'])
    write('engineering','E0的集合质量与独立目标端点（30次运行）','tab:engineering',['方法','HV','IGD$^+$','$C_{max}^{min}$','$TEC^{min}$','$L_{max}^{min}$','代数'],rows,wide=True,note='数值为均值$\\pm$样本标准差；端点分别取每次终点集合中对应目标的最小值，不能拼接为一个计划。$C_{max}$、$L_{max}$单位为min，TEC沿用原数据kWh字段；MOEA/D不采用整代非支配环境选择。')
    ef=pd.read_csv(OUT/'effects.csv');names={'H effect, no VNS':'规则初始化（无VNS）','H effect, with VNS':'规则初始化（有VNS）','VNS effect, random':'VNS（随机初始化）','VNS effect, hybrid':'VNS（规则初始化）','Restart':'重启增量','H x VNS interaction':'初始化与VNS交互','H-MVNS minus H-VNS':'匹配VNS$-$原VNS','H-MVNS minus H-NSGA':'匹配VNS$-$仅规则','H-MVNS minus R-NSGA':'匹配VNS$-$随机NSGA','H-MVNS minus MOEAD':'匹配VNS$-$MOEA/D'}
    rows=[[names[r.contrast],f'{r.delta:+.5f}',f'[{r.lo:+.5f}, {r.hi:+.5f}]',f'{r.p_holm:.4g}',f'{r.wins}/{30-r.wins-r.ties}/{r.ties}'] for r in ef.itertuples()]
    write('effects','E0配对HV效应及多重检验校正','tab:effects',['对比','均值差','95\\%区间','$p_{Holm}$','正/负/零'],rows,wide=True,align='lrrrl',note='区间来自20000次配对bootstrap；检验为双侧Wilcoxon，共10项组成一族。正值表示表列对比的第一项较高。初始化与VNS交互不等同于组合方法相对仅规则方法的收益。')
    bm=df[df.instance.str.startswith('Mk')];bm_methods=['R-NSGA','H-NSGA','H-VNS','H-MVNS','MOEAD'];rows=[]
    for ins,g in bm.groupby('instance'):
        rows.append([ins]+[pm(g[g.method==m],'hv') for m in bm_methods]+[f'{g[g.method==m].igdplus.mean():.4f}' for m in ['H-NSGA','H-MVNS']])
    ranks=json.loads((OUT/'analysis.json').read_text(encoding='utf8'))['cross_instance']['mean_ranks']
    rows.append(['HV平均秩']+[f'{ranks[m]:.2f}' for m in bm_methods]+['--','--'])
    write('benchmarks','Mk01--Mk10的HV、主要方法IGD$^+$及HV跨实例排名','tab:benchmarks',['实例','R-NSGA','H-NSGA','H-VNS','H-MVNS','MOEA/D','IGD$^+_H$','IGD$^+_M$'],rows,wide=True,note='每方法10种子，五个HV列为均值$\\pm$样本标准差；最后两列为H-NSGA及H-MVNS的IGD$^+$均值。各实例使用自身共同尺度；HV平均秩由每实例种子均值计算，秩越小越好。不同实例的绝对HV不可直接比较。')
    st=pd.read_csv(OUT/'single_step.csv');rows=[]
    for f in ['Cmax','TEC','Lmax']:
        for v in ['original','matched']:
            g=st[(st.focus==f)&(st.version==v)]
            vals=g.groupby('parent_seed').relative_target.mean()*100
            rows.append([f,'原邻域' if v=='original' else '目标匹配',f'{vals.mean():+.3f}',f'{g.accepted.mean():.3f}',f'{g.duplicate.mean():.3f}'])
    write('single-step','共同母体单步机器迁移诊断','tab:single-step',['方向','候选机制','目标变化/\\%','接受率','重复率'],rows,wide=True,align='llrrr',note='每个方向/机制使用10个共同母体，各20次提案；目标变化先在每母体内求均值后汇总，负值表示目标改善。接受率包含支配与局部标量接受；重复率按母体内相同基因型提案计算，不等同于主实验实际解码计数。')
    part=df[(df.method=='H-MVNS')&df.instance.isin(['CF0','E1','CFM','E2'])];rows=[]
    for ins,desc in [('CF0','无附加约束'),('E1','换型'),('CFM','维护'),('E2','换型与维护')]:
        g=part[part.instance==ins];rows.append([ins,desc,pm(g,'cmax_min',2),pm(g,'tec_min',2),pm(g,'lmax_min',2)])
    write('constraints','同一匹配方法的物理约束因子实验（12种子）','tab:constraints',['情景','约束','$C_{max}^{min}$','$TEC^{min}$','$L_{max}^{min}$'],rows,wide=True,align='llrrr',note='四种情景使用相同430001--430012种子和E0工艺数据。维护窗口固定给定。比较绝对目标端点；每情景的三个端点仍可来自不同解。')
    tun=pd.read_csv(OUT/'intensity_metrics.csv');rows=[]
    for t,g in tun.groupby('trials'):
        rows.append([t,pm(g,'hv'),pm(g,'igdplus'),f'{g.generations.mean():.1f}',f'{100*g.local_fraction.mean():.1f}'])
    write('intensity','独立种子的局部试探强度探索（每项6种子）','tab:intensity',['每邻域试探','HV','IGD$^+$','代数','局部解码/\\%'],rows,wide=True,note='种子440001--440006与正式比较独立，4000次预算不变；局部解码比例包括四种邻域。表中探索不用于回填主实验参数或重新选择正式结果。')
if __name__=='__main__':main()
