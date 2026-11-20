"""Code-authored journal figures, revision 2026-10-08.

Fixed 180 mm canvases, editable vector text, embedded PDF fonts, and 450 dpi
rasters. All numerical plots use frozen, actually executed v2 experiments.
"""
from pathlib import Path
import json
import sys
import hashlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch, Polygon, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'replication' / 'results_v2'
F = ROOT / 'figures'
sys.path.insert(0, str(ROOT / 'replication'))
from fjsp.dataset_io import records_to_problem
from fjsp.decoder import decode
from fjsp.objectives import static_objectives
from run_stress import check_schedule

INK = '#000000'
GRAY = '#808080'
LIGHT = '#B8B8B8'
BLUE = '#0000FF'
TEAL = '#009600'
RUST = '#FF0000'
C = {'R-NSGA': '#FF00FF', 'H-NSGA': BLUE, 'R-VNS': '#00A0C0',
     'H-VNS': RUST, 'H-VNS-R': '#8050C0', 'H-MVNS': TEAL, 'MOEAD': INK}
MARKERS = {'R-NSGA':'s','H-NSGA':'o','R-VNS':'D','H-VNS':'^',
           'H-VNS-R':'v','H-MVNS':'x','MOEAD':'*'}
LABEL = {k: ('MOEA/D' if k == 'MOEAD' else k) for k in C}
JCOL = ['#D3D0ED','#F6C4AC','#EAF4D7','#F1CCE8','#91BFB6',
        '#B1DAB8','#8ED1E0','#CFB69D','#C3A1E2','#B3CDF1',
        '#D7CDBA','#E0D9F2','#BEDCB3','#EDD5B9','#D9BBDF',
        '#B7D8D4','#DADBE0','#EDBDC4','#CED5B1','#AEC5DA']
BOX_BLUE = '#7DA4CF'
WIDTH = 180 / 25.4
REPORT = []
plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun'], 'font.size': 9.1,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.labelsize': 9.4, 'axes.titlesize': 9.4, 'legend.fontsize': 8.5,
    'axes.linewidth': .45, 'axes.edgecolor': INK, 'lines.linewidth': .65,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': .4, 'ytick.major.width': .4,
    'axes.spines.top': True, 'axes.spines.right': True,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'mathtext.fontset': 'stix', 'savefig.dpi': 450,
    'axes.unicode_minus': True, 'legend.frameon': False,
    'legend.handlelength': 1.8, 'legend.columnspacing': 1.4,
})


def save(fig, name):
    """Keep the canvas dimensions stable and audit text at final placement."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    tick_overlaps = []
    off_axis_ticks = set()
    for ax in fig.axes:
        for axis, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(lim)
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if tick.get_loc() < lo-1e-8 or tick.get_loc() > hi+1e-8:
                    off_axis_ticks.update([id(tick.label1), id(tick.label2)])
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text() or id(t) in off_axis_ticks:
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        b = t.get_window_extent(renderer)
        if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or
                                     b.x1 > fig.bbox.width + 1 or b.y1 > fig.bbox.height + 1):
            outside.append(t.get_text())
    assert not outside, (name, 'text outside canvas', outside)
    for ax in fig.axes:
        if not ax.axison:continue
        lo,hi=sorted(ax.get_xlim())
        ticks=[t for t in ax.xaxis.get_major_ticks() if lo-1e-8<=t.get_loc()<=hi+1e-8 and t.label1.get_visible()]
        for first,second in zip(ticks,ticks[1:]):
            if first.label1.get_window_extent(renderer).overlaps(second.label1.get_window_extent(renderer)):
                tick_overlaps.append([first.label1.get_text(),second.label1.get_text()])
    assert not tick_overlaps,(name,'overlapping x tick labels',tick_overlaps)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(F / (name + '.' + ext), dpi=450, facecolor='white')
    REPORT.append(dict(figure=name, width_mm=round(fig.get_figwidth()*25.4, 2),
                       height_mm=round(fig.get_figheight()*25.4, 2), dpi=450,
                       text_outside_canvas=outside,
                       x_tick_label_overlaps=tick_overlaps,
                       gantt_checks=getattr(fig,'_gantt_checks',None),
                       sha256=hashlib.sha256((F/(name+'.pdf')).read_bytes()).hexdigest()))
    plt.close(fig)


def panel(ax, letter, title):
    ax.text(0, 1.065, f'({letter})', transform=ax.transAxes,
            fontsize=9.4, va='bottom')
    ax.annotate(title, (0, 1.065), xycoords=ax.transAxes,
                xytext=(24, 0), textcoords='offset points',
                fontsize=9.5, va='bottom', annotation_clip=False)


def grid(ax, axis='y'):
    ax.grid(False)
    ax.set_axisbelow(True)


def blue_boxplot(ax, values, positions=None, width=.46):
    """P2: one blue fill, gray whiskers/median, asterisk outliers."""
    return ax.boxplot(values,positions=positions,widths=width,patch_artist=True,
        showfliers=True,boxprops=dict(facecolor=BOX_BLUE,edgecolor=GRAY,lw=.55),
        medianprops=dict(color='#666666',lw=.55),
        whiskerprops=dict(color=GRAY,lw=.45),capprops=dict(color=GRAY,lw=.45),
        flierprops=dict(marker='*',markersize=3,markeredgewidth=.35,
                        markerfacecolor='none',markeredgecolor='#666666'))


def p1_gantt(ax, entries, machines, *, illustrative=False):
    """P1 bars with real task boundaries; close labels occupy two lanes."""
    mi={m:i for i,m in enumerate(machines)}
    for e in entries:
        y=mi[e['machine_id']];job=int(''.join(c for c in e['job_id'] if c.isdigit()))
        if e.get('setup_time',0)>0:
            ax.barh(y,e['setup_time'],left=e['setup_start'],height=.34,
                    color='#E4E4E4',edgecolor=INK,lw=.4,zorder=2)
        ax.barh(y,e['processing_time'],left=e['start'],height=.34,
                color=JCOL[job-1],edgecolor=INK,lw=.4,zorder=3)
        label=rf'$O_{{{job}{e["op_index"]}}}$' if illustrative else str(job)
        ax.text((e['start']+e['end'])/2,y,label,ha='center',va='center',
                fontsize=8.4 if illustrative else 7.6,color=INK,zorder=4)
    last=max(e['end'] for e in entries)
    ax.set(yticks=range(len(machines)),yticklabels=[rf'$M_{{{int(m[1:]):02d}}}$' for m in machines],
           ylim=(len(machines)-.1,-.45),xlim=(0,last*1.018))
    ax.tick_params(axis='y',length=2,labelsize=9)
    for spine in ['top','right']:ax.spines[spine].set_visible(False)
    ax.grid(False)
    ax.axvline(last,color=GRAY,ls=(0,(4,3)),lw=.5)
    ax.figure.canvas.draw();renderer=ax.figure.canvas.get_renderer()
    font=matplotlib.font_manager.FontProperties(family='Times New Roman',size=6.6)
    labels=[]
    for machine,y in mi.items():
        # Identical shared boundaries have a single annotation.
        times=sorted({round(e[k],6) for e in entries if e['machine_id']==machine for k in ['start','end']})
        occupied=[-float('inf'),-float('inf')]
        for time in times:
            label=f'{time:g}';width=renderer.get_text_width_height_descent(label,font,False)[0]
            pixel=ax.transData.transform((time,y))[0]
            lane=next((k for k,end in enumerate(occupied) if pixel-width/2>end+1.5),None)
            assert lane is not None,(machine,time,'two boundary lanes exhausted')
            occupied[lane]=pixel+width/2
            ax.text(time,y+.31+lane*.27,label,ha='center',va='center',fontsize=6.6)
            labels.append(dict(machine=machine,time=time,lane=lane,label=label))
    return dict(operations=len(entries),machines=len(machines),boundary_labels=labels,
                label_lanes=2,job_colors=len({e['job_id'] for e in entries}),
                job_label='job number' if not illustrative else 'operation')


def arrow(ax, start, end, color=GRAY, lw=.8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                 mutation_scale=8, lw=lw, color=color, shrinkA=0, shrinkB=0))


def rect(ax, xy, w, h, text='', fc='white', ec=GRAY, fs=8.8, color=INK):
    ax.add_patch(Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=.65))
    if text:
        ax.text(xy[0]+w/2, xy[1]+h/2, text, va='center', ha='center',
                fontsize=fs, color=color)


def dimension(ax, x0, x1, y, label, *, fs=8.0):
    """Thin, two-headed engineering dimension with extension lines."""
    ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle='<->',
                 mutation_scale=5,lw=.5,color=INK,shrinkA=0,shrinkB=0))
    for x in [x0,x1]:ax.plot([x,x],[y-.08,y+.08],color=INK,lw=.45)
    ax.text((x0+x1)/2,y+.04,label,ha='center',va='bottom',fontsize=fs)


def orth_arrow(ax, points, label=None, label_xy=None, *, dashed=False):
    """Orthogonal flow connector; only the final segment carries an arrow."""
    for first,last in zip(points[:-2],points[1:-1]):
        ax.plot([first[0],last[0]],[first[1],last[1]],color=INK,lw=.55,
                ls='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                 mutation_scale=6.5,color=INK,lw=.55,shrinkA=0,shrinkB=0,
                 linestyle='--' if dashed else '-'))
    if label:ax.text(*label_xy,label,fontsize=7.6,ha='center',va='center')


def flow_node(ax, x, y, w, h, label, *, kind='process', fs=8.0):
    """Conventional white process, decision, or terminator node."""
    if kind=='decision':
        shape=Polygon([(x,y+h/2),(x+w/2,y),(x,y-h/2),(x-w/2,y)],
                      closed=True,facecolor='white',edgecolor=INK,lw=.6)
    elif kind=='terminal':
        shape=FancyBboxPatch((x-w/2,y-h/2),w,h,
                boxstyle='round,pad=0,rounding_size=2.0',
                facecolor='white',edgecolor=INK,lw=.6)
    else:shape=Rectangle((x-w/2,y-h/2),w,h,facecolor='white',edgecolor=INK,lw=.6)
    ax.add_patch(shape)
    text=ax.text(x,y,label,ha='center',va='center',fontsize=fs,linespacing=1.18)
    if kind!='decision':
        ax.figure.canvas.draw()
        tb=text.get_window_extent(ax.figure.canvas.get_renderer())
        sb=shape.get_window_extent(ax.figure.canvas.get_renderer())
        assert sb.contains(tb.x0,tb.y0) and sb.contains(tb.x1,tb.y1),(label,'text does not fit flow node')
    return (x,y,w,h)


def charts(n, height=3.0, ratios=None, bottom=.20, wspace=.40):
    fig = plt.figure(figsize=(WIDTH, height))
    gs = fig.add_gridspec(1, n, left=.09, right=.975, top=.84,
                         bottom=bottom, wspace=wspace, width_ratios=ratios)
    return fig, [fig.add_subplot(gs[0, k]) for k in range(n)]


def toy():
    spec = [('J1', 1, [(1,4,5),(2,3,6)]), ('J1',2,[(2,3,4),(3,4,3)]),
            ('J2',1,[(1,2,3),(3,3,2)]), ('J2',2,[(2,4,3)]),
            ('J3',1,[(2,2,4),(3,4,2)]), ('J3',2,[(1,3,2),(3,2,3)])]
    ops = [dict(job_id=j, operation_id=str(i), op_index=i,
                machine_id=f'M{k:02d}', processing_time=p, setup_time=1,
                running_energy=e) for j,i,candidates in spec for k,p,e in candidates]
    data = records_to_problem(ops,
        [dict(job_id=f'J{i}', family=f'F{i}', release_time=0) for i in range(1,4)],
        [dict(machine_id=f'M{i:02d}') for i in range(1,4)])
    data.machines['M02'].maintenance_windows = [(5,8)]
    seq = ['J1','J2','J3','J1','J2','J3']
    ma = dict(zip(data.operation_ids,['M01','M03','M01','M02','M03','M01']))
    schedule = decode(data,seq,ma)
    obj = static_objectives(data,schedule)
    assert not check_schedule(data,schedule,obj)
    assert np.allclose(obj,[13,18,12])
    # The candidate changes suffix OS/MA, keeping a continuous three-operation prefix.
    from fjsp.decoder import _sequence_to_operation_ids
    from fjsp.nsga_vns import Individual
    from prefix_evaluator import prefix_cut, make_evaluator
    candidate_seq = ['J1','J2','J3','J3','J1','J2']
    candidate_ma = ma.copy();candidate_ma['J1-O2'] = 'M02'
    parent = Individual(seq, ma)
    candidate = Individual(candidate_seq, candidate_ma)
    cut, candidate_ids = prefix_cut(data,parent,candidate)
    parent_ids = _sequence_to_operation_ids(data,seq)
    assert cut == 3
    candidate_full = decode(data,candidate_seq,candidate_ma)
    # Exercise the production common-prefix evaluator, including its full path.
    counters = dict(full_calls=0,prefix_calls=0,reused_operations=0,
                    reinserted_operations=0)
    evaluator = make_evaluator(counters)
    evaluator(parent,data)
    assert parent.schedule.entries == schedule.entries
    candidate._base = parent
    evaluator(candidate,data)
    candidate_reuse = candidate.schedule
    assert counters == dict(full_calls=1,prefix_calls=1,reused_operations=3,
                            reinserted_operations=9)
    assert candidate_full.entries == candidate_reuse.entries
    assert candidate_full.feasible == candidate_reuse.feasible
    candidate_obj = static_objectives(data,candidate_full)
    assert candidate_obj == static_objectives(data,candidate_reuse)
    assert not check_schedule(data,candidate_full,candidate_obj)
    prefix_payload = dict(candidate_order_sequence=candidate_seq,
        parent_operation_sequence=parent_ids,candidate_operation_sequence=candidate_ids,
        candidate_machine_assignment=candidate_ma,cut=cut,
        fixed_operation_ids=candidate_ids[:cut],active_operation_ids=candidate_ids[cut:],
        candidate_objectives=candidate_obj,
        candidate_entries=[vars(e) for e in candidate_full.entries.values()],
        full_reuse_entries_equal=True,full_reuse_objectives_equal=True,
        full_reuse_feasible_equal=True,evaluator_counters=counters,
        evaluator_source='replication/prefix_evaluator.py::make_evaluator')
    payload = dict(order_sequence=seq,machine_assignment=ma,objectives=obj,
                   prefix_reuse=prefix_payload,
                   entries=[vars(e) for e in schedule.entries.values()],
                   note='Explanatory three-job fixture; not a performance experiment')
    (F/'data').mkdir(exist_ok=True)
    (F/'data/encoding_fixture.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
    # Technical schematic palette from the user's insertion-diagram references.
    edge='#285D70';cyan='#C8E7EC';moving='#B74840';pink='#F3DEDB'
    fig = plt.figure(figsize=(WIDTH,6.60))
    a = fig.add_axes([.065,.765,.425,.17]); a.axis('off'); a.set(xlim=(0,1),ylim=(0,1))
    b = fig.add_axes([.55,.765,.425,.17]); b.axis('off'); b.set(xlim=(0,1),ylim=(0,1))
    panel(a,'a','工艺路径与机器资格'); panel(b,'b','OS / MA编码')
    for i,j in enumerate(['J1','J2','J3']):
        y = .86-i*.30
        a.text(0,y,j,fontsize=9.1,va='center',color=INK)
        for k in [1,2]:
            x = .14+(k-1)*.44
            rect(a,(x,y-.045),.33,.11,rf'$O_{{{i+1}{k}}}$',fc=cyan,ec=edge,fs=9)
            candidates = next(c for job,op,c in spec if job==j and op==k)
            desc = '  '.join(f'M{m}({p},{e})' for m,p,e in candidates)
            a.text(x+.165,y-.105,desc,ha='center',va='top',fontsize=7.8)
        arrow(a,(.47,y+.01),(.58,y+.01),color=INK,lw=.55)
    a.text(.14,-.20,'候选机器：Mk(p,e)；准备时间均为1',fontsize=7.9)
    for row,(name,values,y) in enumerate([('OS',seq,.73),('MA',list(ma.values()),.32)]):
        b.text(0,y+.055,name,va='center',fontsize=9)
        for i,v in enumerate(values):
            x = .15+i*.14
            highlight=(i==(3 if row==0 else 1))
            rect(b,(x,y),.14,.13,v.replace('M0','M'),ec=moving if highlight else edge,
                 fc=pink if highlight else 'white',fs=8.7)
            op = (rf'$O_{{{seq[i][-1]}{seq[:i+1].count(seq[i])}}}$'
                  if row==0 else rf'$O_{{{data.operation_ids[i][1]}{data.operation_ids[i][-1]}}}$')
            b.text(x+.07,y-.05,op,ha='center',va='top',fontsize=8)
    b.text(.15,.02,'出现次数映射OS；固定工序顺序索引MA',fontsize=8.1)
    # Read positions, rather than time-axis positions, determine the common prefix.
    prefix_ax=fig.add_axes([.065,.545,.91,.145])
    prefix_ax.axis('off');prefix_ax.set(xlim=(0,1),ylim=(0,1))
    panel(prefix_ax,'c','共同读取前缀与精确后缀重排')
    x0=.14;cell=.14
    for row,(label,ids,y) in enumerate([
            ('母体',parent_ids,.64),('候选',candidate_ids,.23)]):
        prefix_ax.text(.015,y+.11,label,ha='left',va='center',fontsize=8.6)
        for i,oid in enumerate(ids):
            x=x0+i*cell;shared=i<cut
            op=rf'$O_{{{oid[1]}{oid[-1]}}}$'
            rect(prefix_ax,(x,y),cell,.22,op+' / '+(ma if row==0 else candidate_ma)[oid].replace('M0','M'),
                 fc=pink if row==1 and ma[oid]!=candidate_ma[oid] else (cyan if shared else 'white'),
                 ec=moving if row==1 and ma[oid]!=candidate_ma[oid] else edge,fs=8.6)
    cut_x=x0+cut*cell
    prefix_ax.plot([cut_x,cut_x],[.19,.92],color=INK,lw=.65,ls=(0,(3,2)))
    prefix_ax.text(cut_x,.96,'cut=3',ha='center',va='bottom',fontsize=8.3)
    arrow(prefix_ax,(.93,.59),(.63,.48),color=moving,lw=.7)
    prefix_ax.text(.14,-.06,'前缀：工序映射与MA均相同，复制已排条目',fontsize=8.0)
    prefix_ax.text(.60,-.06,'完整新OS输入；仅重排后缀',fontsize=8.0)
    # Symbolic gap geometry illustrates the actual conservative decoder rule.
    # The x positions below are drawing coordinates, not experiment times.
    c=fig.add_axes([.07,.285,.90,.19]);c.axis('off');c.set(xlim=(0,16),ylim=(-.65,2.10))
    panel(c,'d','间隙插入的边界约束（符号示意）')
    for y,label in [(1.20,'插入前'),(.42,'插入后')]:
        c.text(0,y+.1,label,fontsize=8.3,va='center')
        rect(c,(1.7,y),1.7,.20,r'$u$',fc=cyan,ec=edge,fs=9)
        rect(c,(11.4,y),.7,.20,fc='#EEEEEE',ec=edge)
        rect(c,(12.1,y),2.7,.20,r'$w$',fc=cyan,ec=edge,fs=9)
        for x in [3.4,11.4]:c.plot([x,x],[y-.05,y+.30],color=edge,lw=.45)
        c.text(3.4,y+.33,r'$C_u$',ha='center',fontsize=8)
        c.text(11.4,y+.33,r'$B_w$',ha='center',fontsize=8)
    # Proposed v before insertion, and v in the accepted gap.
    for y in [1.70,.42]:
        rect(c,(4.4,y),.7,.20,fc='#EEEEEE',ec=moving)
        rect(c,(5.1,y),3.5,.20,r'$v$',fc=pink,ec=moving,fs=9)
    c.add_patch(FancyArrowPatch((8.6,1.68),(8.6,.64),arrowstyle='-|>',
        connectionstyle='arc3,rad=-.42',mutation_scale=7,color=moving,lw=.65))
    c.text(9.8,1.62,'插入',color=moving,fontsize=8.1)
    dimension(c,3.4,11.4,.98,'扫描机器间隙',fs=7.9)
    dimension(c,4.4,8.6,-.10,r'$q_v+p_{vm}$',fs=8.2)
    c.plot([4.4,4.4],[.36,.92],color=moving,lw=.5,ls='--')
    c.text(1.7,-.48,r'$B_v^0=\max\{R_v,C_u\}$',ha='left',fontsize=8.2)
    c.text(5.7,-.48,'日历冲突时右移',ha='left',fontsize=7.9)
    c.text(9.6,-.48,r'$C_v\leq B_w,\quad q_w^{new}=q_w^{stored}$',ha='left',fontsize=8.2)
    d = fig.add_axes([.09,.100,.88,.140]); panel(d,'e','母体编码的完整解码结果')
    entries=[vars(e) for e in schedule.entries.values()]
    fig._gantt_checks=p1_gantt(d,entries,
                              ['M01','M02','M03'],illustrative=True)
    for i,e in enumerate(entries):
        d.patches[2*i].set(edgecolor=edge,facecolor='#EEEEEE')
        d.patches[2*i+1].set(edgecolor=moving if e['operation_id']=='J1-O2' else edge,
                           facecolor=pink if e['operation_id']=='J1-O2' else cyan)
    d.add_patch(Rectangle((5,.83),3,.34,facecolor='white',edgecolor=edge,
                          hatch='////',lw=.4))
    d.text(6.5,1,'维护',ha='center',va='center',fontsize=8.3)
    d.set(xlim=(0,14),ylim=(2.85,-.45))
    d.set_xlabel('Time / illustrative unit'); d.xaxis.set_major_locator(MultipleLocator(2))
    fig.text(.07,.018,'灰色：准备；浅蓝：工序；浅红：所示工序；斜线：维护。',fontsize=8.0)
    fig.text(.97,.018,r'$C_{max}=13$   $TEC=18$   $L_{max}=12$',ha='right',fontsize=8.4)
    save(fig,'fig1_encoding')


def mechanism():
    fig = plt.figure(figsize=(WIDTH,7.10))
    axes=[fig.add_axes([x,.425,.30,.49]) for x in [.035,.35,.665]]
    for ax in axes:ax.set(xlim=(0,100),ylim=(-7,102));ax.axis('off')
    a,b,c=axes
    panel(a,'a','进化主循环');panel(b,'b','变邻域搜索');panel(c,'c','目标匹配迁移')
    # (a) Shared search layer; the hard gate is also checked inside evaluation.
    flow_node(a,50,96,86,10,'独立运行：D=2k / 4k / 8k',kind='terminal',fs=7.8)
    flow_node(a,50,83,78,13,'40个起点；完整解码\n固定尺度，建立两个档案')
    flow_node(a,50,67,46,15,r'$n_d<D$ ?',kind='decision',fs=8.6)
    flow_node(a,50,51,78,12,'锦标赛、交叉与自适应变异\n完整解码子代；更新输出档案',fs=7.8)
    flow_node(a,50,36,78,10,'合并亲子，预选精英')
    flow_node(a,50,23,78,10,'调用局部搜索（见b、d）')
    flow_node(a,50,12,78,8,'环境选择；更新内部档案')
    for top,bottom in [(91,89.5),(76.5,74.5),(59.5,57),(45,41),(31,28),(18,16)]:
        orth_arrow(a,[(50,top),(50,bottom)])
    a.text(56,59,'是',fontsize=7.6)
    orth_arrow(a,[(11,12),(4,12),(4,67),(27,67)])
    orth_arrow(a,[(73,67),(97,67),(97,-2),(89,-2)],'否',(84,70))
    # Output terminal placed at the bottom of the panel, without crossing steps.
    flow_node(a,50,-2,78,8,'输出全部已评价非支配解',kind='terminal',fs=7.8)
    # (b) Each neighborhood has ell trials; candidate evaluations cap at 4*ell.
    flow_node(b,55,96,74,12,'精英x、权重w；ℓ=2或6\n初始化k=1，t=0')
    flow_node(b,55,80,46,16,'继续条件？',kind='decision',fs=7.8)
    flow_node(b,55,58,76,25,'当前邻域至多试探ℓ次\n生成候选，精确评价（见d）\n相同编码跳过，接受即停\n每次评价：t←t+1',fs=7.7)
    flow_node(b,55,33,50,16,'本邻域有改进？',kind='decision',fs=7.8)
    flow_node(b,30,14,38,10,'更新x\nk←1',fs=8)
    flow_node(b,78,14,32,10,'保持x\nk←k+1',fs=8)
    flow_node(b,55,-2,66,8,'返回当前解x',kind='terminal',fs=8)
    for top,bottom in [(90,88),(72,70.5),(45.5,41)]:orth_arrow(b,[(55,top),(55,bottom)])
    orth_arrow(b,[(32,80),(4,80),(4,-2),(22,-2)],'否',(19,83))
    b.text(62,71,'是',fontsize=7.6)
    orth_arrow(b,[(30,33),(24,33),(24,22),(30,22),(30,19)],'是',(22,36))
    orth_arrow(b,[(80,33),(86,33),(86,22),(78,22),(78,19)],'否',(87,36))
    orth_arrow(b,[(30,9),(30,6),(98,6),(98,80),(78,80)])
    orth_arrow(b,[(78,9),(78,6),(98,6)])
    # (c) Only Nk=machine uses these rules; other neighborhoods are unchanged.
    flow_node(c,53,96,82,12,'',fs=8.2)
    c.text(53,98.5,'确定机器迁移方向',ha='center',va='center',fontsize=8.0)
    c.text(53,93.3,r'$d=\arg\max_i w_i$',ha='center',va='center',fontsize=8.3)
    rules=[(75,'Cmax：末端完工工序\n选择最小p+s的替代机器'),
           (51,'TEC：正节能迁移前5项随机\n否则随机工序选低能耗替代'),
           (27,'Lmax：瓶颈机可迁移工序\n预计最大占用最小前5项随机')]
    for y,label in rules:
        flow_node(c,57,y,76,16,label,fs=7.7)
        orth_arrow(c,[(8,y),(19,y)],f'd={rules.index((y,label))+1}',(14,y+4))
        orth_arrow(c,[(95,y),(99,y),(99,5),(91,5)])
    orth_arrow(c,[(53,90),(53,86),(8,86),(8,27)])
    flow_node(c,53,5,76,10,"修改MA，返回候选x'",kind='terminal',fs=7.9)
    # (d) Cached, full, and common-prefix paths all produce exact objectives.
    d=fig.add_axes([.045,.105,.92,.245]);d.axis('off')
    d.set(xlim=(0,100),ylim=(0,80))
    panel(d,'d','静态候选的精确评价：缓存、完整解码与前缀复用')
    flow_node(d,10,61,16,19,'缓存命中？',kind='decision',fs=8.0)
    flow_node(d,10,20,18,17,'返回缓存\n排程与目标',kind='terminal',fs=8.0)
    flow_node(d,34,61,16,19,r'$n_d<D$ ?',kind='decision',fs=8.4)
    flow_node(d,34,20,18,17,'终止本次\n独立运行',kind='terminal',fs=8.0)
    flow_node(d,59,61,20,19,'静态基准\n排程有效？',kind='decision',fs=7.6)
    flow_node(d,59,20,27,20,'共同前缀cut；复制条目\n完整OS输入，重排后缀',fs=7.9)
    flow_node(d,87,61,22,19,'完整解码\n初始化 / 遗传候选',fs=7.9)
    flow_node(d,87,20,22,20,'计算精确三目标\n缓存；nd←nd+1',fs=7.9)
    orth_arrow(d,[(18,61),(26,61)],'否',(22,69))
    orth_arrow(d,[(10,51.5),(10,28.5)],'是',(13,40))
    orth_arrow(d,[(42,61),(49,61)],'是',(45.5,69))
    orth_arrow(d,[(34,51.5),(34,28.5)],'否',(37,40))
    orth_arrow(d,[(69,61),(76,61)],'否',(72.5,69))
    orth_arrow(d,[(59,51.5),(59,30)],'是',(62,40))
    orth_arrow(d,[(87,51.5),(87,30)])
    orth_arrow(d,[(72.5,20),(76,20)])
    fig.text(.035,.402,'继续条件：k≤4、t<4ℓ且nd<D；Nk依次为插入、瓶颈机相邻OS交换、机器迁移、随机扰动。',fontsize=7.9)
    fig.text(.035,.056,'cut：映射工序与修复后MA同时相同的最长共同读取前缀；cut=0重排全部，cut=n无需重排。',fontsize=8.0)
    fig.text(.035,.027,'匹配规则用于生成候选，不剪枝；缓存命中不增加nd，但计入t。每个新编码在解码前检查独立预算D。',fontsize=8.0)
    save(fig,'fig2_mechanism')


def data_figures():
    df=pd.read_csv(OUT/'metrics.csv');curves=pd.read_csv(OUT/'curves.csv')
    effects=pd.read_csv(OUT/'effects.csv')
    allrows=[]
    for name in ['main','benchmark','benchmark_ablation','matched','stress']:
        allrows.extend(json.loads((OUT/(name+'_runs.json')).read_text(encoding='utf8')))
    main=df[df.instance=='E0']
    methods=['R-NSGA','H-NSGA','R-VNS','H-VNS','H-VNS-R','H-MVNS','MOEAD']

    fig,axs=charts(2,3.10,bottom=.25,wspace=.40)
    shown=['R-NSGA','H-NSGA','H-VNS','H-MVNS','MOEAD']
    for ax,col,label in zip(axs,['hv','igdplus'],['Hypervolume',r'IGD$^+$']):
        for m in shown:
            g=curves[(curves.instance=='E0')&(curves.method==m)].groupby('evaluations')[col]
            mean=g.mean();se=g.std()/np.sqrt(g.count())
            ax.errorbar(mean.index,mean,yerr=1.96*se,color=C[m],ls='-',marker=MARKERS[m],
                        ms=4.3,mfc='none',mew=.65,lw=.65,elinewidth=.35,
                        capsize=1.2,capthick=.35,label=LABEL[m])
        ax.set(xlabel='Unique decode evaluations',ylabel=label,xlim=(-40,4120))
        ax.set_xticks([0,1000,2000,3000,4000]);ax.tick_params(labelsize=8.1);grid(ax)
    panel(axs[0],'a','解集覆盖');panel(axs[1],'b','经验参照集距离')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.010),ncol=5,fontsize=8.4)
    save(fig,'fig3_budget')

    fig=plt.figure(figsize=(WIDTH,4.6))
    left=fig.add_axes([.09,.61,.385,.29]);hvax=fig.add_axes([.59,.61,.385,.29])
    right=fig.add_axes([.09,.14,.88,.29])
    values=[main[main.method==m].hv.to_numpy() for m in methods]
    blue_boxplot(hvax,values)
    blue_boxplot(left,[main[main.method==m].igdplus.to_numpy() for m in methods])
    for ax,metric in [(left,r'IGD$^+$'),(hvax,'Hypervolume')]:
        names=[LABEL[m].replace('-', '-\n',1) if m!='MOEAD' else 'MOEA/D' for m in methods]
        ax.set(xticks=range(1,8),xticklabels=names,ylabel=metric)
        ax.tick_params(axis='x',length=0,labelsize=7.8);grid(ax)
    panel(left,'a',r'IGD+分布');panel(hvax,'b','HV分布')
    contrasts=['H effect, no VNS','VNS effect, random','VNS effect, hybrid',
               'H-MVNS minus H-VNS','H-MVNS minus H-NSGA']
    labels=['规则初始化','原VNS / 随机起点','原VNS / 规则起点','匹配 − 原VNS','匹配 − 仅规则']
    for i,contrast in enumerate(contrasts):
        r=effects[effects.contrast==contrast].iloc[0]
        right.errorbar(i,r.delta,yerr=[[r.delta-r.lo],[r.hi-r.delta]],
                       fmt='o',ms=4,mfc='white',mec=BLUE,ecolor=GRAY,
                       elinewidth=.55,capsize=2,capthick=.5)
    right.axhline(0,color=GRAY,ls=(0,(3,2)),lw=.45)
    right.set(xticks=range(5),xticklabels=labels,ylabel=r'Paired $\Delta$HV',xlim=(-.55,4.55))
    right.tick_params(axis='x',length=0,labelsize=8.7)
    right.yaxis.set_major_locator(MaxNLocator(4));panel(right,'c','配对效应')
    fig.text(.09,.035,'上：各方法30次运行；下：均值差及95%配对bootstrap区间。',fontsize=8.3,color=INK)
    save(fig,'fig4_factorial')

    step=pd.read_csv(OUT/'single_step.csv')
    fig,axs=charts(3,3.0,bottom=.24,wspace=.50)
    for ax,focus,letter,title in zip(axs,['Cmax','TEC','Lmax'],'abc',
                                    ['完工端点方向','加工能耗方向','最大占用方向']):
        p=step[step.focus==focus].groupby(['parent_seed','version']).relative_target.mean().unstack()*100
        blue_boxplot(ax,[p.original.to_numpy(),p.matched.to_numpy()],positions=[0,1],width=.36)
        ax.axhline(0,color=GRAY,ls=(0,(3,2)),lw=.7)
        ax.set(xticks=[0,1],xticklabels=['Original','Matched'],xlim=(-.38,1.38))
        ax.tick_params(axis='x',length=0);grid(ax);panel(ax,letter,title)
    axs[0].set_ylabel('Target objective change / %')
    fig.text(.09,.025,'每个箱体包含10个共同母体的汇总值；各面板采用独立纵轴。',fontsize=8.3,color=INK)
    save(fig,'fig5_single_step')

    be=pd.read_csv(OUT/'benchmark_effects.csv')
    fig,axs=charts(3,3.7,bottom=.20,wspace=.35)
    for ax,base,letter,title in zip(axs,['R-NSGA','H-NSGA','H-VNS'],'abc',
                                   ['匹配 − 随机NSGA','匹配 − 仅规则','匹配 − 原VNS']):
        sub=be[be.contrast=='H-MVNS minus '+base].sort_values('instance')
        for i,r in enumerate(sub.itertuples()):
            col=C[base]
            ax.plot([r.lo,r.hi],[i,i],color=GRAY,lw=.6,zorder=2)
            ax.plot(r.delta,i,marker=MARKERS[base],ms=4,mfc='white',mec=col,mew=.6,ls='',zorder=3)
        ax.axvline(0,ls=(0,(3,2)),lw=.8,color=GRAY)
        ax.set(yticks=range(10),yticklabels=[f'Mk{i:02d}' for i in range(1,11)],
               xlabel=r'Paired $\Delta$HV',ylim=(9.6,-.6))
        ax.tick_params(axis='y',length=0,labelsize=8.5)
        ax.xaxis.set_major_locator(MaxNLocator(4));panel(ax,letter,title)
        if ax!=axs[0]:ax.set_yticklabels([])
    fig.text(.09,.025,'点：每实例10个配对种子的均值差    线：95% bootstrap区间',fontsize=8.3,color=GRAY)
    save(fig,'fig6_benchmarks')

    h=main[main.method=='H-MVNS'];seed=int(h.iloc[(h.hv-h.hv.median()).abs().argmin()].seed)
    rep={r['method']:r for r in allrows if r['instance']=='E0' and r['seed']==seed}
    chosen=rep['H-MVNS']['representative'];points=np.asarray(rep['H-MVNS']['front'])
    ends=[int(points[:,k].argmin()) for k in range(3)]
    assert seed==410009 and np.allclose(chosen['objectives'],[488.5,757.86,478])
    (F/'data/selected_plans.json').write_text(json.dumps(dict(seed=seed,representative=chosen,
        endpoints=points[ends].tolist(),selection='seed closest to median final HV; representative min normalized Euclidean ideal distance within run'),
        ensure_ascii=False,indent=2),encoding='utf8')
    fig,axs=charts(3,3.25,bottom=.28,wspace=.55)
    for ax,(i,j),letter,title in zip(axs,[(0,1),(0,2),(1,2)],'abc',
                                    ['完工—能耗','完工—占用','能耗—占用']):
        for m in ['H-NSGA','H-VNS','H-MVNS']:
            p=np.unique(np.round(np.asarray(rep[m]['front']),9),axis=0)
            ax.scatter(p[:,i],p[:,j],s=3,color=C[m],alpha=.24,lw=0,rasterized=False)
            # Connect only the observed 2D nondominated envelope, never invent
            # a three-objective front by joining independently chosen extremes.
            ordered=p[np.lexsort((p[:,j],p[:,i]))];envelope=[];best=float('inf')
            for q in ordered:
                if q[j]<best-1e-9:envelope.append(q);best=q[j]
            envelope=np.asarray(envelope)
            ax.plot(envelope[:,i],envelope[:,j],color=C[m],lw=.6,
                    marker=MARKERS[m],ms=4.2,mfc='white',mew=.6,
                    markevery=max(1,len(envelope)//9),label=m)
        for k,idx in enumerate(ends):
            ax.scatter(points[idx,i],points[idx,j],s=34,marker=['s','^','o'][k],
                        facecolors='white',edgecolors=INK,lw=.85,zorder=5)
        p=chosen['objectives'];ax.scatter(p[i],p[j],marker='*',s=85,color=INK,zorder=6)
        ax.set_xlabel([r'$C_{max}$ / min',r'$TEC$ / kWh',r'$L_{max}$ / min'][i])
        ax.set_ylabel([r'$C_{max}$ / min',r'$TEC$ / kWh',r'$L_{max}$ / min'][j])
        ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(4))
        grid(ax);panel(ax,letter,title)
        ax.legend(loc='upper left' if i==0 and j==2 else 'upper right',fontsize=7.5,
                  handlelength=1.7,labelspacing=.45,borderaxespad=.5)
    fig.text(.5,.010,'□ 完工端点    △ 能耗端点    ○ 占用端点    ★ 同一折中解',ha='center',fontsize=8.3,color=GRAY)
    save(fig,'fig7_tradeoffs')

    entries=chosen['entries'];machines=[f'M{k:02d}' for k in range(1,11)]
    fig=plt.figure(figsize=(WIDTH,5.35))
    a=fig.add_axes([.075,.325,.90,.60]);b=fig.add_axes([.10,.105,.87,.115])
    fig._gantt_checks=p1_gantt(a,entries,machines)
    a.set_xlabel('Time / min');a.set_xticks([0,100,200,300,400,500])
    a.set_xlim(0,505)
    panel(a,'a','同一折中计划的加工甘特图')
    loads=[sum(e['end']-e['setup_start'] for e in entries if e['machine_id']==m) for m in machines]
    b.plot(range(10),loads,'o-',color=BLUE,ms=3.2,mfc='white',mew=.5,lw=.6)
    for x,v in enumerate(loads):b.annotate(f'{v:.1f}',(x,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=7.3)
    b.set(xticks=range(10),xticklabels=[rf'$M_{{{k:02d}}}$' for k in range(1,11)],
          xlim=(-.4,9.4),ylim=(-40,580),ylabel='Occupancy / min')
    b.set_yticks([0,250,500]);b.tick_params(labelsize=8)
    panel(b,'b','同一计划的机器累计占用')
    fig.text(.075,.025,'条内为工件编号；条下为实际起止时刻；同一工件在各机器上使用相同颜色。',fontsize=8.3)
    pd.DataFrame(fig._gantt_checks['boundary_labels']).to_csv(F/'data/gantt_boundaries.csv',index=False)
    save(fig,'fig8_schedule')

    physical=df[(df.method=='H-MVNS')&df.instance.isin(['CF0','E1','CFM','E2'])]
    ce=pd.read_csv(OUT/'constraint_effects.csv');order=['CF0','E1','CFM','E2']
    fig,axs=charts(3,3.05,bottom=.24,wspace=.55)
    for idx,(col,label,color) in enumerate([('cmax_min',r'$C_{max}^{min}$',BLUE),
                                           ('lmax_min',r'$L_{max}^{min}$',TEAL)]):
        g=physical.groupby('instance')[col];v=g.mean().reindex(order);sd=g.std().reindex(order)
        for has_maintenance,pair,ls in [(0,['CF0','E1'],'-'),(1,['CFM','E2'],(0,(4,2)))]:
            axs[idx].errorbar([0,1],v.loc[pair],yerr=sd.loc[pair],color=BLUE if has_maintenance==0 else RUST,ls='-',
                            marker='o' if has_maintenance==0 else 's',ms=4,
                            mfc='white',mew=.6,lw=.65,elinewidth=.4,capsize=1.5,
                            label='无维护' if has_maintenance==0 else '有维护')
        pp=physical.pivot(index='seed',columns='instance',values=col)
        values=pp.E2-pp.E1-pp.CFM+pp.CF0
        axs[2].scatter(idx+np.linspace(-.10,.10,len(values)),values,s=10,
                         facecolors='white',edgecolors=color,lw=.7,alpha=.65)
        r=ce[(ce.objective==col)&(ce.contrast=='interaction')].iloc[0]
        axs[2].errorbar(idx,r.delta,yerr=[[r.delta-r.lo],[r.hi-r.delta]],fmt='D',
                        ms=4.0,mfc='white',capsize=2,lw=.65,color=color,zorder=4)
    for idx,col in enumerate([r'$C_{max}^{min}$ / min',r'$L_{max}^{min}$ / min']):
        axs[idx].set(xticks=[0,1],xticklabels=['无换型','有换型'],xlim=(-.15,1.15),ylabel=col)
        axs[idx].legend(loc='upper left',fontsize=7.8)
    axs[2].set(xticks=[0,1],xticklabels=[r'$C_{max}^{min}$',r'$L_{max}^{min}$'],
                xlim=(-.45,1.45),ylabel='Interaction / min')
    axs[2].axhline(0,ls=(0,(3,2)),lw=.45,color=GRAY)
    for ax in axs:grid(ax)
    panel(axs[0],'a','完工端点');panel(axs[1],'b','占用端点');panel(axs[2],'c','同种子交互')
    fig.text(.09,.025,'(a)、(b)：均值±SD；(c)：12个种子交互及95%配对bootstrap区间。',fontsize=8.3)
    save(fig,'fig9_constraints')

    tuning=pd.read_csv(OUT/'intensity_metrics.csv');means=tuning.groupby('trials').mean(numeric_only=True)
    fig,axes=charts(3,2.95,bottom=.24,wspace=.58);a,b,c=axes
    p=tuning.pivot(index='seed',columns='trials',values='hv').sort_index(axis=1)
    for _,r in p.iterrows():a.plot(r.index,r.values,color='#C0C0C0',lw=.4,zorder=1)
    for trials,g in tuning.groupby('trials'):
        a.errorbar(trials,g.hv.mean(),yerr=g.hv.std(),fmt='o',capsize=2,
                   mfc='white',color=BLUE,ms=4,lw=.6,zorder=4)
    a.plot(means.index,means.hv,color=BLUE,lw=.6)
    a.set(xlabel='Trials per neighborhood',ylabel='Hypervolume',xticks=[1,2,4,6],xlim=(.6,6.4))
    b.plot(means.index,100*means.local_fraction,'x-',color=TEAL,ms=4,lw=.6,mew=.6)
    b.set(ylabel='Local decode / %',ylim=(0,68),xlabel='Trials per neighborhood',xticks=[1,2,4,6],xlim=(.6,6.4))
    b.set_yticks([0,30,60])
    c.plot(means.index,means.generations,'s-',color=RUST,ms=4,lw=.6,mfc='white',mew=.6)
    c.set(ylabel='Generations',xlabel='Trials per neighborhood',xticks=[1,2,4,6],ylim=(30,86))
    c.set_yticks([40,60,80]);c.set_xlim(.6,6.4)
    for ax in [a,b,c]:grid(ax)
    panel(a,'a','HV探索结果');panel(b,'b','局部解码占比');panel(c,'c','完成代数')
    fig.text(.09,.025,'(a)：6个种子的配对轨迹与均值±SD；(b)、(c)：两项预算指标分别设轴。',fontsize=8.3)
    save(fig,'fig10_intensity')


if __name__=='__main__':
    report_path=ROOT/'qa/figure-render-verification.json'
    previous=json.loads(report_path.read_text(encoding='utf8')) if report_path.exists() else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        toy();mechanism()
        if '--concepts-only' not in sys.argv:data_figures()
    glyph_warnings=[str(w.message) for w in caught if 'Glyph' in str(w.message)]
    assert not glyph_warnings,glyph_warnings
    rendered_count=len(REPORT)
    if '--concepts-only' in sys.argv and len(previous.get('figures',[]))==10:
        updated={item['figure']:item for item in REPORT}
        REPORT[:]=[updated.get(item['figure'],item) for item in previous['figures']]
    report_path.write_text(json.dumps(
        dict(style='technical-insertion-schematic; standard-flowchart; other-plots-P1-P4',figures=REPORT,missing_glyph_warnings=glyph_warnings),
        ensure_ascii=False,indent=2),encoding='utf8')
    print('Rendered',rendered_count,'figures; report covers',len(REPORT),
          'figures; zero missing glyphs and zero out-of-canvas text.')
