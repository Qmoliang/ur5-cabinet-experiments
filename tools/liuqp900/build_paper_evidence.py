"""Derive paper tables/figures from the published frozen records; never run control."""
import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FROZEN = '0f402010bb8335df661fdace8745106b7ee5aede'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    args = ap.parse_args()
    root = Path(args.repo).resolve()
    def git_bytes(path):
        return subprocess.check_output(['git', 'show', FROZEN + ':' + path], cwd=root)
    raw = git_bytes('docs/liuqp900/data.json')
    data = json.loads(raw)
    cases = data['cases']
    assert len(cases) == 900 and len({c['id'] for c in cases}) == 900
    grouped = {a: [c for c in cases if c['arm'] == a] for a in 'CAB'}
    matched = {c['case_id']: {} for c in cases}
    for c in cases:
        matched[c['case_id']][c['arm']] = c
    assert len(matched) == 300 and all(set(t) == set('CAB') for t in matched.values())
    for t in matched.values():
        assert len({c['input_sha256'] for c in t.values()}) == 1
        assert len({c['scene_sha256'] for c in t.values()}) == 1
    stats = {}
    for a, rows in grouped.items():
        stats[a] = dict(n=len(rows), raw=sum(c['goal_reached'] for c in rows),
                        physical=sum(c['physical_safe_arrival'] for c in rows),
                        strict=sum(c['strict_arrival'] for c in rows),
                        statuses=dict(collections.Counter(c['status'] for c in rows)), timing={})
        for key in ['arm_total_wall_s', 'total_with_shared_input_charge_s', 'goal_initialization_s', 'audit_wall_s']:
            v = np.array([c[key] for c in rows], float)
            stats[a]['timing'][key] = dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), maximum=float(max(v)), sum=float(sum(v)))
        assert stats[a]['strict'] == data['arms'][a]['strict_successes']
        assert stats[a]['physical'] == data['arms'][a]['physical_safe_arrivals']
    assert [stats[a]['strict'] for a in 'CAB'] == [128, 186, 294]
    assert [stats[a]['physical'] for a in 'CAB'] == [174, 186, 294]
    pairs = {}
    for a, b in [('C','A'), ('A','B'), ('C','B')]:
        entry = {}
        for metric in ['strict_arrival', 'physical_safe_arrival']:
            counts = collections.Counter(('both_success' if t[a][metric] and t[b][metric] else 'gain' if t[b][metric] else 'regression' if t[a][metric] else 'both_failed') for t in matched.values())
            common = [t for t in matched.values() if t[a][metric] and t[b][metric]]
            entry[metric] = {name: counts[name] for name in ['both_success','gain','regression','both_failed']}
            entry[metric]['common_wall_medians_s'] = [float(np.median([t[x]['arm_total_wall_s'] for t in common])) for x in [a,b]]
        pairs[a+'_'+b] = entry
    assert [(pairs[k]['strict_arrival']['gain'], pairs[k]['strict_arrival']['regression']) for k in ['C_A','A_B','C_B']] == [(59,1),(109,1),(166,0)]
    executor_gains = [t for t in matched.values() if not t['C']['strict_arrival'] and t['A']['strict_arrival']]
    executor_split = collections.Counter('arrived_physical_but_not_strict' if t['C']['physical_safe_arrival'] else 'arrived_but_physical_failed' if t['C']['goal_reached'] else 'not_arrived' for t in executor_gains)
    control_gains = [t['B'] for t in matched.values() if not t['A']['strict_arrival'] and t['B']['strict_arrival']]
    triggers = dict(gains=len(control_gains), advanced=sum(c['advanced_control_activated'] for c in control_gains), surface=sum(c['surface_escape_episodes'] > 0 for c in control_gains), recurrence=sum(c['configuration_recurrences'] > 0 for c in control_gains))
    assert dict(executor_split) == {'arrived_physical_but_not_strict':44, 'not_arrived':14, 'arrived_but_physical_failed':1}
    assert triggers == dict(gains=109, advanced=109, surface=20, recurrence=1)
    failure_keys = ['id','case_id','task','seed','status','final_error_mm','simulation_s','arm_total_wall_s','advanced_control_activated','surface_escape_episodes','configuration_recurrences','physical','feasibility','memory','strict_memory','error_message','raw_evidence','raw_evidence_sha256']
    failures = [{k:c.get(k) for k in failure_keys} for c in sorted(grouped['B'],key=lambda c:(c['task'],c['seed'])) if not c['strict_arrival']]
    legacy = json.loads(git_bytes('docs/closeout/data.json'))
    assert legacy['known_total'] == legacy['online_total'] == 300
    out = dict(frozen_commit=FROZEN, input_data_sha256=hashlib.sha256(raw).hexdigest(), n_conditions=300, n_runs=900,
               arms=stats, pairs=pairs, executor_gain_decomposition=dict(executor_split), control_gain_triggers=triggers,
               failures_B=failures, by_task=data['by_task'], legacy={k:legacy[k] for k in ['known_total','online_total','known_passed','online_passed']},
               inference_scope='Descriptive paired results; no independent-world confidence claim or new simulation.')
    folder = root/'docs/library/files/liuqp900'
    (folder/'paper-evidence.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
    tables = {}
    tables['outcomes'] = table(['组','原始到达','物理安全到达','严格成功'], [[a,f"{stats[a]['raw']}/300",f"{stats[a]['physical']}/300（{stats[a]['physical']/3:.2f}%）",f"{stats[a]['strict']}/300（{stats[a]['strict']/3:.2f}%）"] for a in 'CAB'])
    rows=[]
    for key,entry in pairs.items():
        for metric,label in [('strict_arrival','严格'),('physical_safe_arrival','物理安全到达')]:
            x=entry[metric];rows.append([key.replace('_','→'),label,x.get('both_success',0),x.get('gain',0),x.get('regression',0),x.get('both_failed',0),x.get('gain',0)-x.get('regression',0)])
    tables['paired'] = table(['比较','口径','共同成功','新增','退化','共同失败','净增'],rows)
    rows=[]
    for a in 'CAB':
        t=stats[a]['timing'];v=t['arm_total_wall_s'];shared=t['total_with_shared_input_charge_s']
        rows.append([a,f"{v['median']:.3f}",f"{v['p90']:.3f}",f"{v['maximum']:.3f}",f"{shared['median']:.3f}",f"{t['goal_initialization_s']['median']:.3f}",f"{t['audit_wall_s']['median']:.3f}"])
    tables['timing'] = table(['组','完整单次中位数/s','P90/s','最大值/s','含共享准备计费中位数/s','终态初始化中位数/s','事后审计中位数/s'],rows)
    tables['tasks'] = table(['任务（每组20次）','C 物理/严格','A 物理/严格','B 物理/严格'],[[task]+[f"{x[a]['physical_safe']}/{x[a]['strict']}" for a in 'CAB'] for task,x in data['by_task'].items()])
    site='https://qmoliang.github.io/ur5-cabinet-experiments/liuqp900/'
    tables['failures'] = table(['B组失败案例','末误差/mm','仿真/s','完整墙钟/s','扩展激活','绕行次数','重复检测次数'],[[f"[{c['task']} s{c['seed']:02d}]({site}#case={c['case_id']}&arm=B)",f"{c['final_error_mm']:.6f}",f"{c['simulation_s']:.0f}",f"{c['arm_total_wall_s']:.3f}",'是' if c['advanced_control_activated'] else '否',c['surface_escape_episodes'],c['configuration_recurrences']] for c in failures])
    (folder/'experiment-tables.md').write_text('# 冻结900次实验：逐例重算表格\n\n本页仅从已发布索引重算描述统计，不改变原始判定。源提交：`'+FROZEN+'`。\n\n[论文初稿](paper-draft.md) · [机器可读统计](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/paper-evidence.json)\n\n'+'\n\n'.join('## '+name+'\n\n'+tables[key] for key,name in [('outcomes','总体结果'),('paired','逐例新增与退化'),('tasks','15项任务'),('timing','完整耗时'),('failures','全部六个B组失败')])+'\n',encoding='utf-8')
    (root/'.local/paper-tables.json').write_text(json.dumps(tables,ensure_ascii=False),encoding='utf-8')
    figures=folder/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig, ax = plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    x=np.arange(3);w=.34
    for i,(key,label,color) in enumerate([('physical','Physical-safe arrival','#91b4a6'),('strict','Strict success','#275d50')]):
        vals=[stats[a][key] for a in 'CAB'];bars=ax[0].bar(x+(i-.5)*w,vals,w,label=label,color=color)
        ax[0].bar_label(bars,labels=[str(v) for v in vals],padding=3)
    ax[0].set(xticks=x,xticklabels=['C: original execution','A: shared executor','B: new control'],ylim=(0,330),ylabel='Successful runs / 300',title='(a) Frozen matched comparison')
    ax[0].tick_params(axis='x',labelsize=9);ax[0].legend(loc='upper left',fontsize=9)
    keys=['C_A','A_B'];gx=np.arange(2)
    for i,(metric,label,color) in enumerate([('strict_arrival','Strict','#275d50'),('physical_safe_arrival','Physical-safe','#91b4a6')]):
        pos=gx+(i-.5)*w;g=[pairs[k][metric]['gain'] for k in keys];loss=[-pairs[k][metric]['regression'] for k in keys]
        bars=ax[1].bar(pos,g,w,color=color,label=label);ax[1].bar_label(bars,padding=3)
        bars=ax[1].bar(pos,loss,w,color=color,hatch='///');ax[1].bar_label(bars,labels=[str(v) for v in loss],padding=3)
    ax[1].axhline(0,color='#555',lw=.8);ax[1].set(xticks=gx,xticklabels=['C → A: executor package','A → B: control + initialization'],ylim=(-13,125),ylabel='Paired gains (+) / regressions (−)',title='(b) Incremental contributions')
    ax[1].tick_params(axis='x',labelsize=9)
    fig.suptitle('900 runs = 300 matched input conditions × 3 arms',fontsize=13)
    fig.savefig(figures/'outcomes.svg',metadata={'Date':None});fig.savefig(figures/'outcomes.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,4.1),layout='constrained')
    times=[stats[a]['timing']['arm_total_wall_s'] for a in 'CAB'];med=[v['median'] for v in times];p90=[v['p90'] for v in times];maximum=[v['maximum'] for v in times]
    ax.plot(x,med,'o-',label='Median',color='#275d50');ax.plot(x,p90,'s--',label='P90',color='#b17a35');ax.plot(x,maximum,'^:',label='Maximum',color='#a74444')
    for vals in [med,p90,maximum]:
        for xx,v in zip(x,vals):ax.annotate(f'{v:.1f}',(xx,v),xytext=(7,4),textcoords='offset points',fontsize=9)
    ax.set(xticks=x,xticklabels=['C: original execution','A: shared executor','B: new control'],yscale='log',ylim=(10,1900),xlim=(-.2,2.45),ylabel='Full per-run wall time (s, logarithmic)',title='All attempts; B initialization included; audits reported separately')
    ax.grid(axis='y',alpha=.2);ax.legend(loc='lower right');fig.savefig(figures/'timing.svg',metadata={'Date':None});fig.savefig(figures/'timing.png',dpi=180);plt.close(fig)
    print(json.dumps({'runs':900,'conditions':300,'counts':{a:[stats[a]['physical'],stats[a]['strict']] for a in 'CAB'},'failures_B':len(failures),'figures':2,'data_sha256':out['input_data_sha256']}))

if __name__ == '__main__':
    main()
