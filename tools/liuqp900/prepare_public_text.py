from pathlib import Path
import json,re,subprocess,gzip,struct,hashlib
W=Path(__file__).resolve().parents[2];D=W/'docs/library/files/liuqp900';R=W/'docs/liuqp900';data=json.loads((R/'data.json').read_text(encoding='utf-8'));repo='https://github.com/Qmoliang/ur5-cabinet-experiments';updates=[]
def text(path):
 p=subprocess.run(['git','show',':'+path],cwd=W,capture_output=True)
 if p.returncode:p=subprocess.run(['git','show','HEAD:'+path],cwd=W,capture_output=True,check=True)
 return p.stdout.decode('utf-8-sig')
def put(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8',newline='\n')
def edit(p,s):updates.append(dict(path=p,text=s))
# Verify all historical public initial states and targets, not just matching names.
legacy=json.loads((W/'docs/closeout/data.json').read_text(encoding='utf-8'));old={c['id']:c for c in legacy['cases'] if c['mode']=='online'};cache={};qmax=0.;tmax=0.
for c in data['conditions']:
 p=old[c['id']]
 if p['trajectory'] not in cache:cache[p['trajectory']]=gzip.decompress((W/'docs/closeout'/p['trajectory']).read_bytes())
 q=struct.unpack_from('<6d',cache[p['trajectory']],p['trajectory_offset_frames']*72);qmax=max(qmax,max(abs(x-y) for x,y in zip(q,c['q0'])));tmax=max(tmax,max(abs(x-y) for x,y in zip(p['target'],c['target'])))
assert qmax<1e-12 and tmax<1e-12
put(R/'evidence/legacy-input-check.json',json.dumps(dict(conditions=300,passed=True,max_initial_q_difference=qmax,max_target_difference=tmax,protocols_still_different=True),indent=2))
# Results and interpretation remain derived from the frozen index.
rows=['# 900次控制对照：结果与贡献分离','','> 同一历史冻结输入上的顺序增量对照。原结果不改写，失败不删除；不是900个独立场景，也不是与旧RRT系统的公平排名。','','[打开全部回放](../../../../liuqp900/) · [原始索引](../../../../liuqp900/data.json)','','| 组 | 定义 | 原始到达 | 物理安全到达 | 完整严格成功 |','| --- | --- | ---: | ---: | ---: |']
names={'C':'原QP＋原单步执行与原安全门','A':'原QP＋共同认证执行器','B':'终态初始化与新控制＋共同执行器'}
for arm in ['C','A','B']:
 a=data['arms'][arm];rows.append(f"| {arm} | {names[arm]} | {a['raw_goal_arrivals']}/300 | {a['physical_safe_arrivals']}/300 | {a['strict_successes']}/300 |")
rows+=['','三组同数值求解配置。C不是历史默认OSQP参数的回退；它恢复的是原执行方式。物理安全到达要求原1 mm/50周期、物理阈值与预算；严格成功还要求实际原QP行、末态制动/静止和适用的记忆审计。','','## 逐段比较','','| 比较 | 口径 | 新增 | 退化 | 净增 |','| --- | --- | ---: | ---: | ---: |']
for label,c in data['comparisons'].items():
 for criterion,v in c['counts'].items():rows.append('| '+c['source']+'→'+c['target']+' | '+('严格成功' if criterion=='strict_arrival' else '物理安全到达')+' | '+str(v.get('gain',0))+' | '+str(v.get('regression',0))+' | '+str(v.get('gain',0)-v.get('regression',0))+' |')
rows+=['','执行器新增59个严格成功中，44个原已物理安全到达但QP/制动审计未过；14个原未到达；1个原到达但物理安全失败。新控制的109个新增均为原A未到达。不能把前者都归为新控制收益。','', '新控制新增109例中，continuation机制激活109例，表面绕行20例，构型重复重选1例。触发频率不等于独立因果贡献。静止求解与恒速保持还产生约0.2 rad/s的条件性有效速度界，执行器不是透明外壳。','','## 退化与安全失败','','- C→A严格退化：`iris_bins_t3 / seed 8`，C在8.16秒仿真时间内成功，A运行300秒失败，B成功。','- A→B严格退化：`iris_bins_t0 / seed 17`，A成功，B停在1.38 mm；C可物理安全到达，但有10个原QP行违规周期。因此C→B严格口径双方失败，物理安全到达口径仍是退化。','- C另有`iris_bins_t0 / seed 6`到达但环境间隙5.9575 mm低于6 mm，按安全失败保留。','- A的共享执行器一致性异常及数值求解失败仍保留，不把这些全称为数学局部极小。','','## 完整耗时','','单位秒，包含初始化、设置、执行和收尾；全部尝试保留。','','| 组 | 中位数 | P90 | 最大值 |','| --- | ---: | ---: | ---: |']
for arm in ['C','A','B']:
 t=data['arms'][arm]['timing_all_attempts']['arm_total_wall_s'];rows.append(f"| {arm} | {t['median']:.3f} | {t['p90']:.3f} | {t['maximum']:.3f} |")
rows+=['','共同严格成功时，C→A的中位数为3.776→7.832秒（127对）；A→B为9.131→185.053秒（185对）。B的初始化开销明显，不能宣称每例加速。','', 'C补跑整批112.09分钟，含审计113.09分钟。超时停止与收尾消耗如实记录，任何完整单次超过1200秒的结果不算成功。','','## 条件与限制','','C/A/B均为MuJoCo运动学反馈和历史冻结观测地图。每次最多300秒仿真、1200秒完整计算；C在后续批次执行，延迟和负载影响没有通过同时随机化完全消除。没有“新控制＋原执行器”第四组，不能分离所有交互。','', '[详细代码与证明](methods.md) · [输入与原始证据范围](reconciliation.md)']
put(D/'results.md','\n'.join(rows)+'\n')
recon=['# 旧RRT系统与900次控制对照：版本差异与修正','','旧结果继续保留；本页修正解释口径，不改写历史轨迹或判定。','','## 两种实验回答不同问题','','| 项目 | 旧online final_r17 300例 | 新C/A/B 900次 |','| --- | --- | --- |','| 系统 | 在线观测＋认证RRT路径＋LiuQP跟踪＋安全门 | 冻结历史地图上的局部控制；控制阶段禁止RRT/全局路径搜索 |','| 目标信息 | 规划路径与航点引导 | C/A仅最终目标；B额外计算静态终态假设和固定终态线证书 |','| 输入地图 | 按原规则在线发布、更新 | 从各旧案例首动前最后可用因果发布冻结；无运动案例使用最后可用的运动前发布 |','| 时间 | 原300秒活动时域，感知/规划/等待遵循旧协议 | 300秒物理仿真含控制等待；1200秒完整单次墙钟含B初始化 |','| 验收 | 原协议安全到达；旧字段strict_pass不可直接视为新严格指标 | 物理安全到达与全程QP/备份/记忆严格成功分开 |','| 结果 | 298/300（系统级成绩） | C174/A186/B294物理安全到达；严格128/186/294 |','','起点与目标已逐例对照旧公开回放，300组一致。但地图信息条件、规划参与、执行时序和部分验收性质不同，因此298→294不能直接解释为新控制退化4例。','','## 同任务和seed的描述性交叉表','','以下仅用于找到对应历史记录，不是同协议胜负或因果收益。','','| 新组 | 两边通过 | 仅旧RRT通过 | 仅新组通过 | 两边未通过 |','| --- | ---: | ---: | ---: | ---: |']
for arm,c in data['legacy']['cross_tabs'].items():recon.append(f"| {arm} | {c['both_passed']} | {c['legacy_only']} | {c['controller_only']} | {c['neither']} |")
recon+=['','旧RRT保留的失败为`cage seed 8`与`drawer seed 20`。B这两例通过；B另有6例未通过。网站提供逐例互链，原始记录分别保存在各自队列。','','## 需要纠正的表述','','1. **298/300不是纯LiuQP成功率。** 它包含预先认证路径和跟踪。','2. **新方法并非“完全没有轨迹计算”。** 控制阶段不建RRT/搜索树，但会评估并认证当前构型到终态的固定收缩线，也使用完整终态关节方向。','3. **KKT等价不是全局到达证明。** 它只说明给定条件下同状态提案的有效QP等价；保持、制动和切换另有契约。','4. **本轮没有验证新的在线感知或实机硬实时。** 900条是15任务×20输入×3组，输入曾参与开发。','5. **不同“严格”字段需按所在版本解释。** 新严格结果包含每个实际速度的原QP行及末态备份检查；原始到达、安全到达与严格成功不能混用。','6. **全墙钟必须包含终态初始化。** 不能只报告B的运动阶段；共同成功与失败长尾应分别比较。','7. **所有失败保留。** 执行器退化、新控制退化、原QP不收敛、安全余量不足和预算停止均可查看。','','## 可下载与可播放范围','','- 900条保存q/末端状态全部可播放，无损float64载荷可验证。初始末端位置仅用原模型正运动学补齐，后续为原记录。','- 每条运行可下载所属证据ZIP：原始trajectory.npz、result、审计、事件及日志按字节保存；B终态输入也保留。','- 冻结源码有直接浏览链接和完整分包，提供SHA-256与原冻结清单。','- 约8.86 GB完整输入地图仍在原实验归档；网站公开其哈希、来源选择与采样图，不声称完整地图已托管在Pages。','- 地图采样最多2000个中心，仅为可视化辅助，不是完整安全几何。','- 回放运行正运动学，不运行控制器、不调用mj_step，不是新的试验。','','## 历史内容如何保留','','旧600条已知/在线回放、球表示配对和19项早期网站继续存在，源数据没有覆盖。旧章节作为历史文档保留，最新方法以本次冻结代码对齐稿为准。没有把本地暂缓公开、尚待方法对齐的其他规划器研究顺带发布。','','[900次控制回放](../../../../liuqp900/) · [旧系统队列](../../../../closeout/) · [更正后的方法](methods.md)']
put(D/'reconciliation.md','\n'.join(recon)+'\n')
# Add new Markdown documents to the existing library without changing original source files.
m=json.loads((W/'docs/library/manifest.json').read_text(encoding='utf-8'));m['groups']=[g for g in m['groups'] if g['id']!='liuqp900'];m['groups'].insert(0,dict(id='liuqp900',title='2026-11 冻结控制对照与方法'))
for g in m['groups']:
 if g['id']=='current':g['title']='2026-09 历史19项实验'
m['documents']=[d for d in m['documents'] if d['group']!='liuqp900'];defs=[('methods','methods.md','方法初稿：认证执行与终态引导持续绕行'),('proofs','proofs.md','九项数学性质与条件证明'),('status','property-status.md','证明与实验支持速查'),('code','code-alignment.md','29项主张与冻结源码映射'),('notes','author-notes.md','历史表述更正与实现缺口'),('results','results.md','900次结果与贡献分离'),('reconciliation','reconciliation.md','旧RRT与新控制对照的差异')]
new=[]
for ident,file,title in defs:
 p=D/file;new.append(dict(id='liuqp900-'+ident,title=title,group='liuqp900',format='md',file='files/liuqp900/'+file,filename=file,source='publication/liuqp900/'+file,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),note='2026-11-10冻结源码对齐稿；原实验不改写；条件性证明与经验结论分列。',scope_note='本页对应2026-11的C/A/B控制对照，使用历史冻结地图。旧RRT系统298/300属于另一协议；不主张全局收敛或实机硬实时。'))
m['documents']=new+m['documents'];put(W/'docs/library/manifest.json',json.dumps(m,ensure_ascii=False,indent=2))
# Existing library renderer: contextual note and current navigation only.
s=text('tools/build_library.mjs');s=s.replace('原文中的本机路径对应其原始实验目录。','原文中的本机路径对应其原始实验目录。').replace('历史版本的代理数量、场景和结论不自动代表当前 19 组重跑。','本库跨多个版本；指标须按各文档的队列与协议解释，不能混合统计。')
s=s.replace("'<div class=\"version-note\">阅读说明：正文按原文件保留。", "'<div class=\"version-note\">阅读说明：正文按所标版本保留。")
s=s.replace("</div></header><article id=\"content\"", "</div>\u0027+(d.scope_note?\u0027<div class=\"version-note\">\u0027+esc(d.scope_note)+\u0027</div>\u0027:\u0027\u0027)+\u0027</header><article id=\"content\"")
s=s.replace("'<a href=\"'+base+'../index.html#experiment\">实验回放</a>'", "'<a href=\"'+base+'../liuqp900/\">900次控制对照</a>'")
s=s.replace('当前演示页对应 2026-09-17 的落地柜子重跑。','历史19项页面对应2026-09-17；最新900次控制对照和2026-11方法稿已单独列出。')
s=s.replace('href="pages/current-overview.html">当前实验','href="pages/liuqp900-methods.html">最新方法')
edit('tools/build_library.mjs',s)
# Contextual banners preserve legacy data and existing controls.
s=text('docs/closeout/index.html');banner='<aside class="wrap panel" style="margin-top:24px;border-left:4px solid #147c69"><strong>历史系统队列 · 结果保持原样。</strong><p>本页在线298/300包含RRT路径规划、LiuQP跟踪与安全门，不是纯LiuQP成绩。后续900次控制对照使用冻结历史地图、不同执行与时间协议，不能直接排名。</p><a href="../liuqp900/">查看900次C/A/B回放与逐例审计 →</a> · <a href="../library/pages/liuqp900-reconciliation.html">口径差异与修正说明</a></aside>'
s=s if '历史系统队列 · 结果保持原样。' in s else s.replace('<main>', '<main>'+banner,1);s=s.replace('LATEST PAIRED STUDY / 2026.11.04','HISTORICAL PAIRED STUDY / 2026.11.04').replace('在线 final_r17 · 安全到达','在线 final_r17 · RRT系统安全到达').replace('已知地图 v5 · 严格通过','已知地图 v5 · 原协议通过');edit('docs/closeout/index.html',s)
s=text('docs/closeout/study/index.html');banner2='<aside style="max-width:1240px;margin:20px auto;padding:18px;background:#e9eee5;border-left:4px solid #147c69"><strong>本页为历史表示与组件性能研究。</strong> 球275/300、椭球298/300均处于带规划的系统中，不能作为本轮纯控制的成功率。<a href="../../liuqp900/">最新900次控制对照</a> · <a href="../../library/pages/liuqp900-reconciliation.html">口径说明</a></aside>';s=s if '本页为历史表示与组件性能研究。' in s else re.sub(r'(<body[^>]*>)',r'\1'+banner2,s,count=1);edit('docs/closeout/study/index.html',s)
edit('docs/index.html','<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="0;url=liuqp900/"><title>UR5控制对照与历史实验</title></head><body><h1>900次C/A/B控制对照</h1><p><a href="liuqp900/">打开完整结果、轨迹回放与证据</a></p><p><a href="closeout/">旧RRT系统与已知地图队列（保留原结果）</a> · <a href="library/">数学方法与研究文档库</a> · <a href="historical.html">历史19项实验</a></p></body></html>')
oldread=re.sub(r'\n## 900次冻结控制对照与方法稿.*?(?=\n## |\Z)','\n',text('README.md'),flags=re.S);intro='''## 900次冻结控制对照与方法稿 — 2026-11-10

**[交互结果与全部900条回放](https://qmoliang.github.io/ur5-cabinet-experiments/liuqp900/)** · **[方法Markdown](docs/library/files/liuqp900/methods.md)** · [证明与边界](docs/library/files/liuqp900/proofs.md) · [源码映射](docs/library/files/liuqp900/code-alignment.md)

15类任务×20个固定输入×3组。C为原LiuQP目标＋原执行方式与安全门，A为原目标＋共同认证执行器，B为终态初始化与新控制＋同一执行器。三组保持相同数值求解配置、输入与原安全标准；每次300秒仿真和1200秒完整墙钟预算。所有失败保留，原实验结果未改写。

| 组 | 物理安全到达 | 完整严格成功 | 完整墙钟中位数 / P90 |
| --- | ---: | ---: | ---: |
| C 原执行方式 | 174/300 | 128/300 | 15.19 / 1057.51 s |
| A 共同执行器 | 186/300 | 186/300 | 41.13 / 184.46 s |
| B 新控制方案 | 294/300 | 294/300 | 144.66 / 352.45 s |

严格成功额外要求全程原QP行、最终制动/静止及适用的记忆审计。C→A严格新增59/退化1，物理安全新增15/退化3；A→B新增109/退化1；C→B严格新增166/退化0，但物理安全仍有1个退化。B不是零代价或无退化方案。

全部900条保存状态可在浏览器播放、逐帧定位，并显示三组同一仿真时刻的误差。回放只计算正运动学，不运行控制器或mj_step。900份轨迹NPZ、审计、事件与失败日志按原字节分包下载；初始化成本、代码冻结哈希、输入来源及逐例源码链接均保留。完整约8.86 GB地图仍在原实验归档，公开哈希与采样可视化；网站不声称包含所有原始地图或已验证干净机器一键重跑。

**与旧RRT队列分开解释：** 原online final_r17的298/300是感知＋RRT规划＋跟踪＋安全门的系统结果。新实验使用其历史观测快照，不是重做同一在线系统。起点/目标可逐例关联，信息、时序及严格验收不同；不能把298→294写成公平回归下降4例。[完整差异说明](docs/library/files/liuqp900/reconciliation.md)

方法稿按最终冻结代码重写，数学证明、条件性执行契约、经验效果及未证明主张分列。已知执行器一致性缺口和有效速度保守性未被隐藏。LaTeX与原始方法资料也可下载，未声称全局收敛、完备逃逸或实机硬实时。[三组贡献与失败](docs/library/files/liuqp900/results.md)

'''
put(W/'README.md',oldread.replace('# UR5 Cabinet Experiments','# UR5 Cabinet Experiments\n\n'+intro,1))
put(W/'reports/liuqp900/README.md','# 冻结控制对照发布\n\n- [方法初稿](../../docs/library/files/liuqp900/methods.md)\n- [性质与证明](../../docs/library/files/liuqp900/proofs.md)\n- [结果和贡献](../../docs/library/files/liuqp900/results.md)\n- [旧RRT差异说明](../../docs/library/files/liuqp900/reconciliation.md)\n- [900条公开索引](../../docs/liuqp900/data.json)\n- [逐条轨迹来源](../../docs/liuqp900/evidence/replay-provenance.json)\n\n发布仅对保存数据作格式转换、正运动学检查和网页渲染，不重新运行机器人实验。完整原始地图未托管于Pages，证据范围以网页说明为准。\n')
# Fix time retention and responsive graph coordinates in newly authored app.
s=text('docs/liuqp900/app.js');s=s.replace('if(state.replay){await state.replay.load(state.current,records,token);if(token!==state.token)return;state.replay.setFrame(state.frame);','if(state.replay){const requestedFrame=state.frame;await state.replay.load(state.current,records,token);if(token!==state.token)return;state.replay.setFrame(requestedFrame);')
s=s.replace('const x=54+state.frame/Math.max(1,max)*826;', 'const x=(state.chart?.left||54)+state.frame/Math.max(1,max)*(state.chart?.plotWidth||826);')
s=s.replace('const width=900,height=300,L=54,R=20,T=24,B=42,', 'const width=Math.max(300,$(\'error-chart\').clientWidth),height=innerWidth<800?270:300,L=44,R=18,T=24,B=42,')
s=s.replace('let svg=`<svg viewBox=', 'state.chart={left:L,plotWidth:width-L-R};let svg=`<svg viewBox=')
s=s.replace('${L+arms.indexOf(rec.case.arm)*180}', '${L+arms.indexOf(rec.case.arm)*(width-L-R)/3}').replace('${rec.case.arm} · 结束 ${fmt(rec.case.simulation_s)}s', '${rec.case.arm} ${fmt(rec.case.simulation_s,1)}s')
s=s.replace('三组轨迹 SHA-256 已核对','三组回放载荷 SHA-256 已核对').replace("metric('物理 / 原QP审计'", "metric('物理 / 原行与备份'")
s=s.replace('function renderEvidence(){const d=state.detail;if(!d)return;',"function renderEvidence(){const d=state.detail;if(!d)return;")
s=s.replace("$('events').replaceChildren();\n const ev=", "$('events').replaceChildren();\n if(state.current.bad_physical_intervals.length){const k=state.current.bad_physical_intervals[0],b=document.createElement('button');b.textContent='跳到首次物理安全失败区间 '+fmt(k*.02)+'–'+fmt((k+1)*.02)+' s';b.onclick=()=>seek(k+1);$('events').append(b);}\n if(d.original_feasibility_audit.original_row_violating_cycles.length){const k=d.original_feasibility_audit.original_row_violating_cycles[0],b=document.createElement('button');b.textContent='跳到首次原QP行违规区间 '+fmt(k*.02)+' s';b.onclick=()=>seek(k+1);$('events').append(b);}\n const ev=")
s=s.replace("$('legacy-link').href=c.legacy_url;", "$('legacy-link').href=c.legacy_url;")
s=s.replace("raw_bundle:c.raw_evidence_sha256,source_archive:c.source_archive", "raw_bundle:c.raw_evidence_sha256,freeze_manifest:c.arm==='C'?'evidence/c-FORMAL_FREEZE.json':'evidence/ab-FORMAL_FREEZE.json',source_archive:c.source_archive")
s=s.replace("window.study900={get index()", "new ResizeObserver(()=>{if(state.records)renderCurves();}).observe($('error-chart'));\n window.study900={get index()")
edit('docs/liuqp900/app.js',s)
s=text('docs/liuqp900/replay.js');s=s.replace('this.pause();this.loaded=false;this.case=c;', 'const changedCase=this.case?.case_id!==c.case_id;this.pause();this.loaded=false;this.case=c;');s=s.replace('this.setFrame(0);this.resetCamera();this.setView(this.view);','this.setFrame(0);if(changedCase)this.resetCamera();this.setView(this.view);');edit('docs/liuqp900/replay.js',s)
s=text('docs/liuqp900/style.css');s+='\ntd button.warn{background:#f6edd4;color:#8a651f}td button.bad{background:#f8e7e1;color:#a03a32}\n';edit('docs/liuqp900/style.css',s)
put(W/'.local/liuqp900/text-updates.json',json.dumps(updates,ensure_ascii=False));print(json.dumps(dict(text_updates=len(updates),new_documents=len(new),legacy_q_difference=qmax,legacy_target_difference=tmax)))
