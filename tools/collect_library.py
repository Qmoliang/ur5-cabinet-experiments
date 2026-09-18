"""Copy explicitly selected research documents without changing their bytes."""
from pathlib import Path
import json,hashlib,shutil,re
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
BASE='liu_qp_reproduction/ur5_liuqp_iris_scenes/'
entries=[]
def add(id,group,title,path,note):
    src=WORK/path
    assert src.is_file(),src
    dst=ROOT/'docs/library/files'/id/src.name
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    entries.append(dict(id=id,group=group,title=title,note=note,filename=src.name,format=src.suffix[1:],source=path,file='files/'+id+'/'+src.name,sha256=hashlib.sha256(src.read_bytes()).hexdigest(),bytes=src.stat().st_size))
add('ch01','chapters','第一性原理与问题定义',BASE+'chapters/01_第一性原理与问题定义.md','现存第一章 Markdown，单独保留；不替代用户的第一章 Word。')
add('ch02','chapters','第二章 · 椭球型 LiuQP 数学推导',BASE+'chapters/02_椭球型LiuQP数学推导.md','从原始 LiuQP 到椭球最近点、分离平面和三状态 QP。')
add('ch03','chapters','第三章 · 已知体积实验代码阅读指南',BASE+'chapters/03_环境已知体积实验代码阅读指南.md','自上而下的文件职责、调用关系与公式对应；对应早期 8 盒、368 代理实验。')
add('math-spec','chapters','增量深度点云统一数学规格与实验协议',BASE+'UR5_incremental_depth_LiuQP_ellipsoid_VCC_math_spec.md','历史方法与协议规格，包含当时的研究假设；不是当前实验通过全部门槛的声明。')
add('v42','historical','4.2h · 正式结果与观看方式',BASE+'V4_2H_FORMAL_RESULTS_AND_USAGE.md','历史严格目标实验记录。')
add('v43','historical','4.3 · 不重复增厚核心椭球',BASE+'V4_3_NO_CORE_REINFLATION_FORMAL_RESULTS.md','历史椭球表面核心与独立不确定性表示。')
add('v44','historical','4.4 · 自适应不可约球基线',BASE+'V4_4_ADAPTIVE_IRREDUNDANT_SPHERE_RESULTS.md','历史自适应球构造与结果边界。')
add('v43-v44-evidence','historical','4.3／4.4 · 冻结有限证据汇总',BASE+'formal_results/final_two_camera/v4_3_v4_4_finite_evidence_20260901/FINITE_EVIDENCE_SUMMARY.md','两条闭环轨迹的原始证据归档。')
add('v4-v5-audit','historical','第四、第五章 · 实验版本证据审计',BASE+'plan/review/chapter-04-05-version-evidence-audit.md','区分早期 v5.1 重复对照、v5.2 与 v5.3；这份是审计记录，不冒充未找到的第四、五章成稿。')
add('v5-view','historical','早期 v5 · 观看与复现记录',BASE+'FORMAL_V5_VIEWING_AND_REPRODUCTION.md','原文标注的历史 v5 记录，保留其版本说明。')
add('v51-support','historical','后续实验 5.1 · 连续测量支持体说明','LiuQP_experiment_5_1/README.md','独立的 LiuQP_experiment_5_1 项目，与早期 v5.1 五次重复对照不同。')
add('v51-acceptance','historical','后续实验 5.1 · 达标版本验收报告','LiuQP_experiment_5_1/docs/达标版本验收报告.md','记录连续体积椭球、0.030714 mm 末端误差及感知限制。')
add('v51-protocol','historical','后续实验 5.1 · 连续占据支持体协议','LiuQP_experiment_5_1/docs/轮2连续占据支持体协议.md','连续支持体几何语义和对照约束。')
add('v51-comparison','historical','后续实验 5.1 · 对照结果报告','LiuQP_experiment_5_1/comparisons/结果报告.md','该历史项目的对照记录。')
add('exp07-protocol','experiment7','实验 7 · 实验协议',BASE+'plan/experiment-07-protocol.md','柜子与笼子场景的历史实验设计。')
add('exp07-progress','experiment7','实验 7 · 进展记录',BASE+'plan/experiment-07-progress.md','保留原始阶段状态与结论限制。')
add('et30-diagnosis','experiment7','ET30 · 末端停滞诊断',BASE+'formal_results/experiment_07/terminal_diagnosis/ET30_terminal_diagnosis.md','到目标前最后距离的诊断记录。')
add('exp07-assets','experiment7','官方素材、MVT 与不确定性审查',BASE+'formal_results/experiment_07/terminal_diagnosis/official_assets_mvt_uncertainty_review.md','场景风格、索引作用与包络尺度的历史分析。')
for v in ['3','4']:
    add('exp07-'+v+'-protocol','experiment7','实验 7.'+v+' · 协议',BASE+'plan/experiment-07-'+v+'-protocol.md','对应这次历史运行的协议。')
    add('exp07-'+v+'-math','experiment7','实验 7.'+v+' · 数学与代码',BASE+'formal_results/experiment_07/development_07_'+v+'/MATH_AND_CODE.md','该阶段公式与实现的对照。')
    add('exp07-'+v+'-report','experiment7','实验 7.'+v+' · 结果报告',BASE+'formal_results/experiment_07/development_07_'+v+'/REPORT.md','该阶段保存的运行结果。')
add('exp07-frozen','experiment7','实验 7.4 · 冻结地图诊断',BASE+'plan/experiment-07-4-frozen-map-diagnostic.md','隔离地图与控制因素的诊断方案。')
add('neo-known','neo','NEO · 已知环境椭球对照','LiuQP_known_volume/external_neo/REPORT.md','历史本地 NEO 适配实验，不等同于原论文完整复现。')
add('neo-sphere','neo','NEO · 已知环境球包络对照','LiuQP_known_volume/external_neo/sphere_study/REPORT.md','球包络与影响距离设置的历史对照。')
add('neo-online','neo','NEO · 在线自适应球／椭球对照',BASE+'external_neo_adaptive/REPORT.md','独立闭环观测与求解结果。')
add('neo-ablation','neo','NEO · 去可操作度目标项消融',BASE+'external_neo_adaptive/ablation_no_manip/REPORT.md','停滞原因诊断及完整运行结果。')
add('current-overview','current','当前实验 · 英文总览','Cabinet_Experiments/rerun_grounded_20260917/README.md','2026-09-17 落地柜子重跑：网页 19 组结果对应的完整本地目录说明。')
add('current-results','current','当前实验 · 19 组结果总表','Cabinet_Experiments/rerun_grounded_20260917/实验结果总表.md','与当前网页演示同一轮实验。')
add('current-viewer','current','当前实验 · 如何查看球和椭球','Cabinet_Experiments/rerun_grounded_20260917/如何查看球和椭球.md','本地 MuJoCo 观看与快捷键说明；网页提供录制回放。')
groups=[dict(id=k,title=v) for k,v in [('chapters','数学与代码阅读'),('historical','历史 4.x / 5.x 实验'),('experiment7','实验 7 与后续诊断'),('neo','NEO 对照与消融'),('current','当前落地柜子实验')]]
add('chapter1-word','chapters','LiuQP 数学理论讲义 · 英文扩展版（Word）','liu_qp_reproduction/output/documents/LiuQP_Mathematical_Derivation_and_Theory_Guide_English_Expanded.docx','用户指定的英文扩展版 Word 原件，保留原始排版与公式，可下载阅读。')
entries.insert(0,entries.pop())
for entry in entries:
    if entry['format']!='md':continue
    source=WORK/entry['source']
    entry['assets']={}
    for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',source.read_text(encoding='utf-8')):
        image=(source.parent/link).resolve()
        if image.is_file() and image.suffix.lower() in ['.png','.jpg','.jpeg','.webp','.svg']:
            target=(ROOT/'docs/library'/entry['file']).parent/link
            target=target.resolve()
            assert target.is_relative_to((ROOT/'docs/library').resolve())
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(image,target)
            entry['assets'][link]=target.relative_to(ROOT/'docs/library').as_posix()
(ROOT/'docs/library/manifest.json').write_text(json.dumps(dict(groups=groups,documents=entries),ensure_ascii=False,indent=2),encoding='utf-8')
print('Collected',len(entries),'documents')

