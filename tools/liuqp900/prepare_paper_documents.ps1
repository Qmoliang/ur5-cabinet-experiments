param([string]$RepositoryRoot=(Get-Location).Path)
$ErrorActionPreference='Stop'
$utf8=[System.Text.UTF8Encoding]::new($false)
$folder=Join-Path $RepositoryRoot 'docs/library/files/liuqp900'
$tables=Get-Content -LiteralPath (Join-Path $RepositoryRoot '.local/paper-tables.json') -Raw | ConvertFrom-Json -AsHashtable
$paperPath=Join-Path $folder 'paper-draft.md'
$paper=[System.IO.File]::ReadAllText($paperPath)
foreach($name in @('outcomes','paired','timing','failures')){
  $pattern='(?s)<!-- TABLE: '+$name+' -->(?:.*?<!-- END TABLE: '+$name+' -->)?'
  $block='<!-- TABLE: '+$name+" -->`n`n"+$tables[$name]+"`n`n<!-- END TABLE: "+$name+' -->'
  $paper=[regex]::Replace($paper,$pattern,[System.Text.RegularExpressions.MatchEvaluator]{param($m) $block})
}
$paper=[regex]::Replace($paper,'(?m)^<!-- (?:END )?TABLE: [a-z]+ -->\r?\n','')
[System.IO.File]::WriteAllText($paperPath,$paper,$utf8)
$history=Get-Content -LiteralPath (Join-Path $folder 'history-evidence/manifest.json') -Raw | ConvertFrom-Json
$sourceLines=@('| ID | 原报告 | 原结果表 |','| --- | --- | --- |')
foreach($group in ($history.files | Where-Object {$_.study -match '^H\d+$'} | Group-Object study | Sort-Object Name)){
  $report=$group.Group | Where-Object kind -eq 'original_report'
  $results=$group.Group | Where-Object kind -eq 'original_results'
  $base='https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/'
  $sourceLines+='| '+$group.Name+' | [报告]('+$base+$report.file+') | [结果JSON]('+$base+$results.file+') |'
}
$ledgerPath=Join-Path $folder 'experiment-ledger.md'
$ledger=[System.IO.File]::ReadAllText($ledgerPath)
$block="<!-- HISTORY_SOURCES -->`n`n"+($sourceLines -join "`n")+"`n`n<!-- END HISTORY_SOURCES -->"
$ledger=[regex]::Replace($ledger,'(?s)<!-- HISTORY_SOURCES -->(?:.*?<!-- END HISTORY_SOURCES -->)?',[System.Text.RegularExpressions.MatchEvaluator]{param($m) $block})
$ledger=[regex]::Replace($ledger,'(?m)^<!-- (?:END )?HISTORY_SOURCES -->\r?\n','')
[System.IO.File]::WriteAllText($ledgerPath,$ledger,$utf8)
$manifestPath=Join-Path $RepositoryRoot 'docs/library/manifest.json'
$manifest=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
$specs=@(
 @('liuqp900-paper','paper-draft.md','论文初稿：执行贡献、控制贡献与完整实验'),
 @('liuqp900-ledger','experiment-ledger.md','实验台账：900、1200与历史尝试'),
 @('liuqp900-explanation','explanation-under-2000.md','2000字以内讲解新版方法'),
 @('liuqp900-tables','experiment-tables.md','冻结900次：逐例重算表格与全部失败')
)
$docs=@();$sha=[System.Security.Cryptography.SHA256]::Create()
foreach($s in $specs){
  $bytes=[System.IO.File]::ReadAllBytes((Join-Path $folder $s[1]))
  $doc=[ordered]@{id=$s[0];title=$s[2];group='liuqp900';format='md';file=('files/liuqp900/'+$s[1]);filename=$s[1];source=('publication/liuqp900/'+$s[1]);sha256=([System.Convert]::ToHexString($sha.ComputeHash($bytes)).ToLowerInvariant());note='基于0f40201冻结代码和900次对照补写；包含全部失败、历史否定结果与证据边界，原结果不改写。';scope_note='1200指新900次控制对照加旧在线RRT300条参照，不是同协议四臂或独立场景数。证明、契约和实验支持分别陈述。'}
  if($s[0] -eq 'liuqp900-paper'){$doc.assets=@{'figures/outcomes.svg'='files/liuqp900/figures/outcomes.svg';'figures/timing.svg'='files/liuqp900/figures/timing.svg'}}
  $docs+=$doc
}
$ids=@($docs | ForEach-Object {$_.id})
$manifest.documents=$docs+@($manifest.documents | Where-Object {$_.id -notin $ids})
[System.IO.File]::WriteAllText($manifestPath,($manifest | ConvertTo-Json -Depth 12)+"`n",$utf8)
$homePath=Join-Path $RepositoryRoot 'docs/liuqp900/index.html'
$pageHtml=[System.IO.File]::ReadAllText($homePath)
if(-not $pageHtml.Contains('liuqp900-paper.html')){$pageHtml=$pageHtml.Replace('<div class="downloads">','<div class="downloads"><a href="../library/pages/liuqp900-paper.html">论文完整初稿 · Markdown与图表</a><a href="../library/pages/liuqp900-ledger.html">实验台账 · 900与1200的计数</a><a href="../library/pages/liuqp900-explanation.html">2000字以内讲解新版</a>')}
[System.IO.File]::WriteAllText($homePath,$pageHtml,$utf8)
$readmePath=Join-Path $RepositoryRoot 'README.md'
$readme=[System.IO.File]::ReadAllText($readmePath)
if(-not $readme.Contains('liuqp900-paper.html')){
 $prefix=@'
## 论文初稿与实验台账（2026-11-10）

[完整论文 Markdown](docs/library/files/liuqp900/paper-draft.md) · [公式与图表在线版](https://qmoliang.github.io/ur5-cabinet-experiments/library/pages/liuqp900-paper.html) · [历史实验台账](docs/library/files/liuqp900/experiment-ledger.md) · [2000字以内说明](docs/library/files/liuqp900/explanation-under-2000.md)

以冻结提交 `0f40201` 为证据基准，拆分执行器和控制方案的增量，公开全部六个B组失败、回归退化、成本及证明边界。900次主对照加旧在线RRT300条可称1200条所选记录，不是1200个独立场景或同协议四臂；更早批次另列。此次仅补文稿、来源与图表，不改控制代码或原结果。

---

'@
 $readme=$prefix+"`n"+$readme
}
[System.IO.File]::WriteAllText($readmePath,$readme,$utf8)
$explanation=[System.IO.File]::ReadAllText((Join-Path $folder 'explanation-under-2000.md'))
$count=([regex]::Replace($explanation,'\s','')).Length
if($count -gt 2000){throw "Explanation exceeds 2000 non-whitespace characters: $count"}
Write-Output "Prepared four paper documents; explanation contains $count non-whitespace characters including heading and markup."
