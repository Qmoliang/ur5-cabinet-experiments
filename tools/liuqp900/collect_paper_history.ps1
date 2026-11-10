param(
  [Parameter(Mandatory=$true)][string]$ArchiveRoot,
  [string]$RepositoryRoot = (Get-Location).Path
)
$ErrorActionPreference = 'Stop'
$destination = Join-Path $RepositoryRoot 'docs/library/files/liuqp900/history-evidence'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$studies = @(
  @('H01','execution-ablation','liuqp_execution_ablation_20261105','EXECUTION_ABLATION_REPORT.txt','FINAL_REPORT.json',8),
  @('H02','lowdim-pilot','liuqp_lowdim_pilot_20261106','FINAL_REPORT.txt','FINAL_REPORT.json',33),
  @('H03','release','liuqp_release_study_20261106','FINAL_REPORT.txt','RESULTS.json',38),
  @('H04','goal-entry','liuqp_goal_entry_study_20261106','FINAL_REPORT.txt','RESULTS.json',49),
  @('H05','reheading','liuqp_reheading_study_20261107','FINAL_REPORT.txt','RESULTS.json',15),
  @('H06','entry-certificate','liuqp_entry_certificate_20261107','FINAL_REPORT.txt','RESULTS.json',19),
  @('H07','progress-acceptance','liuqp_progress_acceptance_20261107','FINAL_REPORT.txt','RESULTS.json',62),
  @('H08','holdout','liuqp_holdout_20261108','FINAL_REPORT.txt','RESULTS.json',24),
  @('H09','rank-init','liuqp_rank_init_20261108','FINAL_REPORT.txt','RESULTS.json',54),
  @('H10','t0-continuation','liuqp_t0_continuation_20261109','FINAL_REPORT.txt','RESULTS.json',156),
  @('H11','t0-causal','liuqp_t0_causal_20261109','FINAL_REPORT.txt','RESULTS.json',131),
  @('H12','continuation-exit','liuqp_continuation_exit_20261109','FINAL_REPORT.txt','RESULTS.json',151)
)
$records = [System.Collections.Generic.List[object]]::new()
$sha = [System.Security.Cryptography.SHA256]::Create()
function Copy-Evidence($relativeSource,$name,$studyId,$kind,$count) {
  $source = Join-Path $ArchiveRoot $relativeSource
  $bytes = [System.IO.File]::ReadAllBytes($source)
  $hash = [System.Convert]::ToHexString($sha.ComputeHash($bytes)).ToLowerInvariant()
  [System.IO.File]::WriteAllBytes((Join-Path $destination $name),$bytes)
  $records.Add([ordered]@{study=$studyId;kind=$kind;source=$relativeSource.Replace('\','/');file=$name;bytes=$bytes.Length;sha256=$hash;source_table_entries=$count})
}
foreach($s in $studies){
  Copy-Evidence "$($s[2])/delivery/$($s[3])" "$($s[1])-report.txt" $s[0] 'original_report' $s[5]
  Copy-Evidence "$($s[2])/delivery/$($s[4])" "$($s[1])-results.json" $s[0] 'original_results' $s[5]
}
Copy-Evidence 'liuqp_paired300_20261109/delivery/FINAL_REPORT.txt' 'paired300-report.txt' 'AB900' 'original_report' 600
Copy-Evidence 'liuqp_pure300_20261110/delivery/FINAL_REPORT.txt' 'pure300-report.txt' 'C900' 'original_report' 300
Copy-Evidence 'sphere_r17_pair_20261104/delivery/CLOSEOUT_REPORT.json' 'sphere-r17-report.json' 'historical-sphere' 'original_results' 300
Copy-Evidence 'gcs_path_pilot_20261104/delivery/PILOT_REPORT.txt' 'gcs-pilot-report.txt' 'historical-gcs' 'original_report' 48
Copy-Evidence 'iris_engineering_alignment_20261105/delivery/ALIGNMENT_REPORT.txt' 'iris-alignment-report.txt' 'historical-iris' 'original_report' 48
$manifest = [ordered]@{
  frozen_comparison_commit='0f402010bb8335df661fdace8745106b7ee5aede'
  scope='Copies of original reports and result tables, not new runs; historical success labels remain unchanged.'
  counting='Source-table entries include development, diagnostic replays and initialization/startup failures. Do not sum as independent samples or add final subsets twice.'
  files=@($records)
}
[System.IO.File]::WriteAllText((Join-Path $destination 'manifest.json'),($manifest | ConvertTo-Json -Depth 8),[System.Text.UTF8Encoding]::new($false))
Write-Output "Copied $($records.Count) historical source files without changing originals."
