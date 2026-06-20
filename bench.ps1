# Benchmark de tok/s no hardware real usando llama-bench (do mesmo build do app).
# Compara CPU puro (ngl=0) vs offload parcial na Vega (Vulkan).
# Uso:  .\bench.ps1 -Model "D:\...\Qwen3.5-9B-Q6_K.gguf"
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [int]$Threads = 6,
  [int]$Ctx = 512,
  [int]$NGen = 128
)

$bin = Join-Path $PSScriptRoot "src-tauri\binaries"
$bench = Join-Path $bin "llama-bench.exe"
if (-not (Test-Path $bench)) { Write-Error "llama-bench.exe nao encontrado em $bin"; exit 1 }
if (-not (Test-Path $Model)) { Write-Error "Modelo nao encontrado: $Model"; exit 1 }

Push-Location $bin
Write-Host "`n=== CPU puro (ngl=0, $Threads threads) ===" -ForegroundColor Cyan
& $bench -m $Model -t $Threads -ngl 0 -p $Ctx -n $NGen -fa 1

Write-Host "`n=== Offload Vulkan (ngl=99 -> Vega) ===" -ForegroundColor Yellow
& $bench -m $Model -t $Threads -ngl 99 -p $Ctx -n $NGen -fa 1
Pop-Location

Write-Host "`nLegenda: pp = velocidade de prompt | tg = velocidade de geracao (tok/s)" -ForegroundColor Gray
