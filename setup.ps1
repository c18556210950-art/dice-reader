$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
. (Join-Path $PSScriptRoot 'engine.ps1')

# Everything stays in this project; no system-wide installer or startup entry.
$runtimePath = Join-Path $PSScriptRoot '.runtime'
$null = New-Item -ItemType Directory -Force -Path $runtimePath
$null = New-Item -ItemType Directory -Force -Path (Join-Path $runtimePath 'ollama')
if (-not (Test-Path -LiteralPath $enginePath)) {
    $archivePath = Join-Path $runtimePath 'ollama-windows-amd64.zip'
    Write-Host 'Downloading the local engine (about 1.6 GB). Downloads can be resumed.'
    & curl.exe --fail --location --retry 3 --connect-timeout 30 --continue-at - --output $archivePath 'https://ollama.com/download/ollama-windows-amd64.zip'
    if ($LASTEXITCODE -ne 0) { throw 'Engine download failed. Run setup.ps1 again to resume.' }
    Expand-Archive -LiteralPath $archivePath -DestinationPath (Join-Path $runtimePath 'ollama') -Force
}

python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
Start-DiceEngine
Write-Host 'Downloading Qwen3-VL 4B (about 3.3 GB). Photos are never sent online.'
python download_hf_model.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed. Run setup.ps1 again to resume.' }
Write-Host 'Ready. Run start.ps1. Future recognition works offline.'
