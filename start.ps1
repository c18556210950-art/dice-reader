$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
. (Join-Path $PSScriptRoot 'engine.ps1')
if (Test-Path -LiteralPath $enginePath) { Start-DiceEngine }
Write-Host 'Open http://127.0.0.1:8765 in your browser. Press Ctrl+C to stop the page service.'
python app.py
