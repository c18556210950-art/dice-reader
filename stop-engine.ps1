$ErrorActionPreference = 'Stop'
$expectedPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '.runtime\ollama\ollama.exe'))
# Only stop the engine that belongs to this project, never other Ollama installs.
Get-CimInstance Win32_Process -Filter "Name = 'ollama.exe'" | Where-Object {
    $_.ExecutablePath -and ([IO.Path]::GetFullPath($_.ExecutablePath) -eq $expectedPath) -and ($_.CommandLine -match '\bserve\b')
} | ForEach-Object { Stop-Process -Id $_.ProcessId }
