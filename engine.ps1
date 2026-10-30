$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_MODELS = Join-Path $projectRoot '.runtime\models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_CONTEXT_LENGTH = '8192'
$env:OLLAMA_NUM_PARALLEL = '1'
$enginePath = Join-Path $projectRoot '.runtime\ollama\ollama.exe'

function Test-DiceEngine {
    try {
        $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/tags' -TimeoutSec 3
        return $true
    } catch { return $false }
}

function Start-DiceEngine {
    if (Test-DiceEngine) { return }
    if (-not (Test-Path -LiteralPath $enginePath)) {
        throw 'Local engine is missing. Run setup.ps1 first.'
    }
    $null = Start-Process -FilePath $enginePath -ArgumentList 'serve' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectRoot '.runtime\engine-out.log') -RedirectStandardError (Join-Path $projectRoot '.runtime\engine-error.log')
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if (Test-DiceEngine) { return }
        Start-Sleep -Milliseconds 500
    }
    throw 'Local engine did not start. See .runtime\engine-error.log.'
}
