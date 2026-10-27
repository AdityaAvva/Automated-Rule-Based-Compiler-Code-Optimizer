$ErrorActionPreference = "Stop"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Error "Python was not found on PATH. Install Python 3.10+ or add it to PATH."
    exit 1
}

$serverPath = Join-Path $PSScriptRoot "server.py"
Write-Host "Starting the local Compiler Optimizer dashboard..."
Write-Host "Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop."
& $pythonCommand.Source $serverPath
exit $LASTEXITCODE
