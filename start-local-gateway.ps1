$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $root "llm-gateway-local.exe"
$config = Join-Path $root "config.local.yaml"
$outLog = Join-Path $root "gateway.local.out.log"
$errLog = Join-Path $root "gateway.local.err.log"

if (-not (Test-Path $exe)) {
    throw "Missing gateway executable: $exe"
}
if (-not (Test-Path $config)) {
    throw "Missing gateway config: $config"
}

$existing = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8080 -ErrorAction SilentlyContinue
if ($existing) {
    return
}

$env:CONFIG_PATH = $config
$env:DEEPSEEK_KEY_1 = [Environment]::GetEnvironmentVariable("DEEPSEEK_KEY_1", "User")
$env:LG_LOCAL_VIRTUAL_KEY = [Environment]::GetEnvironmentVariable("LG_LOCAL_VIRTUAL_KEY", "User")

if (-not $env:DEEPSEEK_KEY_1) {
    throw "User environment variable DEEPSEEK_KEY_1 is not set"
}
if (-not $env:LG_LOCAL_VIRTUAL_KEY) {
    throw "User environment variable LG_LOCAL_VIRTUAL_KEY is not set"
}

Start-Process `
    -FilePath $exe `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog
