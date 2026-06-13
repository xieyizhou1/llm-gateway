$connections = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8080 -ErrorAction SilentlyContinue
$pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($pidValue in $pids) {
    $process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($process -and $process.Path -like "*llm-gateway-local.exe") {
        Stop-Process -Id $pidValue -Force
    }
}
