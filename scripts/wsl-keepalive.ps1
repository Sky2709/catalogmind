<#
.SYNOPSIS
    Keeps the WSL2 VM (and therefore the Docker stack) from shutting down when idle.

.DESCRIPTION
    WSL2 stops an idle VM after roughly 60 seconds. That takes the whole datastore
    stack with it, and because `restart: unless-stopped` brings the containers back on
    the next command, the symptom is confusing rather than obvious: containers are
    perpetually "Up N seconds" with RestartCount=0, and Weaviate redoes its ~25 second
    raft leader election every time, so readiness looks intermittent for no reason.

    There is no configuration key for this in WSL 2.7.3 - `vmIdleTimeout` is rejected as
    an unknown key under both [wsl2] and [experimental]. Holding one WSL session open
    from Windows is the reliable workaround.

    Run this once per Windows login. It launches a hidden background process; closing
    your terminal will not kill it.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\wsl-keepalive.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\wsl-keepalive.ps1 -Stop
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [string]$Distro = "Ubuntu"
)

$marker = "catalogmind-wsl-keepalive"

function Get-KeepaliveProcesses {
    Get-CimInstance Win32_Process -Filter "Name = 'wsl.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $marker }
}

if ($Stop) {
    $procs = Get-KeepaliveProcesses
    if (-not $procs) { Write-Host "No keepalive running."; return }
    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Write-Host "Stopped $($procs.Count) keepalive process(es)."
    return
}

if (Get-KeepaliveProcesses) {
    Write-Host "Keepalive already running."
    return
}

# `sleep infinity` holds the session open indefinitely at effectively zero cost. The
# marker string is only there so this script can find and stop its own process later.
Start-Process -FilePath "wsl.exe" `
    -ArgumentList @("-d", $Distro, "--", "sh", "-c", "# $marker`nexec sleep infinity") `
    -WindowStyle Hidden

Start-Sleep -Seconds 2
if (Get-KeepaliveProcesses) {
    Write-Host "Keepalive started. The WSL VM and Docker stack will now stay up."
    Write-Host "Stop it with:  scripts\wsl-keepalive.ps1 -Stop"
} else {
    Write-Warning "Keepalive did not start."
}
