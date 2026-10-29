# Registers the tt_moc paper-incubation runner (Amendment 6) as ONE scheduled task for the
# current user. Paper only: the runner never places an order.
#
# Triggers
#   - Weekdays at 20:50 and 21:50 local. The script decides only between 15:45 and 15:58
#     New York time; on this UTC+1 PC that is 20:45 local while New York is on EDT and 21:45
#     while it is on EST. The run outside the window only reconciles.
#   - At logon (2 minutes after), so a PC that was off catches up on reconciliation and
#     records any missed day as MISSED instead of leaving a gap.
# Settings
#   - Start as soon as possible after a missed scheduled start.
#   - Only start when a network connection is available; the runner also retries for up to
#     6 minutes if the data download fails.
#   - Wake the PC from sleep for the scheduled runs. Cannot start a PC that is shut down.
#   - Runs on battery, one instance at a time, 15 minute limit.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_incubation_tasks.ps1           # install
#   powershell -ExecutionPolicy Bypass -File scripts\install_incubation_tasks.ps1 -Verify   # check
#   powershell -ExecutionPolicy Bypass -File scripts\install_incubation_tasks.ps1 -Remove   # uninstall

param([switch]$Verify, [switch]$Remove)

$name = "QuantIncubate_tt_moc"
$bat = Join-Path $PSScriptRoot "incubate_tt_moc.bat"
$user = "$env:USERDOMAIN\$env:USERNAME"

if ($Remove) {
    schtasks /delete /tn $name /f
    exit $LASTEXITCODE
}

if (-not $Verify) {
    $start = (Get-Date).ToString("yyyy-MM-dd")
    $weekdays = "<Monday /><Tuesday /><Wednesday /><Thursday /><Friday />"
    $xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>tt_moc paper incubation (Amendment 6). Paper only, never places an order.</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>${start}T20:50:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek><WeeksInterval>1</WeeksInterval><DaysOfWeek>$weekdays</DaysOfWeek></ScheduleByWeek>
    </CalendarTrigger>
    <CalendarTrigger>
      <StartBoundary>${start}T21:50:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek><WeeksInterval>1</WeeksInterval><DaysOfWeek>$weekdays</DaysOfWeek></ScheduleByWeek>
    </CalendarTrigger>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>$user</UserId>
      <Delay>PT2M</Delay>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>$user</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <WakeToRun>true</WakeToRun>
    <ExecutionTimeLimit>PT15M</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>"$bat"</Command>
    </Exec>
  </Actions>
</Task>
"@
    $tmp = Join-Path $env:TEMP "$name.xml"
    $xml | Out-File -FilePath $tmp -Encoding Unicode
    schtasks /create /tn $name /xml $tmp /f
    Remove-Item $tmp
    # Older versions of this script registered two separate tasks; remove them so nothing
    # runs twice.
    foreach ($old in "QuantIncubate_tt_moc_2050", "QuantIncubate_tt_moc_2150") {
        schtasks /query /tn $old 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { schtasks /delete /tn $old /f | Out-Null }
    }
}

# Verify by querying, never by trusting the create call.
$q = schtasks /query /tn $name /fo list /v 2>$null
if ($LASTEXITCODE -eq 0) {
    $next = ($q | Select-String "Next Run Time" | Select-Object -First 1).ToString().Split(":", 2)[1].Trim()
    "{0,-22} INSTALLED, next run {1}" -f $name, $next
} else {
    "{0,-22} ABSENT" -f $name
}
$log = Join-Path (Split-Path $PSScriptRoot) "state\incubation_run.log"
if (Test-Path $log) {
    "last log line: " + (Get-Content $log -Tail 20 | Where-Object { $_ -match "^\[" } | Select-Object -Last 1)
}
