# Block until no python.exe command line matches -Pattern. Used to serialise GPU jobs.
#
# WHY THIS EXISTS. `tasklist /V` does not expose command lines and `pgrep` does not see Windows
# processes from Git Bash, so both silently report "nothing running" and let a second job launch on
# top of a live one. That is exactly how two feature accumulations ended up sharing the card
# tonight, and how a chain stage started while the graph measurement was still running. CIM's
# Win32_Process DOES expose CommandLine, and is verified to match a live process.
param([Parameter(Mandatory=$true)][string]$Pattern)
while ($true) {
  $n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -match $Pattern }).Count
  if ($n -eq 0) { break }
  Start-Sleep -Seconds 30
}
