$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  $env:PYTHONPATH = "src"
  python -m research_intel.cli run-daily --profile default_user --report latest --source hybrid
}
finally {
  Pop-Location
}
