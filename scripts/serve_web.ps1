$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  $env:PYTHONPATH = "src"
  python -m research_intel.cli serve-web --host 127.0.0.1 --port 8765
}
finally {
  Pop-Location
}

