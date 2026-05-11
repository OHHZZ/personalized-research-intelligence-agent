$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  $python = $null
  if ($env:RESEARCH_INTEL_PYTHON) {
    $python = $env:RESEARCH_INTEL_PYTHON
  }
  elseif (Test-Path ".\.venv\Scripts\python.exe") {
    $python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
  }
  elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = (Get-Command python).Source
  }
  if (-not $python) {
    throw "No usable python interpreter found. Create .venv or ensure python is on PATH."
  }

  $env:PYTHONPATH = "src"
  & $python -m research_intel.cli --root . serve-web --host 127.0.0.1 --port 8765
}
finally {
  Pop-Location
}
