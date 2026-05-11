$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  $containerName = "research-intel-pgvector"
  $password = "postgres"
  $database = "research_intel"
  $image = if ($env:PGVECTOR_IMAGE) { $env:PGVECTOR_IMAGE } else { "pgvector/pgvector:pg17" }

  $existing = docker ps -a --filter "name=$containerName" --format "{{.Names}}"
  if ($existing -contains $containerName) {
    docker start $containerName | Out-Null
  }
  else {
    try {
      docker run -d `
        --name $containerName `
        -e POSTGRES_PASSWORD=$password `
        -e POSTGRES_DB=$database `
        -p 5432:5432 `
        $image | Out-Null
    }
    catch {
      throw "Unable to start pgvector container with image '$image'. Check Docker registry/network access or set PGVECTOR_IMAGE to a reachable mirror. Original error: $($_.Exception.Message)"
    }
  }

  $running = docker ps --filter "name=$containerName" --filter "status=running" --format "{{.Names}}"
  if ($running -notcontains $containerName) {
    throw "pgvector container '$containerName' is not running after startup."
  }

  Write-Host "pgvector container is running on localhost:5432"
  Write-Host "Suggested PGVECTOR_DSN=postgresql://postgres:$password@localhost:5432/$database"
}
finally {
  Pop-Location
}
