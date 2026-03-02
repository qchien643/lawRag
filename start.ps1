<#
.SYNOPSIS
    Start all services locally (Qdrant via Docker, Python services via venv).

.DESCRIPTION
    1. Starts Qdrant container (docker) — skips if Docker not available
    2. Activates Python venv
    3. Starts LLM Service (port 8001)
    4. Starts RAG Service (port 8000)
    5. Starts Frontend dev server (port 5173)

.EXAMPLE
    .\start.ps1
#>

$ErrorActionPreference = "Continue"
$ROOT = $PSScriptRoot

Write-Host "=== LawRAG Startup ===" -ForegroundColor Cyan

# --- 1. Qdrant (Docker) ---
Write-Host "`n[1/5] Starting Qdrant..." -ForegroundColor Yellow
$dockerAvailable = $false
try {
    docker info *>$null
    $dockerAvailable = $true
} catch {
    $dockerAvailable = $false
}

if ($dockerAvailable) {
    # Try to start existing container first
    $existing = docker ps -a --filter "name=lawrag-qdrant" --format "{{.Names}}" 2>$null
    if ($existing -eq "lawrag-qdrant") {
        docker start lawrag-qdrant *>$null
        Write-Host "  Qdrant container restarted" -ForegroundColor Green
    } else {
        docker run -d --name lawrag-qdrant -p 6333:6333 -p 6334:6334 `
            -v lawrag_qdrant_data:/qdrant/storage `
            qdrant/qdrant:latest *>$null
        Write-Host "  Qdrant container created and started" -ForegroundColor Green
    }
    Write-Host "  Qdrant ready at http://localhost:6333" -ForegroundColor Green
} else {
    Write-Host "  WARNING: Docker not available! Please start Docker Desktop." -ForegroundColor Red
    Write-Host "  Assuming Qdrant is already running at localhost:6333..." -ForegroundColor Gray
}

# --- 2. Activate venv ---
Write-Host "`n[2/5] Activating venv..." -ForegroundColor Yellow
$venvActivate = Join-Path $ROOT ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
    Write-Host "  Venv activated" -ForegroundColor Green
} else {
    Write-Host "  ERROR: .venv not found at $ROOT\.venv" -ForegroundColor Red
    Write-Host "  Create it with: python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt" -ForegroundColor Gray
    exit 1
}

# --- 3. LLM Service ---
Write-Host "`n[3/5] Starting LLM Service (port 8001)..." -ForegroundColor Yellow
$llmProcess = Start-Process -FilePath "python" -ArgumentList "llm_api.py" `
    -WorkingDirectory (Join-Path $ROOT "llm_service") `
    -PassThru -NoNewWindow
Write-Host "  LLM Service started (PID: $($llmProcess.Id))" -ForegroundColor Green

# --- 4. RAG Service ---
Write-Host "`n[4/5] Starting RAG Service (port 8000)..." -ForegroundColor Yellow
$ragProcess = Start-Process -FilePath "python" -ArgumentList "rag_api.py" `
    -WorkingDirectory (Join-Path $ROOT "rag_service") `
    -PassThru -NoNewWindow
Write-Host "  RAG Service started (PID: $($ragProcess.Id))" -ForegroundColor Green

# --- 5. Frontend ---
Write-Host "`n[5/5] Starting Frontend (port 5173)..." -ForegroundColor Yellow
$feProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "npm run dev" `
    -WorkingDirectory (Join-Path $ROOT "frontend") `
    -PassThru -NoNewWindow
Write-Host "  Frontend started (PID: $($feProcess.Id))" -ForegroundColor Green

# --- Done ---
Write-Host "`n=== All services started ===" -ForegroundColor Cyan
Write-Host "  Frontend:    http://localhost:5173" -ForegroundColor White
Write-Host "  RAG Service: http://localhost:8000" -ForegroundColor White
Write-Host "  LLM Service: http://localhost:8001" -ForegroundColor White
Write-Host "  Qdrant:      http://localhost:6333" -ForegroundColor White
Write-Host "`nPress Enter to stop all services..." -ForegroundColor Gray

# Wait for user input
Read-Host | Out-Null

# Cleanup
Write-Host "`nStopping services..." -ForegroundColor Yellow
@($llmProcess, $ragProcess, $feProcess) | ForEach-Object {
    if ($_ -and -not $_.HasExited) {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  Stopped PID $($_.Id)" -ForegroundColor Gray
    }
}
Write-Host "Services stopped." -ForegroundColor Green
if ($dockerAvailable) {
    Write-Host "Qdrant container still running (use 'docker stop lawrag-qdrant' to stop)." -ForegroundColor Gray
}
