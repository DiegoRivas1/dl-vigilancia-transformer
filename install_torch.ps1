python -m venv .venv
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
python -m pip install --upgrade pip

$hasGpu = (Get-Command nvidia-smi -ErrorAction SilentlyContinue) -ne $null
if ($hasGpu) {
    Write-Host "GPU detectada, instalando PyTorch con CUDA..." -ForegroundColor Cyan
    pip install torch --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "Sin GPU, instalando PyTorch CPU..." -ForegroundColor Yellow
    pip install torch
}

pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"