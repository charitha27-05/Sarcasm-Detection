Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting Sarcasm Detection AI Server locally" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Checking / Installing dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt
Write-Host ""
Write-Host "Starting Flask Application on http://localhost:5050 ..." -ForegroundColor Green
python app.py
