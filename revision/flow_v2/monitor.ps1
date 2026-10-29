$Host.UI.RawUI.WindowTitle = 'LITE-XIDS ortak akis ve maliyet duyarli egitim - CANLI'
$log = Join-Path $PSScriptRoot 'logs\live.log'
Write-Host 'Gercek calisma gunlugu. PASS/FAIL ve engellenen asamalar buraya yazilir.' -ForegroundColor Cyan
Write-Host "Gunluk: $log"
while (-not (Test-Path -LiteralPath $log)) { Start-Sleep -Seconds 1 }
Get-Content -LiteralPath $log -Encoding UTF8 -Tail 100 -Wait
