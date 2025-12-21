# GitHub Push Script
# This script helps you push to GitHub using a Personal Access Token

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Push Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if already pushed
$status = git status
if ($status -match "Your branch is up to date") {
    Write-Host "Repository is already up to date!" -ForegroundColor Green
    exit
}

Write-Host "To push to GitHub, you need a Personal Access Token." -ForegroundColor Yellow
Write-Host ""
Write-Host "Option 1: Use GitHub CLI (if authenticated)" -ForegroundColor Cyan
Write-Host "  Run: gh auth login" -ForegroundColor White
Write-Host "  Then: git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "Option 2: Use Personal Access Token" -ForegroundColor Cyan
Write-Host "  1. Create token at: https://github.com/settings/tokens" -ForegroundColor White
Write-Host "  2. Select 'repo' scope" -ForegroundColor White
Write-Host "  3. Copy the token" -ForegroundColor White
Write-Host "  4. Run: git push -u origin main" -ForegroundColor White
Write-Host "  5. Username: muinul-official" -ForegroundColor White
Write-Host "  6. Password: [paste your token]" -ForegroundColor White
Write-Host ""
Write-Host "Current repository status:" -ForegroundColor Cyan
git status

