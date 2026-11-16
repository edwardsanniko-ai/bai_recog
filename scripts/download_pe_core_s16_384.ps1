$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$revision = '255fa76e26fdf8daa47e0d607da46b7051879045'
$expectedSize = 95147760
$expectedSha256 = 'b07ccd4489dcef7e1d2f3878a8232eb2457ba5b2d2086bd1512def4644cbfdda'
$destinationDirectory = Join-Path $repo "hf_cache\pe_core_s16_384\$revision"
$destination = Join-Path $destinationDirectory 'model.safetensors'
$url = "https://hf-mirror.com/timm/vit_pe_core_small_patch16_384.fb/resolve/$revision/model.safetensors"

New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null

if (Test-Path -LiteralPath $destination) {
    $existing = Get-Item -LiteralPath $destination
    if ($existing.Length -gt $expectedSize) {
        throw "Existing file is larger than the locked official size: $($existing.Length)"
    }
    if ($existing.Length -eq $expectedSize) {
        $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
        if ($existingHash -eq $expectedSha256) {
            Write-Host 'PE-Core weights already downloaded and SHA256 verified.'
            Write-Host "Weights: $destination"
            exit 0
        }
        throw "Existing complete-size file has the wrong SHA256: $existingHash"
    }
    Write-Host "Resuming partial download at $($existing.Length) / $expectedSize bytes ..."
} else {
    Write-Host "Downloading pinned PE-Core weights ($expectedSize bytes) ..."
}

& curl.exe -L --fail --retry 12 --retry-all-errors --retry-delay 3 --connect-timeout 30 `
    --continue-at - --output $destination $url
if ($LASTEXITCODE -ne 0) {
    throw "curl failed with exit code $LASTEXITCODE. Run this script again to resume."
}

$downloaded = Get-Item -LiteralPath $destination
if ($downloaded.Length -ne $expectedSize) {
    throw "Downloaded size $($downloaded.Length) does not match locked size $expectedSize. Run again to resume."
}
$downloadedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
if ($downloadedHash -ne $expectedSha256) {
    throw "Downloaded SHA256 does not match the official repository metadata: $downloadedHash"
}

Write-Host 'Download complete and SHA256 verified: True'
Write-Host "Revision: $revision"
Write-Host "SHA256: $downloadedHash"
Write-Host "Weights: $destination"
