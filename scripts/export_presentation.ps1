param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
$deckPath = Join-Path $ProjectRoot 'presentation\credit_default_final.pptx'
$previewPath = Join-Path $ProjectRoot 'presentation\preview'
$pdfPath = Join-Path $ProjectRoot 'presentation\credit_default_final.pdf'
if (-not (Test-Path -LiteralPath $deckPath)) { throw "Missing deck: $deckPath" }
New-Item -ItemType Directory -Force -Path $previewPath | Out-Null
$powerPoint = $null
$deck = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $deck = $powerPoint.Presentations.Open($deckPath, -1, 0, 0)
    $deck.Export($previewPath, 'PNG', 1600, 900)
    $deck.SaveAs($pdfPath, 32)
    Write-Output "PowerPoint opened and rendered $($deck.Slides.Count) slides."
} finally {
    if ($null -ne $deck) { $deck.Close() }
    if ($null -ne $powerPoint) { $powerPoint.Quit() }
}
