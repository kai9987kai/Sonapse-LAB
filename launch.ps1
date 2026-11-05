param([int]$Port = 8765)
Set-Location -LiteralPath $PSScriptRoot
python -m synapse serve --port $Port
