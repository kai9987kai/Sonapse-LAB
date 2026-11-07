param([int]$Port = 8765, [int]$TotalNeurons = 4000)
Set-Location -LiteralPath $PSScriptRoot
python -m synapse --total-neurons $TotalNeurons serve --port $Port
