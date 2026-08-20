# Two GENUINELY different vocal tracts - the honest diarization test that the
# single-voice fixture could not provide. Speaker embedders model the vocal tract,
# so prosody variation of one voice is (correctly) NOT a second speaker.
param([string]$OutDir = "$PSScriptRoot\fixtures_2spk")

Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$VOICE_A = 'Microsoft Helena Desktop'   # es-ES female
$VOICE_B = 'Microsoft Zira Desktop'     # en-US female - different vocal tract

$turns = @(
  @{ voice = $VOICE_A; spk = 1; text = 'Buenos dias, vamos a revisar el avance del proyecto de facturacion.' }
  @{ voice = $VOICE_B; spk = 2; text = 'Por mi parte el modulo de validacion ya quedo terminado ayer.' }
  @{ voice = $VOICE_A; spk = 1; text = 'Y como vamos con la integracion al sistema del banco?' }
  @{ voice = $VOICE_B; spk = 2; text = 'Ahi tenemos un bloqueo con las credenciales de pruebas.' }
  @{ voice = $VOICE_A; spk = 1; text = 'Entendido, yo lo escalo con el contacto comercial esta semana.' }
  @{ voice = $VOICE_B; spk = 2; text = 'Perfecto, mientras avanzo con el simulador y las pruebas unitarias.' }
  @{ voice = $VOICE_A; spk = 1; text = 'El presupuesto aprobado es de cuarenta y cinco mil pesos.' }
  @{ voice = $VOICE_B; spk = 2; text = 'De acuerdo, lo entrego para el proximo martes sin falta.' }
)

$manifest = @()
$i = 0
foreach ($t in $turns) {
  $i++
  $name = 't{0:d2}_spk{1}.wav' -f $i, $t.spk
  $path = Join-Path $OutDir $name
  $syn = New-Object System.Speech.Synthesis.SpeechSynthesizer
  try {
    $syn.SelectVoice($t.voice)
    $syn.SetOutputToWaveFile($path)
    $syn.Speak($t.text)
    $syn.SetOutputToNull()
  } finally {
    $syn.Dispose()
  }
  $manifest += [pscustomobject]@{ index = $i; speaker = $t.spk; voice = $t.voice; file = $name; text = $t.text }
  Write-Output ('  {0}  spk{1}  {2,-26} {3:n0} bytes' -f $name, $t.spk, $t.voice, (Get-Item $path).Length)
}

$manifest | ConvertTo-Json -Depth 3 | Set-Content -Path (Join-Path $OutDir 'manifest.json') -Encoding utf8
Write-Output ''
Write-Output ('OK - {0} locuciones, 2 voces distintas' -f $manifest.Count)
