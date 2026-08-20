# Generates deterministic Spanish speech fixtures with the Windows SAPI engine.
# Fully offline and reproducible: no network, no licensing, no recorded humans.
# Prosody is varied per "speaker" so the same voice yields acoustically distinct
# talkers - enough to exercise diarization clustering.

param([string]$OutDir = "$PSScriptRoot\fixtures_3spk")

Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# A plausible project meeting, in the register these transcripts actually contain:
# decisions, owners, dates, numbers - the things the summarizer must extract.
$turns = @(
  @{ spk = 1; rate = 0;  pitch = "default"; text = "Buenos dias a todos, gracias por conectarse. Vamos a revisar el avance del proyecto de facturacion electronica." }
  @{ spk = 2; rate = 2;  pitch = "high";    text = "Perfecto. Por mi parte el modulo de validacion ya esta terminado, lo subi al repositorio ayer por la tarde." }
  @{ spk = 1; rate = 0;  pitch = "default"; text = "Excelente. Y como vamos con la integracion al sistema del banco?" }
  @{ spk = 3; rate = -2; pitch = "low";     text = "Ahi tenemos un bloqueo. El banco todavia no nos entrega las credenciales del ambiente de pruebas." }
  @{ spk = 1; rate = 0;  pitch = "default"; text = "Entendido. Entonces necesitamos escalarlo. Yo hablo con el contacto comercial esta misma semana." }
  @{ spk = 2; rate = 2;  pitch = "high";    text = "De acuerdo. Mientras tanto puedo avanzar con las pruebas unitarias usando un simulador." }
  @{ spk = 3; rate = -2; pitch = "low";     text = "Me parece bien. Calculo que el simulador nos toma dos dias de trabajo, maximo tres." }
  @{ spk = 1; rate = 0;  pitch = "default"; text = "Listo. Entonces quedamos asi: yo escalo el tema de las credenciales antes del viernes." }
  @{ spk = 2; rate = 2;  pitch = "high";    text = "Y yo entrego el simulador y las pruebas unitarias para el proximo martes." }
  @{ spk = 1; rate = 0;  pitch = "default"; text = "Muy bien. El presupuesto aprobado es de cuarenta y cinco mil pesos para esta fase. Nos vemos la proxima semana." }
)

function New-Utterance {
  param($Turn, $Path)
  $syn = New-Object System.Speech.Synthesis.SpeechSynthesizer
  try {
    $voice = $syn.GetInstalledVoices() |
      Where-Object { $_.VoiceInfo.Culture.Name -like "es*" } |
      Select-Object -First 1
    if ($voice) { $syn.SelectVoice($voice.VoiceInfo.Name) }
    $syn.Rate = $Turn.rate
    $syn.SetOutputToWaveFile($Path)
    $escaped = $Turn.text -replace '&', '&amp;' -replace '<', '&lt;' -replace '>', '&gt;'
    $ssml = @"
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='es-ES'>
<prosody pitch='$($Turn.pitch)'>$escaped</prosody>
</speak>
"@
    $syn.SpeakSsml($ssml)
    $syn.SetOutputToNull()
  } finally { $syn.Dispose() }
}

$manifest = @()
$i = 0
foreach ($t in $turns) {
  $i++
  $name = "turn_{0:d2}_spk{1}.wav" -f $i, $t.spk
  $path = Join-Path $OutDir $name
  New-Utterance -Turn $t -Path $path
  $manifest += [pscustomobject]@{ index = $i; speaker = $t.spk; file = $name; text = $t.text }
  Write-Output ("  {0}  spk{1}  {2:n0} bytes" -f $name, $t.spk, (Get-Item $path).Length)
}

$manifest | ConvertTo-Json -Depth 3 | Set-Content -Path (Join-Path $OutDir "manifest.json") -Encoding utf8
Write-Output ""
Write-Output ("Generadas {0} locuciones en {1}" -f $manifest.Count, $OutDir)
