param([Parameter(Mandatory=$true)][string]$TextBase64,
      [Parameter(Mandatory=$true)][string]$OutputPath)
$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($TextBase64))
Add-Type -AssemblyName System.Speech
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
  24000,
  [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
  [System.Speech.AudioFormat.AudioChannel]::Mono
)
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.SetOutputToWaveFile($OutputPath, $format)
$voice.Speak($text)
$voice.Dispose()
