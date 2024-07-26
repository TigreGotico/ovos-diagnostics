
## Usage

run `ovos-diagnostics`
```bash
Usage: ovos-diagnostics [OPTIONS] COMMAND [ARGS]...

  OVOS Diagnostics Tool
  
Options:
  --help  Show this message and exit.

Commands:
  audio     Manage audio plugins
  core      Manage skills plugins
  gui       Manage GUI platform plugins
  language  Manage language plugins
  listener  Manage listener plugins
  phal      Manage PHAL platform plugins
```

run `ovos-diagnostics core`
```bash
Usage: ovos-diagnostics core [OPTIONS] COMMAND [ARGS]...

  Manage skills plugins

Options:
  --help  Show this message and exit.

Commands:
  recommend-metadata-transformers
                                  Metadata Transformers recommendations
  recommend-pipeline              Pipeline recommendations
  recommend-reranker              TTS recommendations
  recommend-skills                Skill recommendations
  recommend-utterance-transformers
                                  Utterance Transformers recommendations
  scan-metadata                   List available metadata plugins
  scan-pipeline                   List available pipeline plugins
  scan-reranker                   List available reranker plugins
  scan-skills                     List available skills
  scan-utterance                  List available utterance plugins
```

run `ovos-diagnostics audio`
```bash
Usage: ovos-diagnostics audio [OPTIONS] COMMAND [ARGS]...

  Manage audio plugins

Options:
  --help  Show this message and exit.

Commands:
  recommend-dialog-transformers  Dialog Transformers recommendations
  recommend-ocp                  OCP stream extractor recommendations
  recommend-players              Audio Player recommendations
  recommend-tts                  TTS recommendations
  recommend-tts-transformers     TTS Transformers recommendations
  recommend-wake-words           List available Wake Word plugins
  scan-audio-players             List available Audio Player plugins
  scan-dialog-transformers       List available Dialog Transformer plugins
  scan-ocp                       List available Audio Player plugins
  scan-tts                       List available TTS plugins
  scan-tts-transformers          List available TTS Transformer plugins
  scan-wake-words                List available Wake Word plugins
```
run `ovos-diagnostics listener`
```bash
Usage: ovos-diagnostics listener [OPTIONS] COMMAND [ARGS]...

  Manage listener plugins

Options:
  --help  Show this message and exit.

Commands:
  recommend-microphone     Microphone recommendations
  recommend-stt            STT recommendations
  recommend-vad            VAD recommendations
  scan-audio-transformers  List available Audio Transformer plugins
  scan-lang-detect         List available Audio Language Detector plugins
  scan-microphone          List available Microphone plugins
  scan-stt                 List available STT plugins
  scan-vad                 List available VAD plugins
```


run `ovos-diagnostics language`
```bash
Usage: ovos-diagnostics language [OPTIONS] COMMAND [ARGS]...

  Manage language plugins

Options:
  --help  Show this message and exit.

Commands:
  recommend-detector    language detector recommendations
  recommend-translator  translation recommendations
  scan-detection        List available language detection plugins
  scan-translation      List available translation plugins

```
run `ovos-diagnostics phal`
```bash
Usage: ovos-diagnostics.py phal [OPTIONS] COMMAND [ARGS]...

  Manage PHAL platform plugins

Options:
  --help  Show this message and exit.

Commands:
  recommend-platform  platform specific recommendations
```


run `ovos-diagnostics gui`
```bash
Usage: ovos-diagnostics gui [OPTIONS] COMMAND [ARGS]...

  Manage GUI platform plugins

Options:
  --help  Show this message and exit.

Commands:
  recommend-extensions  GUI recommendations
  scan-extensions       List available VAD plugins
```
