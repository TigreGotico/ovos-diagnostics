# OVOS Diagnostics Tool

The OVOS Diagnostics Tool is a command-line interface (CLI) utility designed to manage and diagnose various plugins used in the Open Voice OS (OVOS) ecosystem. It provides functionalities to list, recommend, and manage plugins across different categories such as audio, skills, GUI, language, listener, and platform-specific plugins.

## Usage

Once installed, you can use the `ovos-diagnostics` command to interact with the tool. The general syntax is:

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
  recommend-reranker              CommonQuery ReRanker recommendations
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

## Examples

### Listing Installed TTS Plugins

To list all available TTS plugins, you can use the following command:

```bash
ovos-diagnostics audio scan-tts
```
example output
```
List of TTS Plugins
 0 - ovos-tts-plugin-mimic3
 1 - ovos-tts-plugin-beepspeak
 2 - ovos-tts-plugin-dummy
 3 - ovos-tts-plugin-SAM
 4 - ovos-tts-plugin-polly
 5 - ovos_tts_plugin_espeakng
 6 - ovos-tts-plugin-piper
 7 - ovos-tts-plugin-pico
 8 - ovos-tts-plugin-edge-tts
 9 - ovos-tts-plugin-azure
 10 - ovos-tts-plugin-server
 11 - ovos-tts-plugin-marytts
 12 - neon-tts-plugin-larynx-server
 13 - ovos-tts-plugin-mimic
 14 - ovos-tts-plugin-google-tx
```

> NOTE: if a plugin is installed but doesnt show up here, then ovos-plugin-manager is failing to load it


### Recommending an STT Configuration

To get recommendations for the best STT configuration based on your platform, use:

```bash
ovos-diagnostics listener recommend-stt
```

example output
```
Available plugins:
 - ovos-stt-plugin-dummy
 - ovos-stt-plugin-azure
 - ovos-stt-plugin-vosk
 - ovos-stt-plugin-vosk-streaming
 - ovos-stt-plugin-chromium
 - ovos-stt-plugin-fasterwhisper
 - ovos-stt-plugin-server
OFFLINE RECOMMENDATION: ovos-stt-plugin-fasterwhisper - multilingual, GPU allows fast inference
ONLINE RECOMMENDATION: ovos-stt-plugin-server - multilingual, variable performance, self hosted, community maintained public  (fasterwhisper)
STT RECOMMENDATION: ovos-stt-plugin-fasterwhisper - recommended offline plugin
FALLBACK STT RECOMMENDATION: None - already offline, no need to reach out to the internet!
```

### Recommending Skills

To recommend extra skills, companion plugins or diagnosing missing settings

```bash
ovos-diagnostics core recommend-skills
```

example output
```
Listing installed skills...
 0 - skill-ovos-parrot.openvoiceos
 1 - skill-ovos-homescreen.openvoiceos
 2 - neon_homeassistant_skill.mikejgray
 3 - skill-ovos-hello-world.openvoiceos
 4 - skill-ovos-volume.openvoiceos
 5 - skill-ovos-fallback-chatgpt.openvoiceos
 6 - ovos-skill-application-launcher.openvoiceos
 7 - skill-ovos-youtube-music.openvoiceos
 8 - ovos-skill-spotify.openvoiceos
 9 - skill-ovos-youtube.openvoiceos
 10 - skill-ovos-fallback-unknown.openvoiceos
 11 - skill-ovos-date-time.openvoiceos
 12 - ovos-skill-personal.OpenVoiceOS
 13 - skill-ovos-wolfie.openvoiceos
 14 - ovos-skill-alerts.openvoiceos
 15 - skill-ovos-naptime.openvoiceos
 16 - skill-ovos-news.openvoiceos
 17 - skill-ovos-bandcamp.openvoiceos
Skill checks ...
ERROR: 'skill-ovos-wolfie.openvoiceos' is installed but 'api_key' not set in '/home/miro/.config/mycroft/skills/skill-ovos-wolfie.openvoiceos/settings.json'
ERROR: 'skill-ovos-fallback-chatgpt.openvoiceos' is installed but 'key' not set in '/home/miro/.config/mycroft/skills/skill-ovos-fallback-chatgpt.openvoiceos/settings.json'
ERROR: 'neon_homeassistant_skill.mikejgray' is installed but 'PHAL.ovos-PHAL-plugin-homeassistant.api_key' not set in 'mycroft.conf'
ERROR: 'ovos-skill-spotify.openvoiceos' is installed but OAuth token is missing from '/home/miro/.cache/mycroft/json_database/ovos_oauth.json'
   INFO: 'ovos-skill-spotify.openvoiceos' OAuth can be performed with the command 'ovos-spotify-oauth'
ERROR: OCP stream extractor plugin missing 'ovos-ocp-youtube-plugin', required by 'skill-ovos-youtube-music.openvoiceos'
ERROR: OCP stream extractor plugin missing 'ovos-ocp-youtube-plugin', required by 'skill-ovos-youtube.openvoiceos'
ERROR: OCP stream extractor plugin missing 'ovos-ocp-bandcamp-plugin', required by 'skill-ovos-bandcamp.openvoiceos'
ERROR: OCP stream extractor plugin missing 'ovos-ocp-rss-plugin', required by 'skill-ovos-news.openvoiceos'
ERROR: OCP stream extractor plugin missing 'ovos-ocp-news-plugin', required by 'skill-ovos-news.openvoiceos'
ERROR: PHAL plugin missing 'ovos-PHAL-plugin-alsa', required by 'skill-ovos-volume.openvoiceos'
ERROR: PHAL plugin missing 'ovos-PHAL-plugin-homeassistant', required by 'neon_homeassistant_skill.mikejgray'
RECOMMENDED PHAL PLUGIN: 'ovos-PHAL-plugin-oauth' is recommended by 'ovos-skill-spotify.openvoiceos'
RECOMMENDED PHAL PLUGIN: 'ovos-PHAL-plugin-oauth' is recommended by 'neon_homeassistant_skill.mikejgray'
RECOMMENDED SKILL: 'skill-ovos-wordnet.openvoiceos' is the companion skill to 'ovos-question-solver-wordnet' solver plugin
```

## License

This project is licensed under the terms of the MIT license.
