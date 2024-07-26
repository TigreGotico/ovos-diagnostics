import importlib
import json
import os
import platform

import click
from mycroft.skills.common_play_skill import CommonPlaySkill
from ovos_backend_client.database import OAuthTokenDatabase
from ovos_config import Configuration
from ovos_config.locations import get_xdg_config_save_path
from ovos_plugin_manager.skills import find_skill_plugins
from ovos_utils.gui import is_installed
from ovos_utils.log import LOG
from ovos_workshop.skills.common_play import OVOSCommonPlaybackSkill

CONFIG = Configuration()

LOG.set_level("ERROR")


def is_raspberry_pi():
    if os.path.isfile('/sys/firmware/devicetree/base/model'):
        with open('/sys/firmware/devicetree/base/model', 'r') as f:
            model = f.read().strip()
            return 'Raspberry Pi' in model
    return False


def is_gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        pass
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except:
        pass
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        return any(provider.startswith('CUDA') for provider in available_providers)
    except:
        pass
    return False


###################################3
# Platform Info
LANG = CONFIG.get("lang")
HOMESCREEN_ID = CONFIG.get("gui", {}).get("idle_display_skill")
HAS_SHELL = is_installed('ovos-shell')
HAS_GUI = is_installed('ovos-shell') or is_installed('ovos-gui-app') or is_installed('mycroft-gui-app')
HAS_GPU = is_gpu_available()
IS_RPI = is_raspberry_pi()
IS_LINUX = platform.system() == "Linux"
IS_MK_1 = False  # TODO
IS_MK_2 = False  # TODO
IS_MK_2_DEVKIT = False  # TODO
IS_DOTSTAR = False  # TODO


###################################3
# COMMAND GROUPS


@click.group()
def cli():
    """OVOS Plugin Manager"""
    pass


@click.group()
def listener():
    """Manage listener plugins"""
    pass


@click.group()
def skills():
    """Manage skills plugins"""
    pass


@click.group()
def phal():
    """Manage PHAL platform plugins"""
    pass


@click.group()
def gui():
    """Manage GUI platform plugins"""
    pass


@click.group()
def language():
    """Manage language plugins"""
    pass


@click.group()
def audio():
    """Manage audio plugins"""
    pass


# Add groups to the main CLI group
cli.add_command(language)
cli.add_command(audio)
cli.add_command(listener)
cli.add_command(skills)
cli.add_command(phal)
cli.add_command(gui)


###################################3
# SCAN

@language.command()
def scan_translation():
    """List available translation plugins"""
    click.echo("Listing Translation Plugins...")
    from ovos_plugin_manager.language import find_tx_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@language.command()
def scan_detection():
    """List available language detection plugins"""
    click.echo("Listing Language Detection Plugins...")
    from ovos_plugin_manager.language import find_lang_detect_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@audio.command()
def scan_tts():
    """List available TTS plugins"""
    click.echo("List of TTS Plugins")
    from ovos_plugin_manager.tts import find_tts_plugins
    for idx, plugin in enumerate(find_tts_plugins()):
        click.echo(f" {idx} - {plugin}")


@audio.command()
def scan_wake_words():
    """List available Wake Word plugins"""
    click.echo("List of Wake Word Plugins")
    from ovos_plugin_manager.wakewords import find_wake_word_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@audio.command()
def scan_dialog_transformers():
    """List available Dialog Transformer plugins"""
    click.echo("Listing Dialog Transformer Plugins...")
    from ovos_plugin_manager.dialog_transformers import find_dialog_transformer_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@audio.command()
def scan_tts_transformers():
    """List available TTS Transformer plugins"""
    click.echo("Listing TTS Transformer Plugins...")
    from ovos_plugin_manager.dialog_transformers import find_tts_transformer_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@audio.command()
def scan_audio_players():
    """List available Audio Player plugins"""
    click.echo("Listing Audio Player Plugins...")
    from ovos_plugin_manager.audio import find_audio_service_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@audio.command()
def scan_ocp():
    """List available Audio Player plugins"""
    click.echo("Listing Stream Extractor Plugins...")
    from ovos_plugin_manager.ocp import find_ocp_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@listener.command()
def scan_microphone():
    """List available Microphone  plugins"""
    click.echo("Listing Microphone Plugins...")
    from ovos_plugin_manager.microphone import find_microphone_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@listener.command()
def scan_audio_transformers():
    """List available Audio Transformer plugins"""
    click.echo("Listing Audio Transformer Plugins...")
    from ovos_plugin_manager.audio_transformers import find_audio_lang_detector_plugins, \
        find_audio_transformer_plugins as finder
    lang = list(find_audio_lang_detector_plugins())
    for idx, plugin in enumerate(finder()):
        if plugin in lang:
            continue
        click.echo(f" {idx} - {plugin}")


@listener.command()
def scan_lang_detect():
    """List available Audio Transformer plugins"""
    click.echo("Listing Audio Language Detector Plugins...")
    from ovos_plugin_manager.audio_transformers import find_audio_lang_detector_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@listener.command()
def scan_stt():
    """List available STT plugins"""
    click.echo("List of STT Plugins")
    from ovos_plugin_manager.stt import find_stt_plugins
    for idx, plugin in enumerate(find_stt_plugins()):
        click.echo(f" {idx} - {plugin}")


@listener.command()
def scan_vad():
    """List available VAD plugins"""
    click.echo("List of VAD Plugins")
    from ovos_plugin_manager.vad import find_vad_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@skills.command()
def scan_pipeline():
    """List available pipeline plugins"""
    click.echo("Listing Pipeline Plugins...")
    click.echo("  WARNING: pipeline is not yet managed by OPM, valid config values are hardcoded")
    plugins = [
        "stop_high",
        "converse",
        "ocp_high",
        "padatious_high",
        "adapt_high",
        "ocp_medium",
        "fallback_high",
        "stop_medium",
        "adapt_medium",
        "padatious_medium",
        "adapt_low",
        "common_qa",
        "fallback_medium",
        "fallback_low"
    ]
    for idx, plugin in enumerate(plugins):
        click.echo(f" {idx} - {plugin}")


@skills.command()
def scan_reranker():
    """List available reranker plugins"""
    click.echo("Listing ReRanker Plugins...")
    from ovos_plugin_manager.solvers import find_multiple_choice_solver_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@skills.command()
def scan_utterance():
    """List available utterance plugins"""
    click.echo("Listing Utterance Plugins...")
    from ovos_plugin_manager.text_transformers import find_utterance_transformer_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@skills.command()
def scan_skills():
    """List available skills"""
    from ovos_plugin_manager.skills import find_skill_plugins as finder
    plugs = list(finder())
    click.echo("Listing installed skills...")
    for idx, plugin in enumerate(plugs):
        click.echo(f" {idx} - {plugin}")


@skills.command()
def scan_metadata():
    """List available metadata plugins"""
    click.echo("Listing Metadata Plugins...")
    from ovos_plugin_manager.metadata_transformers import find_metadata_transformer_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


@gui.command()
def scan_extensions():
    """List available VAD plugins"""
    click.echo("List of GUI Plugins")
    from ovos_plugin_manager.gui import find_gui_plugins as finder
    for idx, plugin in enumerate(finder()):
        click.echo(f" {idx} - {plugin}")


###################################3
# RECOMMEND


@gui.command()
def recommend_extensions():
    """recommend GUI config """
    from ovos_plugin_manager.gui import find_gui_plugins as finder
    from ovos_plugin_manager.skills import find_skill_plugins
    plugs = list(finder())
    skills = list(find_skill_plugins())

    # perform these checks
    homescreen_installed = HOMESCREEN_ID in skills

    if not plugs:
        click.echo("WARNING: No GUI plugins installed!!!")
    else:
        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")

    EXT = "ovos-gui-plugin-shell-companion"
    if IS_MK_2 and not HAS_SHELL:
        click.echo("ERROR: ovos-shell not installed")
    elif not HAS_GUI and not IS_MK_1:
        click.echo("WARNING: no GUI client installed")

    if IS_MK_2 and CONFIG.get("gui", {}).get("extension", "") != EXT:
        click.echo(f"ERROR: 'gui.extension' is NOT set to '{EXT}' in 'mycroft.conf'")
    if HAS_GUI and not homescreen_installed:
        click.echo(f"WARNING: GUI installed, but homescreen missing, please install '{HOMESCREEN_ID}'")

    if HAS_SHELL and EXT not in plugs:
        click.echo(
            f"RECOMMENDED GUI PLUGIN: ovos-shell is installed, but the companion plugin is missing, please install '{EXT}'")
    if not HAS_SHELL and not IS_MK_2 and "ovos-gui-plugin-shell-companion" in plugs:
        click.echo(
            f"UNINSTALL: '{EXT}' is installed, but ovos-shell is missing, either remove the plugin or install ovos-shell")


@audio.command()
def recommend_wake_words():
    """List available Wake Word plugins"""
    click.echo("List of Wake Word Plugins")
    from ovos_plugin_manager.wakewords import find_wake_word_plugins as finder
    plugs = list(finder())
    for idx, plugin in enumerate(plugs):
        click.echo(f" {idx} - {plugin}")

    main_ww = CONFIG.get("listener", {}).get("wake_word")
    main_standup = CONFIG.get("listener", {}).get("stand_up_word")
    hotwords = CONFIG.get("hotwords")

    def validate_ww(name: str, listen: bool = False,
                    wakeup: bool = False,
                    stopword: bool = False):
        if name in hotwords:
            listen = listen or hotwords[name].get("listen")
            wakeup = wakeup or hotwords[name].get("wakeup")
            stopword = stopword or hotwords[name].get("stopword")

        t = "wake word (listen mode)" if listen else None
        t = t or ("stand up word (sleep mode)" if wakeup else None)
        t = t or ("stop word (recording mode)" if stopword else None)
        t = t or "hotword"

        click.echo(f"  '{name}' - {t}")
        if name not in hotwords:
            click.echo("   - ERROR: not defined in 'hotwords' section of config")
            return
        m = hotwords[name]['module']
        click.echo(f"   - plugin: {m}")
        if not listen and not wakeup and not stopword:
            snd = hotwords[name].get("sound")
            if snd:
                click.echo(f"   - plays sound: {snd}")
            bus = hotwords[name].get("bus_event")
            if bus:
                click.echo(f"   - bus event: {bus}")
            utt = hotwords[name].get("utterance")
            if utt:
                click.echo(f"   - utterance: {utt}")

        if "model" in hotwords[name]:
            click.echo(f"   - model: {hotwords[name]['model']}")
        if m == "ovos-ww-plugin-pocketsphinx":
            s = (hotwords[name].get("phonemes", []) or
                 "[]\n     WARNING: config incomplete, 'phonemes' not defined")
            click.echo(f"   - phonemes: {s}")
        if m == "ovos-ww-plugin-vosk":
            s = (hotwords[name].get('samples', []) or
                 "[]\n     WARNING: config incomplete, 'samples' not defined")
            click.echo(f"   - samples: {s}")
        if m not in plugs:
            click.echo(f"   - ERROR: {m} does not appear to be installed!")
            f = hotwords[name].get("fallback_ww")
            if f:
                click.echo(f"   - WARNING: fallback to be used: {f}")
                validate_ww(f)
            else:
                click.echo(f"   - ERROR: {m} does not have a fallback word!")

    click.echo("Validating HotWords config:")
    validate_ww(main_ww, listen=True)
    validate_ww(main_standup, wakeup=True)

    # print extra hotwords (explicitly enabled)
    for ww, data in CONFIG.get("hotwords", {}).items():
        if ww in [main_ww, main_standup]:
            continue
        if data.get("active"):
            validate_ww(ww)


@audio.command()
def recommend_tts():
    """recommend TTS config """
    from ovos_plugin_manager.tts import find_tts_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No TTS plugins installed!!!")
    else:
        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")

        # determine best online
        online_plugs = [p for p, _ in TTS_ONLINE_PREFS if p in plugs]
        if IS_RPI:
            online_plugs = [p for p in online_plugs if p not in TTS_RPI_BLACKLIST]
        best_online = None
        online_recommendation = f"not sure what to recommend"
        if online_plugs:
            for p, info in TTS_ONLINE_PREFS:
                if p in online_plugs:
                    best_online = p
                    online_recommendation = info
                    break
            else:
                if len(online_plugs) == 1:
                    best_online = online_plugs[0]
                    online_recommendation = "only available remote plugin"
                else:
                    best_online = online_plugs[0]
                    online_recommendation = f"random selection"

        # determine best offline
        best_offline = None
        offline_recommendation = f"not sure what to recommend"
        offline_plugs = [p for p, _ in TTS_OFFLINE_PREFS if p in plugs]
        if IS_RPI:
            offline_plugs = [p for p in offline_plugs if p not in TTS_RPI_BLACKLIST]

        if offline_plugs:
            for p, info in TTS_OFFLINE_PREFS:
                if p in offline_plugs:
                    best_offline = p
                    offline_recommendation = info
                    break
            else:
                best_offline = offline_plugs[0]
                if len(offline_plugs) == 1:
                    offline_recommendation = "only available offline plugin"
                else:
                    offline_recommendation = f"random selection"

        click.echo(f"OFFLINE RECOMMENDATION: {best_offline} - {offline_recommendation}")
        click.echo(f"ONLINE RECOMMENDATION: {best_online} - {online_recommendation}")

        if IS_RPI:
            # prefer online
            if best_online:
                best = best_online
                click.echo(f"TTS RECOMMENDATION: {best} - recommended online plugin, for optimal latency")
                if best_offline:
                    click.echo(
                        f"FALLBACK TTS RECOMMENDATION: {best_offline} - best offline plugin, to handle internet outages")
                elif len(online_plugs) > 1:
                    best_fallback = [p for p in online_plugs if p != best_online][0]
                    click.echo(
                        f"FALLBACK TTS RECOMMENDATION: {best_fallback} - second best online plugin, no offline TTS available")
                else:
                    click.echo(f"FALLBACK TTS RECOMMENDATION: None - main TTS is already offline")
            elif best_offline:  # no online
                best = best_offline
                click.echo(f"TTS RECOMMENDATION: {best} - recommended offline plugin")
                click.echo(
                    f"FALLBACK TTS RECOMMENDATION: None - Raspberry Pi has limited resources, avoid loading extra plugins")
            elif len(plugs) == 1:
                click.echo(f"TTS RECOMMENDATION: {plugs[0]} - only installed plugin")
                click.echo(f"FALLBACK TTS RECOMMENDATION: None - no extra plugins detected")
            else:
                click.echo(
                    f"TTS RECOMMENDATION: None - no suitable plugins detected, consider installing 'ovos-tts-plugin-server'")
                click.echo(
                    f"FALLBACK TTS RECOMMENDATION: None - Raspberry Pi has limited resources, avoid loading extra plugins")
        else:
            # prefer offline
            if best_offline:
                best = best_offline
                click.echo(f"TTS RECOMMENDATION: {best} - recommended offline plugin, for maximum privacy")
                click.echo(f"FALLBACK TTS RECOMMENDATION: None - main TTS is already offline")
            elif best_online:  # no offline plugins available
                best = best_online
                click.echo(f"TTS RECOMMENDATION: {best} - recommended online plugin")
                if len(online_plugs) > 1:  # 2 online plugins
                    best_fallback = [p for p in online_plugs if p != best_online][0]
                    click.echo(
                        f"FALLBACK TTS RECOMMENDATION: {best_fallback} - second best online plugin, in case the first fails")
                else:
                    click.echo(f"FALLBACK TTS RECOMMENDATION: None - no suitable plugins detected")
            elif len(plugs) == 1:
                click.echo(f"TTS RECOMMENDATION: {plugs[0]} - only installed plugin")
                click.echo(f"FALLBACK TTS RECOMMENDATION: None - no extra plugins detected")
            else:
                click.echo(
                    f"TTS RECOMMENDATION: None - no suitable plugins detected, consider installing 'ovos-tts-plugin-piper'")
                click.echo(f"FALLBACK TTS RECOMMENDATION: None - no suitable plugins detected")


@audio.command()
def recommend_players():
    """recommend TTS config """
    from ovos_plugin_manager.audio import find_audio_service_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No Audio Player plugins installed!!!")
    else:

        # perform these checks
        # TODO - check if current device is also a spotify player
        is_spotiplayer = True
        spotify_oauth = False
        click.echo(f"  - OCP installed: {'ovos_common_play' in plugs}")
        click.echo(f"  - is Spotify Device: {is_spotiplayer}")
        if is_spotiplayer:
            click.echo(f"  - Spotify Oauth complete: {spotify_oauth}")

        click.echo("Available plugins:")
        for p in plugs:
            if p == "ovos_common_play":
                continue
            click.echo(f" - {p}")

        best = None
        if "ovos_mpv" in plugs:
            best = "ovos_mpv"
            recommendation = "best performance and support for a large number of formats"
        elif "ovos_vlc" in plugs:
            best = "ovos_vlc"
            recommendation = "has issues with some streams, support for a large number of formats"
        elif len(plugs) == 1 and plugs[0] == ["ovos_spotify"]:
            best = plugs[0]
            recommendation = f"'{best}' - only installed plugin"
        else:
            recommendation = f"not sure what to recommend"

        click.echo(f"RECOMMENDED DEFAULT: {best} - {recommendation}")
        if "ovos_spotify" in plugs:
            if is_spotiplayer:
                recommendation = "can play locally or in remote players, needs premium"
            else:
                recommendation = "can play in remote players, needs premium"
            if not spotify_oauth:
                recommendation += ", needs oauth"
            click.echo(f"RECOMMENDED EXTRA: ovos_spotify - {recommendation}")


@audio.command()
def recommend_ocp():
    """recommend TTS config """
    from ovos_plugin_manager.ocp import find_ocp_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No OCP extractor plugins installed!!!")
    else:
        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")

        for plug, info in OCP_ESSENTIAL.items():
            if plug not in plugs:
                click.echo(f"RECOMMENDED OCP PLUGIN: '{plug}' - {info}")
            else:
                click.echo(f"INFO: '{plug}' - {info}")


@audio.command()
def recommend_dialog_transformers():
    """recommend Dialog Transformers config """
    from ovos_plugin_manager.dialog_transformers import find_dialog_transformer_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No Dialog Transformers plugins installed!!!")
    else:
        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")

    for p, info in DIALOG_TRANSFORMER_INFO.items():
        if p in plugs:
            click.echo(f"INFO: '{p}' - {info}")

    click.echo("Nothing to recommend")


@audio.command()
def recommend_tts_transformers():
    """recommend TTS Transformers config """
    from ovos_plugin_manager.dialog_transformers import find_tts_transformer_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No TTS Transformers plugins installed!!!")
    else:

        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")

    for p, info in TTS_TRANSFORMER_INFO.items():
        if p in plugs:
            click.echo(f"INFO: '{p}' - {info}")

    click.echo("Nothing to recommend")


@skills.command()
def recommend_pipeline():
    """recommend Pipeline config """
    PIPELINE = CONFIG.get("intents", {}).get("pipeline", [])
    HAS_OCP_SKILLS = any([issubclass(plug, OVOSCommonPlaybackSkill)
                          for plug in find_skill_plugins().values()]) or True
    HAS_LEGACY_OCP_SKILLS = any([issubclass(plug, CommonPlaySkill)
                                 for plug in find_skill_plugins().values()])
    OCP_IN_PIPELINE = any([p.startswith("ocp_") for p in PIPELINE if p != "ocp_legacy"])
    PADATIOUS_IN_PIPELINE = any([p.startswith("padatious_") for p in PIPELINE])
    PADACIOSO_IN_PIPELINE = any([p.startswith("padacioso_") for p in PIPELINE])
    LEGACY_OCP_IN_PIPELINE = "ocp_legacy" in PIPELINE

    if PADATIOUS_IN_PIPELINE:
        if not importlib.util.find_spec("padatious"):
            click.echo(
                "WARNING: 'padatious' is not installed, intent matching will be much slower via 'padacioso' fallback")
    if PADACIOSO_IN_PIPELINE:
        click.echo("WARNING: 'padacioso' is enabled in pipeline config, expect latency in intent matching")

    if HAS_OCP_SKILLS and not OCP_IN_PIPELINE:
        click.echo(
            "WARNING: OCP Skills detected, but OCP not in 'pipeline' config, add 'ocp_high' to your mycroft.conf to enable media queries")

    if HAS_LEGACY_OCP_SKILLS and not LEGACY_OCP_IN_PIPELINE:
        click.echo(
            "ERROR: Deprecated Mycroft CommonPlay skills detected, you need to add 'ocp_legacy' to your 'pipeline' config in mycroft.conf")
    elif not HAS_LEGACY_OCP_SKILLS and LEGACY_OCP_IN_PIPELINE:
        click.echo(
            "WARNING: your pipeline contains 'ocp_legacy', but no Mycroft CommonPlay skills installed, expect latency in intent matching")


@skills.command()
def recommend_skills():
    """recommend skills"""
    from ovos_plugin_manager.skills import find_skill_plugins as finder
    installed_skills = list(finder())

    click.echo("Listing installed skills...")
    for idx, plugin in enumerate(installed_skills):
        click.echo(f" {idx} - {plugin}")

    click.echo("Skill checks ...")

    from ovos_plugin_manager.phal import find_phal_plugins
    from ovos_plugin_manager.solvers import find_question_solver_plugins
    from ovos_plugin_manager.audio import find_audio_service_plugins
    from ovos_plugin_manager.ocp import find_ocp_plugins
    ocp = list(find_ocp_plugins())
    players = list(find_audio_service_plugins())
    phals = list(find_phal_plugins())
    solvers = list(find_question_solver_plugins())
    blacklist = CONFIG.get("skills", {}).get("blacklisted_skills", [])

    # check blacklist
    for s in blacklist:
        if s in installed_skills:
            click.echo(f"UNINSTALL: '{s}' is blacklisted, it should be uninstalled")

    # check missing settings setup
    for skill, requires in SKILLS_NEED_SETTINGS.items():
        if skill in installed_skills:
            settings = f"{get_xdg_config_save_path()}/skills/{skill}/settings.json"
            if os.path.isfile(settings):
                with open(settings) as f:
                    cfg = json.load(f)
            else:
                cfg = {}
            for k in requires:
                if k not in cfg:
                    click.echo(f"ERROR: '{skill}' is installed but '{k}' not set in '{settings}'")
                    # TODO - create a helper cli util to change skill settings
                    helper_cmd = None
                    if helper_cmd:
                        click.echo(f"   INFO: '{skill}' can be configured with the command '{helper_cmd}'")

    # check missing mycroft.conf changes
    for skill, requires in SKILLS_NEED_CONFIG.items():
        if skill in installed_skills:
            for k in requires:
                cfg = CONFIG
                for _ in k.split("."):
                    cfg = cfg.get(_, {})
                if not cfg:
                    click.echo(f"ERROR: '{skill}' is installed but '{k}' not set in 'mycroft.conf'")
                    # TODO - ovos-config helper command syntax
                    helper_cmd = None
                    if helper_cmd:
                        click.echo(f"   INFO: '{skill}' can be configured with the command '{helper_cmd}'")

    # check missing OAuth
    with OAuthTokenDatabase() as db:
        for skill, tok in SKILLS_NEED_OAUTH.items():
            if skill in installed_skills and tok not in db:
                click.echo(f"ERROR: '{skill}' is installed but OAuth token is missing from '{db.path}'")
                helper_cmd = SKILLS_OAUTH_HELPERS.get(skill)
                if helper_cmd:
                    click.echo(f"   INFO: '{skill}' OAuth can be performed with the command '{helper_cmd}'")

    # OS specific warnings
    if not IS_LINUX:
        for skill in LINUX_ONLY_SKILLS:
            if skill in installed_skills:
                click.echo(f"UNINSTALL: '{skill}' is for Linux only!")

    if not HAS_GPU:
        for skill in GPU_ONLY_SKILLS:
            click.echo(f"UNINSTALL: '{skill}' requires a GPU")

    if not IS_RPI:
        for skill in RPI_ONLY_SKILLS:
            click.echo(f"UNINSTALL: '{skill}' is for RaspberryPi only!")

    # required OCP warnings
    for skill, plugs in SKILLS_NEED_OCP.items():
        for plug in plugs:
            if skill in installed_skills and plug not in ocp:
                click.echo(f"ERROR: OCP stream extractor plugin missing '{plug}', required by '{skill}'")

    # required audio warnings
    for skill, plug in SKILLS_NEED_AUDIO.items():
        if skill in installed_skills and plug not in players:
            click.echo(f"ERROR: Audio Player plugin missing '{plug}', required by '{skill}'")

    # required PHAL warnings
    for skill, plug in SKILLS_NEED_PHAL.items():
        if skill in installed_skills and plug in LINUX_ONLY_PHAL and not IS_LINUX:
            click.echo(f"UNINSTALL: PHAL plugin missing '{plug}', required by '{skill}', but it is Linux ONLY!")
        elif skill in installed_skills and plug not in phals:
            click.echo(f"ERROR: PHAL plugin missing '{plug}', required by '{skill}'")

    # recommended extra phal plugins
    for skill, plug in SKILLS_RECOMMEND_PHAL.items():
        if plug not in phals and skill in installed_skills:
            click.echo(f"RECOMMENDED PHAL PLUGIN: '{plug}' is recommended by '{skill}'")

    # recommended extra skills
    skill = "skill-ovos-fallback-unknown.openvoiceos"
    if skill not in installed_skills:
        click.echo(
            f"RECOMMENDED SKILL: '{skill}' is recommended to ensure users always receive an answer to their questions")

    for plug, skill in PLAYER2SKILL.items():
        if plug in solvers and skill not in installed_skills:
            click.echo(f"RECOMMENDED SKILL: '{skill}' is the companion skill to '{plug}' audio playback plugin")

    for plug, skill in SOLVER2SKILL.items():
        if plug in solvers and skill not in installed_skills:
            click.echo(f"RECOMMENDED SKILL: '{skill}' is the companion skill to '{plug}' solver plugin")

    for plug, skill in PHAL2SKILL.items():
        if plug in solvers and skill not in installed_skills:
            click.echo(f"RECOMMENDED SKILL: '{skill}' is the companion skill to '{plug}' PHAL plugin")


@skills.command()
def recommend_reranker():
    """recommend TTS config """
    from ovos_plugin_manager.solvers import find_multiple_choice_solver_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No ReRanker plugins installed!!!")
    else:

        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")

        for plug, info in RERANKER_PREFS:
            if plug in plugs:
                if IS_RPI and plug in RERANKER_RPI_BLACKLIST:
                    continue
                click.echo(f"RECOMMENDED DEFAULT: {plug} - {info}")
                break
        else:
            if len(plugs) == 1:
                best = plugs[0]
                recommendation = "only installed plugin"
            else:
                best = None
                recommendation = "not sure what to recommend"
            click.echo(f"RECOMMENDED DEFAULT: {best} - {recommendation}")


@listener.command()
def recommend_stt():
    """recommend STT config """
    from ovos_plugin_manager.stt import find_stt_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No STT plugins installed!!!")
    else:
        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")

        # determine best online
        online_plugs = [p for p, _ in STT_ONLINE_PREFS if p in plugs]
        if IS_RPI:
            online_plugs = [p for p in online_plugs if p not in STT_RPI_BLACKLIST]
        best_online = None
        online_recommendation = f"not sure what to recommend"
        if online_plugs:
            for p, info in STT_ONLINE_PREFS:
                if p in online_plugs:
                    best_online = p
                    online_recommendation = info
                    break
            else:
                if len(online_plugs) == 1:
                    best_online = online_plugs[0]
                    online_recommendation = "only available remote plugin"
                else:
                    best_online = online_plugs[0]
                    online_recommendation = f"random selection"

        # determine best offline
        best_offline = None
        offline_recommendation = f"not sure what to recommend"
        offline_plugs = [p for p, _ in STT_OFFLINE_PREFS if p in plugs]
        if IS_RPI:
            offline_plugs = [p for p in offline_plugs if p not in STT_RPI_BLACKLIST]

        if offline_plugs:
            for p, info in STT_OFFLINE_PREFS:
                if p in offline_plugs:
                    best_offline = p
                    offline_recommendation = info
                    break
            else:
                best_offline = offline_plugs[0]
                if len(offline_plugs) == 1:
                    offline_recommendation = "only available offline plugin"
                else:
                    offline_recommendation = f"random selection"

        click.echo(f"OFFLINE RECOMMENDATION: {best_offline} - {offline_recommendation}")
        click.echo(f"ONLINE RECOMMENDATION: {best_online} - {online_recommendation}")

        if IS_RPI:
            if best_online:
                best = best_online
                click.echo(f"STT RECOMMENDATION: {best} - recommended online plugin")
                if len(online_plugs) > 1:  # 2 online plugins
                    best_fallback = [p for p in online_plugs if p != best_online][0]
                    click.echo(
                        f"FALLBACK STT RECOMMENDATION: {best_fallback} - second best online plugin, Raspberry Pi is not suited for offline STT")
                else:
                    click.echo(f"FALLBACK STT RECOMMENDATION: None - Raspberry Pi is not suited for offline STT")
            elif best_offline:
                best = best_offline
                click.echo(f"STT RECOMMENDATION: {best} - recommended offline plugin")
                click.echo(f"FALLBACK STT RECOMMENDATION: None - Raspberry Pi is not suited for offline STT")
            elif len(plugs) == 1:
                click.echo(f"STT RECOMMENDATION: {plugs[0]} - only installed plugin")
                click.echo(f"FALLBACK STT RECOMMENDATION: None - no extra plugins detected")
            else:
                click.echo(
                    f"STT RECOMMENDATION: None - no suitable plugins detected, consider installing 'ovos-stt-plugin-server'")
                click.echo(f"FALLBACK STT RECOMMENDATION: None - Raspberry Pi is not suited for offline STT")
        elif HAS_GPU:
            if best_offline:
                best = best_offline
                click.echo(f"STT RECOMMENDATION: {best} - recommended offline plugin")
                click.echo("FALLBACK STT RECOMMENDATION: None - already offline, no need to reach out to the internet!")
            elif best_online:
                best = best_online
                click.echo(f"STT RECOMMENDATION: {best} - recommended online plugin")
                if len(online_plugs) > 1:  # 2 online plugins
                    best_fallback = [p for p in online_plugs if p != best_online][0]
                    click.echo(
                        f"FALLBACK STT RECOMMENDATION: {best_fallback} - second best online plugin, in case the first fails")
                else:
                    click.echo(f"FALLBACK STT RECOMMENDATION: None - no suitable plugins detected")
            elif len(plugs) == 1:
                click.echo(f"STT RECOMMENDATION: {plugs[0]} - only installed plugin")
                click.echo(f"FALLBACK STT RECOMMENDATION: None - no extra plugins detected")
            else:
                click.echo(
                    f"STT RECOMMENDATION: None - no suitable plugins detected, consider installing 'ovos-stt-plugin-fasterwhisper'")
                click.echo(f"FALLBACK STT RECOMMENDATION: None - no suitable plugins detected")
        else:
            if best_online:
                best = best_online
                click.echo(f"STT RECOMMENDATION: {best} - recommended online plugin")
                if best_offline:
                    click.echo(
                        f"FALLBACK STT RECOMMENDATION: {best_offline} - recommended offline plugin, handle internet outages")
                elif len(online_plugs) > 1:
                    best_fallback = [p for p in online_plugs if p != best_online][0]
                    click.echo(f"FALLBACK STT RECOMMENDATION: {best_fallback} - second best online plugin")
                else:
                    click.echo(f"FALLBACK STT RECOMMENDATION: None - no suitable plugins detected")
            elif best_offline:
                best = best_offline
                click.echo(f"STT RECOMMENDATION: {best} - recommended offline plugin")
                if len(offline_plugs) > 1:
                    best_fallback = [p for p in offline_plugs if p != best_offline][0]
                    click.echo(f"FALLBACK STT RECOMMENDATION: {best_fallback} - second best offline plugin")
                else:
                    click.echo(f"FALLBACK STT RECOMMENDATION: None - no extra plugins detected")
            elif len(plugs) == 1:
                click.echo(f"STT RECOMMENDATION: {plugs[0]} - only installed plugin")
                click.echo(f"FALLBACK STT RECOMMENDATION: None - no extra plugins detected")
            else:
                click.echo(
                    f"STT RECOMMENDATION: None - no suitable plugins detected, consider installing 'ovos-stt-plugin-server'")
                click.echo(f"FALLBACK STT RECOMMENDATION: None - no suitable plugins detected")


@listener.command()
def recommend_vad():
    """recommend VAD config """
    click.echo("VAD plugin recommendations:")
    from ovos_plugin_manager.vad import find_vad_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No VAD plugins installed!!!")
    else:
        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")

        for p, info in VAD_PREFS:
            if p in plugs:
                if IS_RPI and p in VAD_RPI_BLACKLIST:
                    continue
                click.echo(f"RECOMMENDATION:  {p}' - {info}")
                break
        else:
            if len(plugs) == 1:
                click.echo(f"'RECOMMENDATION: {plugs[0]}' - only installed plugin")
            else:
                click.echo("RECOMMENDATION: not sure what to recommend")


@skills.command()
def recommend_utterance_transformers():
    """recommend Utterance Transformers config """
    from ovos_plugin_manager.text_transformers import find_utterance_transformer_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No Utterance Transformers plugins installed!!!")
    else:

        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")

    for plug, info in UTTERANCE_INFO.items():
        if plug in plugs:
            click.echo(f"INFO: '{plug}' - {info}")

    click.echo("Nothing to recommend")


@skills.command()
def recommend_metadata_transformers():
    """recommend Metadata Transformers config """
    from ovos_plugin_manager.metadata_transformers import find_metadata_transformer_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No Metadata Transformers plugins installed!!!")
    else:
        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")

    for plug, info in METADATA_INFO.items():
        if plug in plugs:
            click.echo(f"INFO: '{plug}' - {info}")

    click.echo("Nothing to recommend")


@listener.command()
def recommend_microphone():
    """recommend Microphone config """
    click.echo("Microphone plugin recommendations:")
    from ovos_plugin_manager.microphone import find_microphone_plugins as finder
    plugs = list(finder())
    if not plugs:
        click.echo("WARNING: No microphone plugins installed!!!")
    else:
        click.echo("Available plugins:")
        for p in plugs:
            click.echo(f" - {p}")
        is_alsa_compatible = platform.system() == "Linux"
        if is_alsa_compatible and "ovos-microphone-plugin-alsa" in plugs:
            recommendation = "'ovos-microphone-plugin-alsa' - low audio latency"
        elif "ovos-microphone-plugin-sounddevice" in plugs:
            recommendation = "'ovos-microphone-plugin-sounddevice' - multi-platform"
        elif "ovos-microphone-plugin-pyaudio" in plugs:
            recommendation = "'ovos-microphone-plugin-pyaudio' - multi-platform"
        elif len(plugs) == 1:
            recommendation = f"'{plugs[0]}' - only installed plugin"
        else:
            recommendation = f"not sure what to recommend"
        click.echo(f"RECOMMENDATION: {recommendation}")


@phal.command()
def recommend_platform():
    """recommend platform specific plugins"""
    from ovos_plugin_manager.phal import find_phal_plugins as finder
    plugs = list(finder())

    click.echo("Listing installed plugins...")
    for idx, plugin in enumerate(plugs):
        click.echo(f" {idx} - {plugin}")

    for plug, rec in PHAL_ESSENTIAL.items():
        if plug not in plugs:
            click.echo(f"RECOMMENDED PHAL PLUGIN: '{plug}' - {rec}")

    if not HAS_GPU:
        for p in GPU_ONLY_PHAL:
            if p in plugs:
                click.echo(f"UNINSTALL: '{p}', it requires a GPU")

    if IS_RPI:
        for plug, rec in PHAL_RPI_ESSENTIAL.items():
            if plug not in plugs:
                click.echo(f"RECOMMENDED PHAL PLUGIN: '{plug}' - {rec}")
    else:
        for p in RPI_ONLY_PHAL:
            if p in plugs:
                click.echo(f"UNINSTALL: '{p}', it is for raspberry pi only")

    if IS_LINUX:
        for plug, rec in PHAL_LINUX_ESSENTIAL.items():
            if plug not in plugs:
                click.echo(f"RECOMMENDED PHAL PLUGIN: '{plug}' - {rec}")
    else:
        for p in LINUX_ONLY_PHAL:
            if p in plugs:
                click.echo(f"UNINSTALL: '{p}', it is for Linux only")

    if IS_MK_1:
        for plug, rec in PHAL_MK1_ESSENTIAL.items():
            if plug not in plugs:
                click.echo(f"RECOMMENDED PHAL PLUGIN: '{plug}' - {rec}")
    else:
        for p in MK1_ONLY_PHAL:
            if p in plugs:
                click.echo(f"UNINSTALL: '{p}', it is for Mark1 only")

    if IS_MK_2:
        for plug, rec in PHAL_MK2_ESSENTIAL.items():
            if plug not in plugs:
                click.echo(f"RECOMMENDED PHAL PLUGIN: '{plug}' - {rec}")
    else:
        for p in MK2_ONLY_PHAL:
            if p in plugs:
                click.echo(f"UNINSTALL: '{p}', it is for Mark2 only")

    if IS_MK_2_DEVKIT:
        for plug, rec in PHAL_MK2DEV_ESSENTIAL.items():
            if plug not in plugs:
                click.echo(f"RECOMMENDED PHAL PLUGIN: '{plug}' - {rec}")
    else:
        for p in MK2DEV_ONLY_PHAL:
            if p in plugs:
                click.echo(f"UNINSTALL: '{p}', it is for Mark2 DevKit only")


#######################################################
# Recommendations are defined here, manually maintained

TTS_RPI_BLACKLIST = []
TTS_ONLINE_PREFS = [
    ("ovos-tts-plugin-edge-tts", "multilingual, fast, high quality, no api key required"), # TODO - filter by supported langs
    ("ovos-tts-plugin-server", "self hosted, community maintained public servers (piper)"), # TODO - filter by supported langs
    ("ovos-tts-plugin-google-tx", "extensive language support, free, single voice"), # TODO - filter by supported langs
    ("ovos-tts-plugin-polly", "fast and accurate, but requires api key"), # TODO - filter by supported langs
    ("ovos-tts-plugin-azure", "fast and accurate, but requires api key"), # TODO - filter by supported langs
    ("ovos-tts-plugin-mimic3",  "self hosted, models available for several languages, abandoned project, precursor to piper"), # TODO - filter by supported langs
    ("neon-tts-plugin-larynx-server",  "self hosted, models available for several languages, abandoned project, precursor to mimic3"), # TODO - filter by supported langs
    ("ovos-tts-plugin-marytts", "self hosted, models available for several languages, many projects offer compatible apis") # TODO - filter by supported langs
]
TTS_OFFLINE_PREFS = [
    ("ovos-tts-plugin-piper", "lightweight, models available for several languages"), # TODO - filter by supported langs
    ("ovos-tts-plugin-mimic3", "lightweight, models available for several languages, abandoned project, precursor to piper"), # TODO - filter by supported langs
    ("ovos-tts-plugin-mimic", "lightweight, sounds robotic, english only") 
    if LANG.startswith("en") else ("", ""),
    ("ovos-tts-plugin-pico", "very lightweight, limited language support") 
    if any([LANG.startswith(l) for l in ["de", "es", "fr", "it", "en"]]) else ("", ""),
    ("ovos-tts-plugin-espeakng", "very lightweight, sounds robotic, extensive language support"), # TODO - filter by supported langs
    ("ovos-tts-plugin-SAM", "TTS from the 80s, VERY robotic, english only") 
    if LANG.startswith("en") else ("", ""),
    ("ovos-tts-plugin-beepspeak", "robot beeps, for fun and testing only") 
]


STT_RPI_BLACKLIST = ["ovos-stt-plugin-fasterwhisper", "neon-stt-plugin-nemo"]
STT_ONLINE_PREFS = [
    ("ovos-stt-plugin-chromium", "multilingual, free, unmatched performance, but does not respect your privacy"),# TODO - filter by supported langs
    ("ovos-stt-plugin-server", "multilingual, variable performance, self hosted, community maintained public  (fasterwhisper)"),# TODO - filter by supported langs
    ("ovos-stt-plugin-azure", "multilingual, fast and accurate, but requires api key") # TODO - filter by supported langs 
]
if not HAS_GPU:
    STT_OFFLINE_PREFS = [
        ("neon-stt-plugin-nemo", "monolingual models, medium accuracy"), # TODO - filter by supported langs
        ("ovos-stt-plugin-fasterwhisper", "multilingual, slow without a GPU"), # TODO - filter by supported langs
        ("ovos-stt-plugin-vosk", "monolingual models, not very accurate, lightweight, suitable for raspberry pi"), # TODO - filter by supported langs
        ("ovos-stt-plugin-pocketsphinx", "worst accuracy, only use as last resort"), # TODO - filter by supported langs
    ]
else:
    STT_OFFLINE_PREFS = [
        ("ovos-stt-plugin-fasterwhisper", "multilingual, GPU allows fast inference"), # TODO - filter by supported langs
        ("neon-stt-plugin-nemo", "monolingual models, medium accuracy"), # TODO - filter by supported langs
        ("ovos-stt-plugin-vosk", "monolingual models, not very accurate, lightweight, suitable for raspberry pi"), # TODO - filter by supported langs
        ("ovos-stt-plugin-pocketsphinx", "worst accuracy, only use as last resort") # TODO - filter by supported langs
    ]

RERANKER_RPI_BLACKLIST = ["ovos-flashrank-reranker-plugin"]
RERANKER_PREFS = [
    ("ovos-flashrank-reranker-plugin", "best, lightweight and fast"), # TODO - filter by supported langs
    ("ovos-bm25-reranker-plugin", "lightweight and fast, based on text similarity"), # TODO - filter by supported langs
    ("ovos-choice-solver-bm25", "no extra dependencies, comes from 'ovos-classifiers'") # TODO - filter by supported langs
]
VAD_RPI_BLACKLIST = []
VAD_PREFS = [
    ("ovos-vad-plugin-silero", "best accuracy, lightweight"),
    ("ovos-vad-plugin-noise", "worst accuracy, lightweight, silence based, no external dependencies, configurable"),
    ("ovos-vad-plugin-precise", "moderate accuracy, lightweight, needs tweaking to work well"),
    ("ovos-vad-plugin-webrtcvad", "lightweight, silence based, has issues in some platforms")
]

OCP_ESSENTIAL = {
    "ovos-ocp-rss-plugin": "allows extracting streams from rss feeds, crucial for news",
    "ovos-ocp-youtube-plugin": "yt-dlp allows extracting streams from several webpages, not only youtube!",
    "ovos-ocp-m3u-plugin": "needed for .pls and .m3u streams, common in internet radio"
}
METADATA_INFO = {}
UTTERANCE_INFO = {
    "ovos-utterance-plugin-cancel": "can cancel utterances on false activations or when you change your mind mid sentence",
    "ovos-utterance-corrections-plugin": " allows you to manually correct common STT mistakes",
    "ovos-utterance-translation-plugin": "works together with 'ovos-dialog-translation-plugin' providing bidirectional translation for utterances in unsupported languages, very useful for chat inputs"
}
DIALOG_TRANSFORMER_INFO = {
    "ovos-dialog-transformer-openai-plugin": "can rewrite dialogs on demand via ChatGPT (or OpenAI compatible API)",
    "ovos-dialog-translation-plugin": "works together with 'ovos-utterance-translation-plugin', automatically translates utterances back to the user language"
}
TTS_TRANSFORMER_INFO = {"ovos-tts-transformer-sox-plugin": "can apply effects to TTS, such as pitch or rate changes"}

PHAL_ESSENTIAL = {
    "ovos-phal-plugin-connectivity-events": "improves reaction to network state changes",
    "ovos-phal-plugin-ipgeo": "ensures approximate location and timezone until users configure it",
    "ovos-PHAL-plugin-oauth": "allows other components to perform oauth",
}
PHAL_LINUX_ESSENTIAL = {
    "ovos-PHAL-plugin-system": "allows OVOS to shutdown/restart the system",
    "ovos-PHAL-plugin-alsa": "allows OVOS to control global volume",
    "ovos-PHAL-plugin-network-manager": "allows OVOS to setup wifi"
}
PHAL_RPI_ESSENTIAL = {}
PHAL_MK1_ESSENTIAL = {
    "ovos-phal-mk1": "needed to control Mark1 eyes and faceplate",
    "ovos-PHAL-plugin-balena-wifi": "allows OVOS to setup wifi via hotspot",
}
PHAL_MK2_ESSENTIAL = {
    'ovos-PHAL-plugin-hotkeys': "needed to react to Mark2 key presses",
    "ovos-PHAL-plugin-gui-network-client": "needed to setup Wifi via the touchscreen",
    "ovos-PHAL-plugin-wallpaper-manager": "needed to manage wallpapers",
}
PHAL_MK2DEV_ESSENTIAL = {
    'ovos-PHAL-plugin-mk2-v6-fan-control': "needed to control Mark2 DevKit fan",

}

GPU_ONLY_PHAL = []
LINUX_ONLY_PHAL = [
    'ovos-PHAL-plugin-alsa',
    "ovos-PHAL-plugin-wallpaper-manager",  # TODO update platform support upstream
    "ovos-PHAL-plugin-system",
    "ovos-PHAL-plugin-network-manager"
]
RPI_ONLY_PHAL = [
    "ovos-PHAL-plugin-dotstar"
]
MK1_ONLY_PHAL = ["ovos-phal-mk1"]
MK2_ONLY_PHAL = ["ovos-PHAL-plugin-gui-network-client"]
MK2DEV_ONLY_PHAL = ["ovos-PHAL-plugin-mk2-v6-fan-control"]

LINUX_ONLY_SKILLS = ["ovos-skill-application-launcher.openvoiceos"]
GPU_ONLY_SKILLS = []
RPI_ONLY_SKILLS = []
MK1_ONLY_SKILLS = []
MK2_ONLY_SKILLS = []

SKILLS_NEED_OCP = {
    "skill-ovos-youtube-music.openvoiceos": ["ovos-ocp-youtube-plugin"],
    "skill-ovos-youtube.openvoiceos": ["ovos-ocp-youtube-plugin"],
    "skill-ovos-bandcamp.openvoiceos": ["ovos-ocp-bandcamp-plugin"],
    "skill-ovos-news.openvoiceos": ["ovos-ocp-rss-plugin", "ovos-ocp-news-plugin"],
}
SKILLS_NEED_PHAL = {
    "skill-ovos-volume.openvoiceos": 'ovos-PHAL-plugin-alsa',
    "neon_homeassistant_skill.mikejgray": 'ovos-PHAL-plugin-homeassistant'
}
SKILLS_NEED_AUDIO = {
    "ovos-skill-spotify.openvoiceos": 'ovos_spotify'
}
SKILLS_NEED_SETTINGS = {
    'skill-ovos-wolfie.openvoiceos': ["api_key"],
    'skill-ovos-fallback-chatgpt.openvoiceos': ["key"]
}
SKILLS_NEED_CONFIG = {
    "neon_homeassistant_skill.mikejgray": [
        "PHAL.ovos-PHAL-plugin-homeassistant.api_key"
    ]
}
SKILLS_NEED_OAUTH = {
    'ovos-skill-spotify.openvoiceos': "ocp_spotify"
}
SKILLS_OAUTH_HELPERS = {
    "ovos-skill-spotify.openvoiceos": "ovos-spotify-oauth"
}

# recommended by skilla
SKILLS_RECOMMEND_PHAL = {
    "ovos-skill-spotify.openvoiceos": 'ovos-PHAL-plugin-oauth',
    "neon_homeassistant_skill.mikejgray": 'ovos-PHAL-plugin-oauth'
}

# recommends skills
SOLVER2SKILL = {
    'ovos-solver-openai-persona-plugin': "skill-ovos-fallback-chatgpt.openvoiceos",
    'ovos-question-solver-wordnet': "skill-ovos-wordnet.openvoiceos"
}
PHAL2SKILL = {
    'ovos-PHAL-plugin-alsa': "skill-ovos-volume.openvoiceos",
    'ovos-PHAL-plugin-homeassistant': "neon_homeassistant_skill.mikejgray"
}
PLAYER2SKILL = {
    'ovos_spotify': "ovos-skill-spotify.openvoiceos"
}

if __name__ == "__main__":
    cli()
