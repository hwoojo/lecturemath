"""Call original script functions without ConsoleUIProcess's dataset XML wrapper."""
import importlib.util
import sys
from functools import lru_cache
from .config import RELEASE


def activate():
    if str(RELEASE) not in sys.path:
        sys.path.insert(0, str(RELEASE))


@lru_cache
def script(stage):
    activate()
    names = {2: "cc_analaysis", 3: "cc_grouping", 4: "vid_segmentation"}
    path = RELEASE / f"pre_ST3D_v3.0_0{stage}_{names[stage]}.py"
    spec = importlib.util.spec_from_file_location(f"boardnotes_stage_{stage}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def research_config(settings):
    activate()
    from AM_CommonTools.configuration.configuration import Configuration
    config = Configuration.from_file(str(settings.config_path))
    config.set("VIDEO_SEGMENTATION_METHOD", "3")
    config.set("SAMPLING_FPS", str(settings.sampling_fps))
    # Original thresholds use sampled frame counts; keep their duration at other FPS.
    for key in ("CC_STABILITY_MAX_GAP", "CC_STABILITY_MIN_TIMES", "CC_GROUPING_TEMPORAL_WINDOW", "VIDEO_SEGMENTATION_DEL_EVENT_MIN_LENGTH"):
        config.set(key, str(max(1, round(config.get_int(key) * settings.sampling_fps))))
    return config
