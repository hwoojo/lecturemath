import pickle
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
from .upstream import script, research_config, activate


def preserve_conflicting_erasures(intervals, conflict_data):
    """Keep stable, temporally disjoint writing that upstream's last-wins
    conflict resolution would discard in segments too short for a delete peak.
    Use upstream's CC conflict graph; do not infer any missing handwriting.
    """
    ages, conflicts = conflict_data
    refined = []
    for start, end in intervals:
        cuts = set()
        for first, neighbors in conflicts.items():
            for second, info in neighbors.items():
                if first >= second or info["matched"] <= 0:
                    continue
                older, newer = sorted((first, second), key=lambda key: ages[key][0])
                old_last, new_first = ages[older][-1], ages[newer][0]
                if start <= old_last < new_first <= end:
                    cuts.add((old_last + new_first + 1) // 2)
        boundaries = [start, *sorted(cuts), end + 1]
        refined.extend((left, right-1) for left, right in zip(boundaries, boundaries[1:]))
    return refined


def summarize(binary_data, settings, output_dir, progress=lambda *args: None):
    """Reuse original stage 02, 03, 04 and stage 05's keyframe generator.

    Times are milliseconds; masks are PNG encoded, white foreground on black.
    Pickles are local debug artifacts only and are never read from uploads.
    """
    activate()
    from AccessMath.preprocessing.content.keyframe_extractor import KeyframeExtractor
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = SimpleNamespace(configuration=research_config(settings), params={},
                              img_dir=str(output_dir), current_lecture=SimpleNamespace(title="upload"))

    def save(name, value):
        if settings.keep_intermediates:
            with (output_dir / f"{name}.pickle").open("wb") as stream:
                pickle.dump(value, stream)

    save("01_binary", binary_data)
    progress("cc_analysis", 60)
    stability = script(2).process_input(context, binary_data)
    save("02_stability", stability)
    progress("cc_grouping", 72)
    grouped = script(3).process_input(context, stability)
    save("03_grouped", grouped)
    progress("deletion_segmentation", 83)
    deletion_intervals = script(4).process_input(context, grouped)
    intervals = preserve_conflicting_erasures(deletion_intervals, grouped[1])
    save("04_intervals", intervals)
    progress("keyframes", 90)
    images, cc_times = KeyframeExtractor.GenerateFromST3DForIntervals(grouped[2], intervals)
    save("05_keyframes", (images, cc_times))
    times = binary_data[0]
    return [(image, times[start] / 1000, times[end] / 1000) for image, (start, end) in zip(images, intervals)], {
        "sample_count": len(times), "raw_components": stability[2].get_raw_cc_count(),
        "unique_components": len(stability[2].unique_cc_objects),
        "groups": len(grouped[2].cc_group_ages), "segments": intervals,
        "deletion_intervals": deletion_intervals,
        "conflict_preservation_splits": len(intervals) - len(deletion_intervals),
        "segmentation": "upstream deletion events (method 3), plus disjoint CC conflict preservation"}
