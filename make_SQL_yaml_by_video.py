"""Detect character states and emit SQL, frame logs and an annotated video.

The layout detector registers source-derived face crops to the moving camera.
Light temporal regularisation rejects weak noise while retaining the deliberate
two-frame flashes in the source. The original global matcher remains available
with --detector templates for reproducible comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


STATES = ("absent", "real", "portrait")


@dataclass(frozen=True)
class Match:
    score: float
    point: tuple[int, int]
    size: tuple[int, int]


def read_image(path: Path) -> np.ndarray:
    """cv2.imread does not reliably handle Japanese paths on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read template: {path}")
    return image


def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 120)
    return gray, edges


def load_templates(root: Path, scales: tuple[float, ...]):
    templates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    for image_path in sorted(root.glob("*/*/*")):
        if not image_path.is_file():
            continue
        name, status = image_path.parent.parent.name, image_path.parent.name
        source = read_image(image_path)
        variants = templates.setdefault(name, {}).setdefault(status, [])
        for scale in scales:
            w = max(8, round(source.shape[1] * scale))
            h = max(8, round(source.shape[0] * scale))
            resized = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            variants.append(preprocess(resized))
    if not templates:
        raise ValueError(f"No templates below {root}")
    return templates


def best_match(frame_features, variants) -> Match:
    gray, edges = frame_features
    best = Match(-1.0, (0, 0), (0, 0))
    for template_gray, template_edges in variants:
        h, w = template_gray.shape
        if h > gray.shape[0] or w > gray.shape[1]:
            continue
        # Intensity preserves the subtle differences between similar faces;
        # edges make the result less sensitive to fades and colour grading.
        intensity = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, intensity_score, _, point = cv2.minMaxLoc(intensity)
        # Edge matching is only a verification step.  Running it over the full
        # frame doubles the dominant cost and rarely changes the peak location.
        x, y = point
        edge_local = cv2.matchTemplate(edges[y:y + h, x:x + w], template_edges,
                                       cv2.TM_CCOEFF_NORMED)
        edge_score = float(edge_local[0, 0])
        # Keep the intensity peak for localisation; its edge score rejects
        # smooth, look-alike patches introduced by captions/fades.
        score = intensity_score * 0.75 + edge_score * 0.25
        if score > best.score:
            best = Match(float(score), point, (w, h))
    return best


def score_video(cap, templates):
    names = sorted(templates)
    scores = {name: {state: [] for state in STATES[1:]} for name in names}
    matches: list[dict[str, dict[str, Match]]] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        features = preprocess(frame)
        frame_matches = {}
        for name in names:
            frame_matches[name] = {}
            for status in STATES[1:]:
                variants = templates[name].get(status, ())
                match = best_match(features, variants) if variants else Match(-1, (0, 0), (0, 0))
                scores[name][status].append(match.score)
                frame_matches[name][status] = match
        matches.append(frame_matches)
        if len(matches) % 100 == 0:
            print(f"scored {len(matches)} frames", flush=True)
    return scores, matches


def decode_states(status_scores, threshold: float, switch_cost: float) -> list[str]:
    """Viterbi decode with a cost for implausibly rapid state changes."""
    count = len(status_scores["real"])
    if count == 0:
        return []
    emission = np.zeros((count, len(STATES)), np.float32)
    for state_index, state in enumerate(STATES[1:], 1):
        # A score below threshold is evidence against presence.  Scaling makes
        # switch_cost interpretable in roughly 'strong frames'.
        emission[:, state_index] = (np.asarray(status_scores[state]) - threshold) * 12.0

    value = np.full((count, len(STATES)), -np.inf, np.float32)
    previous = np.zeros((count, len(STATES)), np.int8)
    value[0] = emission[0]
    for frame in range(1, count):
        for current in range(len(STATES)):
            transitions = value[frame - 1] - switch_cost
            transitions[current] += switch_cost
            # A direct real <-> portrait jump is rarer than appearing/disappearing.
            if current and current != 0:
                for old in range(1, len(STATES)):
                    if old != current:
                        transitions[old] -= switch_cost * 0.5
            old = int(np.argmax(transitions))
            previous[frame, current] = old
            value[frame, current] = transitions[old] + emission[frame, current]

    result = [0] * count
    result[-1] = int(np.argmax(value[-1]))
    for frame in range(count - 1, 0, -1):
        result[frame - 1] = int(previous[frame, result[frame]])
    return [STATES[index] for index in result]


def sql_for(states: dict[str, list[str]], frame: int) -> str:
    conditions = [
        f"(name = '{name}' AND status = '{values[frame]}')"
        for name, values in states.items()
        if values[frame] != "absent"
    ]
    if not conditions:
        conditions = ["1 = 0"]
    lines = [" OR ".join(conditions[i:i + 2]) for i in range(0, len(conditions), 2)]
    return "SELECT * FROM characters WHERE " + " OR\n".join(lines) + ";"


def write_debug_video(path: Path, source: Path, states, matches, fps: float):
    cap = cv2.VideoCapture(str(source))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width + 320, max(height, 420)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = np.full((max(height, 420), width + 320, 3), 22, np.uint8)
        canvas[:height, :width] = frame
        cv2.putText(canvas, f"frame {frame_index} / {frame_index/fps:.3f}s", (width+8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
        for row, (name, values) in enumerate(states.items()):
            status = values[frame_index]
            color = (80, 230, 100) if status == 'real' else (60, 180, 255) if status == 'portrait' else (150, 150, 150)
            score = matches[frame_index][name][status].score if status != 'absent' else max(m.score for m in matches[frame_index][name].values())
            cv2.putText(canvas, f"{name:11} {status:8} {score:.2f}", (width+8, 42+row*19),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1)
            if status == "absent":
                continue
            match = matches[frame_index][name][status]
            x, y = match.point
            w, h = match.size
            cv2.rectangle(canvas, (max(0,x), max(0,y)), (min(width-1,x+w), min(height-1,y+h)), color, 1)
            cv2.putText(canvas, name.replace('student', ''), (max(0,x), max(10,y-2)),
                        cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)
        writer.write(canvas)
        frame_index += 1
    cap.release()
    writer.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("video/original.mp4"))
    parser.add_argument("--templates", type=Path, default=Path("pattern_matches"))
    parser.add_argument("--output", type=Path, default=Path("SQL.yaml"))
    parser.add_argument("--debug-video", type=Path, default=Path("video/OpenCV_processing_process.mp4"))
    parser.add_argument("--threshold", type=float, default=.72)
    parser.add_argument("--switch-cost", type=float, default=None,
                        help='State change penalty (layout: 1, templates: 5).')
    parser.add_argument("--scales", type=float, nargs="+", default=[.9, 1.0, 1.1])
    parser.add_argument("--no-debug-video", action="store_true")
    parser.add_argument("--detector", choices=['layout', 'templates'], default='layout')
    parser.add_argument("--profile", type=Path, default=Path('recognition_profile'))
    parser.add_argument("--log", type=Path, default=Path('video/recognition.csv'))
    parser.add_argument("--scores-json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.switch_cost is None:
        args.switch_cost = 1.0 if args.detector == 'layout' else 5.0
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.setNumThreads(2)
    metadata = []
    if args.detector == 'layout':
        from layout_detector import LayoutDetector
        detector = LayoutDetector(args.profile)
        scores = {name: {state: [] for state in STATES[1:]} for name in detector.templates}
        matches = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            found, meta = detector.score(frame)
            # Profile coordinates are 640x360; logs/video use input pixels.
            sx, sy = frame.shape[1]/detector.size[0], frame.shape[0]/detector.size[1]
            found = {name: {state: Match(m.score, (round(m.point[0]*sx), round(m.point[1]*sy)),
                           (round(m.size[0]*sx), round(m.size[1]*sy))) for state,m in ss.items()}
                     for name,ss in found.items()}
            matches.append(found)
            metadata.append(meta)
            for name, ss in found.items():
                for state, match in ss.items():
                    scores[name][state].append(match.score)
            if len(matches) % 100 == 0:
                print(f'scored {len(matches)} frames', flush=True)
    else:
        templates = load_templates(args.templates, tuple(args.scales))
        scores, matches = score_video(cap, templates)
    cap.release()
    states = {name: decode_states(values, args.threshold, args.switch_cost) for name, values in scores.items()}
    frame_count = len(matches)
    document = {
        "meta": {"fps": fps, "frames": frame_count, "detector": args.detector,
                 "threshold": args.threshold, "switch_cost": args.switch_cost},
        "body": [{"frame": frame, "SQL": sql_for(states, frame)} for frame in range(frame_count)],
    }
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame','seconds','character','state','real_score','portrait_score',
                         'x','y','width','height','registration_inliers'])
        for frame in range(frame_count):
            for name, values in states.items():
                state = values[frame]
                m = matches[frame][name].get(state)
                writer.writerow([frame,round(frame/fps,4),name,state,
                    round(scores[name]['real'][frame],5),round(scores[name]['portrait'][frame],5),
                    *(m.point+m.size if m else ('','','','')),
                    metadata[frame]['registration_inliers'] if metadata else ''])
    if args.scores_json:
        args.scores_json.write_text(json.dumps({'scores':scores,'states':states,'metadata':metadata,
            'configuration':{'detector':args.detector,'threshold':args.threshold,'switch_cost':args.switch_cost}}),encoding='utf-8')
    if not args.no_debug_video:
        temp = args.debug_video.with_name(args.debug_video.stem + "_temp.mp4")
        write_debug_video(temp, args.input, states, matches, fps)
        command = ["ffmpeg", "-loglevel", "error", "-i", str(temp), "-i", str(args.input),
                   "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "18",
                   "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   "-c:a", "copy", "-y", str(args.debug_video)]
        subprocess.run(command, check=True)
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
