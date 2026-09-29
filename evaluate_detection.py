"""Evaluate saved predictions against independently inspected source frames."""
import argparse
import json
from pathlib import Path

STATES = ('absent', 'real', 'portrait')


def evaluate(labels, prediction):
    reports = {}
    for split in ('development', 'validation', 'all'):
        matrix = {truth: dict.fromkeys(STATES, 0) for truth in STATES}
        errors, exact, frames = [], 0, 0
        for row in labels['frames']:
            if split != 'all' and row['split'] != split:
                continue
            frame = row['frame']
            if 'states' not in prediction and str(frame) not in prediction:
                continue
            frame_errors = 0
            for name, truth in row['states'].items():
                if 'states' in prediction:
                    actual = prediction['states'][name][frame]
                else:
                    sample = prediction[str(frame)]
                    sample = sample.get('matches', sample)
                    scores = {s: m['score'] for s, m in sample[name].items()}
                    actual = max(scores, key=scores.get)
                    if scores[actual] < .72:
                        actual = 'absent'
                matrix[truth][actual] += 1
                if actual != truth:
                    frame_errors += 1
                    errors.append({'frame':frame, 'character':name, 'expected':truth, 'actual':actual})
            exact += not frame_errors
            frames += 1
        total = sum(sum(row.values()) for row in matrix.values())
        if not total:
            continue
        correct = sum(matrix[state][state] for state in STATES)
        reports[split] = {
            'frames':frames, 'character_states':total, 'correct':correct,
            'accuracy':correct/total, 'exact_frames':exact,
            'exact_frame_accuracy':exact/frames, 'confusion_matrix':matrix,
            'real_recall':matrix['real']['real']/max(1,sum(matrix['real'].values())),
            'real_precision':matrix['real']['real']/max(1,sum(row['real'] for row in matrix.values())),
            'errors':errors,
        }
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--labels', type=Path, default=Path('evaluation/labels.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.labels.read_text(encoding='utf-8')),
                      json.loads(args.predictions.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for split, metrics in result.items():
        print(split, {k:v for k,v in metrics.items() if k not in ('errors','confusion_matrix')})
