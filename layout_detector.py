"""Reference-image registration for this video's fixed character layout.

The profile contains actual face crops and reference images, never a timeline of
expected states. A similarity transform follows the camera; each face is scored
only in its own seat. Portraits belong to a separate, stationary screen layer.
"""
from pathlib import Path
import json

import cv2
import numpy as np

from make_SQL_yaml_by_video import Match, preprocess, read_image


class LayoutDetector:
    def __init__(self, root: Path):
        self.config = json.loads((root / 'profile.json').read_text())
        self.size = tuple(self.config['source_size'])
        self.sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.02)
        self.matcher = cv2.BFMatcher()
        self.references = []
        for file, transform in [('group.png', np.eye(2, 3)),
                                ('solo.png', np.eye(2, 3)),
                                ('opening.png', self.config['opening_transform'])]:
            im = read_image(root / file)
            mask = np.zeros(im.shape[:2], np.uint8)
            if file == 'group.png':
                mask[85:330, 70:550] = 255
            elif file == 'solo.png':
                mask[145:340, 290:350] = 255
            else:
                mask[:] = 255
            keys, desc = self.sift.detectAndCompute(im, mask)
            self.references.append((keys, desc, np.asarray(transform, np.float64)))
        self.templates = {}
        for name, states in self.config['characters'].items():
            self.templates[name] = {
                state: [preprocess(read_image(root / file)) for file in spec['images']]
                for state, spec in states.items()
            }

    def register(self, frame):
        # Recover feature contrast during fades without changing score brightness.
        peak = float(np.percentile(frame, 99))
        if 3 < peak < 120:
            frame = np.clip(frame.astype(np.float32) * (150 / peak), 0, 255).astype(np.uint8)
        keys, desc = self.sift.detectAndCompute(frame, None)
        if desc is None or len(keys) < 8:
            return None, 0
        for ref_keys, ref_desc, canonical in self.references:
            pairs = self.matcher.knnMatch(ref_desc, desc, k=2)
            good = [a for pair in pairs if len(pair) == 2 for a, b in [pair]
                    if a.distance < .70 * b.distance]
            if len(good) < 8:
                continue
            a = np.float32([ref_keys[m.queryIdx].pt for m in good])
            b = np.float32([keys[m.trainIdx].pt for m in good])
            cv2.setRNGSeed(0)
            transform, mask = cv2.estimateAffinePartial2D(
                a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5)
            if transform is None or int(mask.sum()) < 8:
                continue
            full = np.vstack([transform, [0, 0, 1]]) @ np.vstack([canonical, [0, 0, 1]])
            scale = np.hypot(full[0, 0], full[1, 0])
            # This video zooms about its centre and never rotates the camera.
            centre = full @ np.array([320, 180, 1])
            if not (.8 <= scale <= 12 and abs(full[1, 0]) < .025 * scale
                    and np.linalg.norm(centre[:2] - [320, 180]) < 65):
                continue
            return full[:2], int(mask.sum())
        return None, 0

    @staticmethod
    def local_match(features, variants, box, valid=None):
        gray, _ = features
        x, y, w, h = box
        pad = 4
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(gray.shape[1], x+w+pad), min(gray.shape[0], y+h+pad)
        if valid is not None and np.mean(valid[y:y+h, x:x+w] > 250) < .25:
            return Match(-1., (x, y), (w, h))
        best = Match(-1., (x, y), (w, h))
        for template, _ in variants:
            # White lyrics obscure the faces. Compute NCC over visible, non-text
            # pixels only, including cropped heads at the screen boundary.
            patches = np.lib.stride_tricks.sliding_window_view(gray[y0:y1, x0:x1], (h, w)).astype(np.float32)
            weights = (patches < 160) & (patches < template * 1.5 + 15)
            if valid is not None:
                weights &= np.lib.stride_tricks.sliding_window_view(valid[y0:y1, x0:x1], (h, w)) > 250
            weights = weights.astype(np.float32)
            count = weights.sum(axis=(-2, -1))
            safe_count = np.maximum(count, 1)
            avg = (patches * weights).sum(axis=(-2, -1)) / safe_count
            tavg = (template * weights).sum(axis=(-2, -1)) / safe_count
            a = (patches - avg[..., None, None]) * weights
            b = (template - tavg[..., None, None]) * weights
            numerator = (a*b).sum(axis=(-2, -1))
            variance = (a*a).sum(axis=(-2, -1))
            denom = np.sqrt(variance * (b*b).sum(axis=(-2, -1)))
            intensity = numerator / np.maximum(denom, 1e-6)
            intensity[(count < w*h*.25) | (variance / safe_count < 9)] = -1
            _, score, _, point = cv2.minMaxLoc(intensity)
            px, py = point[0]+x0, point[1]+y0
            value = float(score)
            # Correlation alone is invariant even when the image has faded to black.
            if float(gray[py:py+h, px:px+w].std()) < 3:
                value = -1.
            if value > best.score:
                best = Match(float(value), (px, py), (w, h))
        return best

    def score(self, frame):
        frame = cv2.resize(frame, self.size)
        transform, inliers = self.register(frame)
        if transform is None:
            # The fixed wide shot can still be scored when no body features are
            # visible; this is not a carry-forward of an earlier detection.
            transform = np.eye(2, 3)
        features = preprocess(frame)
        if transform is not None:
            inverse = cv2.invertAffineTransform(transform)
            normal = cv2.warpAffine(frame, inverse, self.size)
            valid = cv2.warpAffine(np.full(frame.shape[:2], 255, np.uint8), inverse, self.size)
            real_features = preprocess(normal)
        matches = {}
        for name, states in self.config['characters'].items():
            matches[name] = {}
            for state, spec in states.items():
                if state == 'real' and transform is None:
                    match = Match(-1., (0, 0), (0, 0))
                else:
                    match = self.local_match(real_features if state == 'real' else features,
                                             self.templates[name][state], spec['box'],
                                             valid if state == 'real' else None)
                    if state == 'real':
                        x, y = match.point
                        w, h = match.size
                        corners = cv2.transform(np.float32([[[x, y], [x+w, y+h]]]), transform)[0]
                        p, q = np.rint(corners).astype(int)
                        match = Match(match.score, tuple(map(int, p)), tuple(map(int, q-p)))
                matches[name][state] = match
        return matches, {'registration_inliers': inliers,
                         'transform': transform.tolist() if transform is not None else None}
