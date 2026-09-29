import unittest
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from make_SQL_yaml_by_video import decode_states, sql_for
from layout_detector import LayoutDetector


class TemporalDecoderTest(unittest.TestCase):
    def test_removes_short_false_positive(self):
        scores = {"real": [.2] * 5 + [.9] + [.2] * 5, "portrait": [.1] * 11}
        self.assertEqual(decode_states(scores, .72, 5), ["absent"] * 11)

    def test_keeps_sustained_detection_and_fills_dropout(self):
        real = [.2] * 5 + [.95] * 8 + [.1] + [.95] * 8 + [.2] * 5
        result = decode_states({"real": real, "portrait": [.1] * len(real)}, .72, 5)
        self.assertEqual(result[6:21], ["real"] * 15)

    def test_layout_keeps_deliberate_two_frame_appearance(self):
        real = [.3]*5 + [.97]*2 + [.3]*5
        states = decode_states({'real':real,'portrait':[.1]*len(real)}, .72, 1)
        self.assertEqual(states, ['absent']*5 + ['real']*2 + ['absent']*5)

    def test_sql_is_valid_for_empty_frame(self):
        self.assertEqual(sql_for({"studentA": ["absent"]}, 0),
                         "SELECT * FROM characters WHERE 1 = 0;")

    def test_sql_executes_with_more_than_two_characters(self):
        connection = sqlite3.connect(':memory:')
        connection.execute('CREATE TABLE characters (name TEXT, status TEXT)')
        rows = [('studentA','real'),('studentB','portrait'),('teacher','real')]
        connection.executemany('INSERT INTO characters VALUES (?,?)', rows)
        query = sql_for({name:[state] for name,state in rows}, 0)
        self.assertEqual(connection.execute(query).fetchall(), rows)
        connection.close()


class LayoutScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = LayoutDetector(Path(__file__).parent / 'recognition_profile')

    def test_black_frame_has_no_detections(self):
        matches, _ = self.detector.score(np.zeros((360,640,3), np.uint8))
        self.assertTrue(all(m.score < .72 for states in matches.values() for m in states.values()))

    def test_face_with_white_text_is_still_detected(self):
        template = self.detector.templates['studentC']['real'][0]
        gray = np.full((60,70), 30, np.uint8)
        h,w = template[0].shape
        gray[10:10+h,10:10+w] = template[0]
        cv2.line(gray, (11,22), (10+w-2,22), 240, 3)
        match = self.detector.local_match((gray,cv2.Canny(gray,45,120)), [template], (10,10,w,h))
        self.assertGreater(match.score, .9)

    def test_partly_offscreen_face_uses_visible_pixels(self):
        template = self.detector.templates['studentC']['real'][0]
        gray = np.zeros((60,70), np.uint8)
        h,w = template[0].shape
        gray[10:10+h,10:10+w] = template[0]
        valid = np.full(gray.shape, 255, np.uint8)
        valid[:10+h//2] = 0
        gray[valid==0] = 0
        match = self.detector.local_match((gray,cv2.Canny(gray,45,120)), [template], (10,10,w,h), valid)
        self.assertGreater(match.score, .95)


if __name__ == "__main__":
    unittest.main()
