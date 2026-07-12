import unittest

from make_SQL_yaml_by_video import decode_states, sql_for


class TemporalDecoderTest(unittest.TestCase):
    def test_removes_short_false_positive(self):
        scores = {"real": [.2] * 5 + [.9] + [.2] * 5, "portrait": [.1] * 11}
        self.assertEqual(decode_states(scores, .72, 5), ["absent"] * 11)

    def test_keeps_sustained_detection_and_fills_dropout(self):
        real = [.2] * 5 + [.95] * 8 + [.1] + [.95] * 8 + [.2] * 5
        result = decode_states({"real": real, "portrait": [.1] * len(real)}, .72, 5)
        self.assertEqual(result[6:21], ["real"] * 15)

    def test_sql_is_valid_for_empty_frame(self):
        self.assertEqual(sql_for({"studentA": ["absent"]}, 0),
                         "SELECT * FROM characters WHERE 1 = 0;")


if __name__ == "__main__":
    unittest.main()
