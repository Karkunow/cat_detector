"""Тести фільтра «людина в кадрі». Запуск: .venv/bin/python -m unittest test_person_blocks -v"""
import unittest

from main import person_blocks


class PersonBlocks(unittest.TestCase):
    def test_no_person_never_blocks(self):
        self.assertFalse(person_blocks({"cat": "white_passby", "confidence": 0.88}, False))

    def test_person_with_confident_cat_does_not_block(self):
        # події 28.10 00:16:40 (white_passby 0.885) і 00:17:31: білий кіт, прийнятий за людину
        self.assertFalse(person_blocks({"cat": "white_passby", "confidence": 0.885}, True))
        self.assertFalse(person_blocks({"cat": "black_inside", "confidence": 0.92}, True))

    def test_person_without_cat_blocks(self):
        self.assertTrue(person_blocks({"cat": "unknown", "confidence": 0.0}, True))
        self.assertTrue(person_blocks({"cat": "pass_through", "confidence": 0.89}, True))

    def test_person_with_weak_cat_blocks(self):
        self.assertTrue(person_blocks({"cat": "white_passby", "confidence": 0.3}, True))
        self.assertTrue(person_blocks({"cat": "white", "confidence": None}, True))


if __name__ == "__main__":
    unittest.main()
