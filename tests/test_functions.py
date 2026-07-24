import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import split_complete_sentences
from src import functions


class TestSentenceSplitting(unittest.TestCase):
    def test_splits_complete_sentences(self):
        buf = "Hello there. How are you? "
        complete, tail = split_complete_sentences(buf)
        self.assertEqual(complete, ["Hello there.", "How are you?"])
        self.assertEqual(tail, "")

    def test_keeps_incomplete_tail(self):
        buf = "The weather today is"
        complete, tail = split_complete_sentences(buf)
        self.assertEqual(complete, [])
        self.assertEqual(tail, "The weather today is")

    def test_mixed(self):
        buf = "First sentence. Second is still going"
        complete, tail = split_complete_sentences(buf)
        self.assertEqual(complete, ["First sentence."])
        self.assertEqual(tail, "Second is still going")


class TestReminders(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_file = functions.Config.REMINDERS_FILE
        functions.Config.REMINDERS_FILE = os.path.join(self.tmpdir, "reminders.json")

    def tearDown(self):
        functions.Config.REMINDERS_FILE = self._orig_file

    def test_set_and_list_reminder(self):
        msg = functions.set_reminder("water the plants", 10)
        self.assertIn("water the plants", msg)
        listing = functions.list_reminders()
        self.assertIn("water the plants", listing)


if __name__ == "__main__":
    unittest.main()
