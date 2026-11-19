import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_utils import SentenceChunker, clean_for_speech, split_complete_sentences


class TestSentenceSplitting(unittest.TestCase):
    def test_splits_complete_sentences(self):
        complete, tail = split_complete_sentences("Hello there. How are you? ")
        self.assertEqual(complete, ["Hello there.", "How are you?"])
        self.assertEqual(tail, "")

    def test_keeps_incomplete_tail(self):
        self.assertEqual(split_complete_sentences("The weather today is"),
                         ([], "The weather today is"))

    def test_mixed(self):
        complete, tail = split_complete_sentences("First sentence. Second is still going")
        self.assertEqual(complete, ["First sentence."])
        self.assertEqual(tail, "Second is still going")

    def test_abbreviations_do_not_split(self):
        complete, tail = split_complete_sentences("Dr. Smith lives in the U.S. now. He is happy. ")
        self.assertEqual(complete, ["Dr. Smith lives in the U.S. now.", "He is happy."])

    def test_eg_and_initials(self):
        complete, _ = split_complete_sentences("Use fruit, e.g. apples. Ask J. Smith. ")
        self.assertEqual(complete, ["Use fruit, e.g. apples.", "Ask J. Smith."])

    def test_decimals_and_growing_numbers_stay_in_tail(self):
        # "3." at the end of the buffer could still become "3.5"
        self.assertEqual(split_complete_sentences("It costs 3."), ([], "It costs 3."))
        complete, _ = split_complete_sentences("It costs 3.5 dollars. Fine. ")
        self.assertEqual(complete, ["It costs 3.5 dollars.", "Fine."])

    def test_numbered_list_markers(self):
        complete, _ = split_complete_sentences("Steps: 1. Mix. 2. Bake. ")
        self.assertEqual(complete, ["Steps: 1. Mix.", "2. Bake."])

    def test_question_and_exclamation_always_split(self):
        complete, _ = split_complete_sentences("Really? Yes! Ok. ")
        self.assertEqual(complete, ["Really?", "Yes!", "Ok."])


class TestChunker(unittest.TestCase):
    def feed_all(self, chunker, pieces):
        out = []
        for p in pieces:
            out += chunker.feed(p)
        out += chunker.flush()
        return out

    def test_sentence_mode_matches_plain_splitting(self):
        c = SentenceChunker(0)
        out = self.feed_all(c, ["Hello ", "world. Second ", "one"])
        self.assertEqual(out, ["Hello world.", "Second one"])

    def test_clause_flush_only_for_first_chunk(self):
        c = SentenceChunker(first_chunk_min_chars=20)
        out = []
        out += c.feed("Well, the answer to your very long question, is complex")
        # first chunk cut at a clause break after >= 20 chars
        self.assertEqual(out, ["Well, the answer to your very long question,"])
        out += c.feed(", and more, and more things here, but ")
        # no further clause cuts: only sentences from now on
        self.assertEqual(len(out), 1)
        out += c.flush()
        self.assertEqual(len(out), 2)

    def test_short_clause_not_cut(self):
        c = SentenceChunker(first_chunk_min_chars=40)
        self.assertEqual(c.feed("Yes, I can "), [])

    def test_flush_returns_tail_once(self):
        c = SentenceChunker(0)
        c.feed("no end")
        self.assertEqual(c.flush(), ["no end"])
        self.assertEqual(c.flush(), [])


class TestCleanForSpeech(unittest.TestCase):
    def test_strips_markdown_and_urls(self):
        text = "**Bold** and `code` see [docs](http://x.io) or http://y.io now"
        self.assertEqual(clean_for_speech(text), "Bold and code see docs or now")

    def test_strips_list_markers(self):
        self.assertEqual(clean_for_speech("- one\n- two"), "one two")


if __name__ == "__main__":
    unittest.main()
