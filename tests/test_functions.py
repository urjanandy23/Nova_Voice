import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import functions


class ReminderTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig = functions.Config.REMINDERS_FILE
        functions.Config.REMINDERS_FILE = os.path.join(self.tmpdir, "reminders.json")

    def tearDown(self):
        functions.Config.REMINDERS_FILE = self._orig


class TestReminders(ReminderTestCase):
    def test_set_and_list(self):
        self.assertIn("water the plants", functions.set_reminder("water the plants", 10))
        listing = functions.list_reminders()
        self.assertIn("water the plants", listing)
        self.assertIn("in 10 minutes", listing)

    def test_rejects_bad_input(self):
        self.assertTrue(functions.set_reminder("x", 0).startswith("Error"))
        self.assertTrue(functions.set_reminder("x", -5).startswith("Error"))
        self.assertTrue(functions.set_reminder("   ", 5).startswith("Error"))

    def test_cancel(self):
        functions.set_reminder("call mom", 20)
        functions.set_reminder("buy milk", 30)
        self.assertIn("Cancelled 1", functions.cancel_reminder("MOM"))
        listing = functions.list_reminders()
        self.assertNotIn("call mom", listing)
        self.assertIn("buy milk", listing)
        self.assertIn("No pending", functions.cancel_reminder("nothing like this"))

    def test_due_reminders_fire_once(self):
        functions.set_reminder("soon", 1)
        later = datetime.now().astimezone() + timedelta(minutes=2)
        self.assertEqual(functions.collect_due_reminders(now=later), ["soon"])
        self.assertEqual(functions.collect_due_reminders(now=later), [])
        self.assertEqual(functions.list_reminders(), "No pending reminders.")

    def test_not_yet_due_does_not_fire(self):
        functions.set_reminder("later", 60)
        self.assertEqual(functions.collect_due_reminders(), [])

    def test_naive_legacy_timestamps_are_handled(self):
        past = (datetime.now() - timedelta(minutes=5)).isoformat()  # naive, old format
        with open(functions.Config.REMINDERS_FILE, "w") as f:
            json.dump([{"text": "old", "fire_at": past, "fired": False}], f)
        self.assertEqual(functions.collect_due_reminders(), ["old"])

    def test_corrupt_file_does_not_crash(self):
        with open(functions.Config.REMINDERS_FILE, "w") as f:
            f.write("{not json")
        self.assertEqual(functions.list_reminders(), "No pending reminders.")


class TestDispatch(ReminderTestCase):
    def test_missing_argument_returns_error_text(self):
        out = functions.dispatch_tool("set_reminder", {"text": "x"})
        self.assertTrue(out.startswith("Error"))

    def test_wrong_type_returns_error_text(self):
        out = functions.dispatch_tool("set_reminder", {"text": "x", "minutes_from_now": "soon"})
        self.assertTrue(out.startswith("Error"))

    def test_json_string_arguments(self):
        out = functions.dispatch_tool("set_reminder",
                                      json.dumps({"text": "x", "minutes_from_now": 5}))
        self.assertIn("Reminder set", out)

    def test_unknown_tool(self):
        self.assertIn("Unknown tool", functions.dispatch_tool("nope", {}))

    def test_empty_search_query(self):
        self.assertTrue(functions.dispatch_tool("search_web", {"query": " "}).startswith("Error"))

    def test_every_tool_schema_has_a_dispatch_branch(self):
        for tool in functions.TOOLS:
            self.assertNotIn("Unknown tool", functions.dispatch_tool(tool["name"], {}) or "")


if __name__ == "__main__":
    unittest.main()
