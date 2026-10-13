import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


def _load_server_module():
  os.environ["QUANTUM_CAT_WEB_SKIP_TORCH"] = "1"
  path = Path(__file__).resolve().parent / "quantum_cat_web" / "server.py"
  spec = importlib.util.spec_from_file_location("quantum_cat_web_server", path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


server = _load_server_module()


class HumanMatchLoggingTest(unittest.TestCase):

  def test_completed_human_bot_game_writes_one_jsonl_record(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "human_matches.jsonl"
      logger = server.HumanMatchLogger(path)
      session = server.GameSession(
          humans=1,
          bots=1,
          seed=20297101,
          bot_mode="heuristic",
          match_logger=logger,
      )

      for _ in range(200):
        if session.state.is_terminal():
          break
        snapshot = session.snapshot()
        self.assertEqual(snapshot["currentRole"], "human")
        self.assertTrue(snapshot["actions"])
        session.apply_human_action(int(snapshot["actions"][0]["id"]))
      self.assertTrue(session.state.is_terminal())

      lines = path.read_text(encoding="utf-8").splitlines()
      self.assertEqual(len(lines), 1)
      record = json.loads(lines[0])
      self.assertEqual(record["schemaVersion"], 1)
      self.assertEqual(record["source"], "quantum_cat_web")
      self.assertTrue(record["eligibleHumanVsBot"])
      self.assertEqual(record["players"], 2)
      self.assertEqual(record["humanCount"], 1)
      self.assertEqual(record["botCount"], 1)
      self.assertEqual(len(record["rounds"]), 1)
      self.assertEqual(len(record["final"]["returns"]), 2)
      self.assertEqual(len(record["final"]["paradoxed"]), 2)
      self.assertTrue(record["actions"])
      self.assertIn("botConfigSha256", record)
      self.assertNotIn("discardedRanks", record["rounds"][0])
      self.assertNotIn("discardedRanks", record["final"])

      session.advance()
      self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
  unittest.main()
