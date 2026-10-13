#!/usr/bin/env python3
"""Local web UI server for playing Quantum Cat."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import secrets
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyspiel

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_ai import make_bot

if os.environ.get("QUANTUM_CAT_WEB_SKIP_TORCH") == "1":
  torch = None
  AZBeliefPolicyBot = None
  load_neural = None
else:
  try:
    import torch
    from quantum_cat_alphazero_torch import AZBeliefPolicyBot
    from quantum_cat_full_match_elo import load_neural
  except Exception:  # pragma: no cover - optional local model path.
    torch = None
    AZBeliefPolicyBot = None
    load_neural = None


STATIC = ROOT / "static"
COLORS = ["R", "B", "Y", "G"]
COLOR_NAMES = {"R": "Red", "B": "Blue", "Y": "Yellow", "G": "Green"}

BEST_CHECKPOINT = REPO_ROOT / "az_runs/best_3p_policy_checkpoint.pt"
BEST_EVIDENCE = (
    "az_runs/context_candidate_vs_champion_belief_policy_gate_180.json "
    "rating 1203.77"
)
DEFAULT_HUMAN_MATCH_LOG = REPO_ROOT / "experiments/human_match_logs/web_human_matches.jsonl"


def utc_now_iso():
  return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path):
  if not path.exists() or not path.is_file():
    return None
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


class HumanMatchLogger:
  """Append-only JSONL logger for completed human visualizer games."""

  def __init__(self, path: Path):
    self.path = Path(path)
    self._lock = threading.Lock()

  def write(self, record):
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with self._lock:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      with self.path.open("a", encoding="utf-8") as handle:
        handle.write(line)


HUMAN_MATCH_LOGGER = None


class BotFactory:
  def __init__(self):
    self._neural_cache = {}
    self.last_status = {}

  def make(self, players: int, mode: str, seed: int, seat: int):
    mode = mode or "best"
    if mode == "random":
      self.last_status[str(seat)] = {
          "name": "random",
          "source": "Uniform random baseline",
      }
      return make_bot("random", seed=seed + seat)
    if mode == "heuristic":
      self.last_status[str(seat)] = {
          "name": "heuristic_target2",
          "source": "Strong rule-aware fallback",
      }
      return make_bot("heuristic_target2", seed=seed + seat)

    bot = self._try_best_neural(players, seat)
    if bot is not None:
      return bot
    self.last_status[str(seat)] = {
        "name": "heuristic_target2",
        "source": "Fallback: champion checkpoint is only confirmed for 3-player games",
    }
    return make_bot("heuristic_target2", seed=seed + seat)

  def _try_best_neural(self, players: int, seat: int):
    if players != 3 or torch is None or load_neural is None or AZBeliefPolicyBot is None:
      return None
    checkpoint = BEST_CHECKPOINT
    if not checkpoint.exists():
      return None
    key = (players, str(checkpoint))
    try:
      if key not in self._neural_cache:
        torch.set_num_threads(1)
        device = torch.device("cpu")
        args = SimpleNamespace(
            belief_source="infostate",
            belief_candidates=8,
            belief_policy_temperature=1.0,
            belief_uniform_mix=0.15,
            belief_ref_policy_mix="model:1.0",
            belief_ref_policy_mix_by_phase="",
            belief_logprob_floor=1e-6,
            belief_ranker="",
            belief_ranker_candidates=64,
            belief_ranker_temperature=0.7,
            belief_ranker_uniform_mix=0.25,
            root_rollout_samples=2,
            root_rollouts=1,
            root_rollout_max_actions=4,
            root_rollout_top_policy=2,
            root_rollout_include_bots="heuristic,heuristic_target2,heuristic_adj2",
            root_rollout_max_plies=0,
            root_rollout_phases="",
            root_rollout_full_match=False,
        )
        model, model_args = load_neural(str(checkpoint), players, device, args)
        self._neural_cache[key] = (model, model_args, device)
      model, model_args, device = self._neural_cache[key]
      self.last_status[str(seat)] = {
          "name": "champion_belief_policy",
          "source": f"Best confirmed bot: {BEST_EVIDENCE}",
      }
      return AZBeliefPolicyBot(
          model,
          f"champion_belief_policy_p{seat}",
          device,
          model_args.value_scale,
          4,
          model_args,
      )
    except Exception as exc:  # Keep all supported games playable.
      self.last_status[str(seat)] = {
          "name": "heuristic_target2",
          "source": f"Fallback after champion load failed: {type(exc).__name__}",
      }
      return None


BOT_FACTORY = BotFactory()


class GameSession:
  def __init__(
      self,
      humans: int,
      bots: int,
      seed: int,
      bot_mode: str,
      match_logger: HumanMatchLogger | None = None,
  ):
    total = humans + bots
    if humans not in (1, 2):
      raise ValueError("Choose 1 or 2 human players.")
    if bots < 0 or bots > 5:
      raise ValueError("Choose between 0 and 5 bots.")
    if total < 2 or total > 5:
      raise ValueError("Quantum Cat supports 2 to 5 total seats in this engine.")
    self.id = secrets.token_hex(8)
    self.started_at = utc_now_iso()
    self.seed = int(seed)
    self.bot_mode = bot_mode or "best"
    self.match_logger = match_logger
    self._completion_logged = False
    self.rng = np.random.RandomState(self.seed)
    self.roles = ["human"] * humans + ["bot"] * bots
    self.game = pyspiel.load_game("python_quantum_cat", {"players": total})
    self.state = self.game.new_initial_state()
    self.bots = {
        seat: BOT_FACTORY.make(total, self.bot_mode, self.seed + 1000, seat)
        for seat, role in enumerate(self.roles)
        if role == "bot"
    }
    self.bot_status = {
        str(seat): BOT_FACTORY.last_status.get(str(seat), {})
        for seat in self.bots
    }
    self.log = []
    self.action_history = []
    self._log("system", f"New game: {humans} human, {bots} bot, seed {self.seed}.")
    self.advance()

  def _log(self, kind: str, text: str):
    self.log.append({"kind": kind, "text": text})
    del self.log[:-80]

  def advance(self):
    guard = 0
    while not self.state.is_terminal():
      guard += 1
      if guard > 300:
        raise RuntimeError("Turn advancement guard tripped.")
      current = int(self.state.current_player())
      if self.state.is_chance_node():
        outcomes = self.state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = int(self.rng.choice(actions, p=probs))
        self.state.apply_action(action)
        continue
      if self.roles[current] == "human":
        break
      bot = self.bots[current]
      legal = self.state.legal_actions(current)
      action = int(bot.step(self.state.clone(), current))
      if action not in legal:
        raise RuntimeError(f"Bot {current} chose illegal action {action}.")
      label = self.state.action_to_string(current, action)
      name = getattr(bot, "name", "bot")
      self._record_action("bot", current, action, label, name)
      self._log("bot", f"Player {current} ({name}) chose {label}.")
      self.state.apply_action(action)
    if self.state.is_terminal():
      if self._completion_logged:
        return
      returns = ", ".join(
          f"P{idx} {score:g}" for idx, score in enumerate(self.state.returns())
      )
      self._log("system", f"Game finished. Final returns: {returns}.")
      self._completion_logged = True
      self._write_human_match_log()

  def apply_human_action(self, action: int):
    if self.state.is_terminal():
      raise ValueError("The game is already finished.")
    player = int(self.state.current_player())
    if self.roles[player] != "human":
      raise ValueError("It is not a human player's turn.")
    legal = self.state.legal_actions(player)
    if action not in legal:
      raise ValueError("That action is not legal now.")
    label = self.state.action_to_string(player, action)
    self._record_action("human", player, action, label, None)
    self._log("human", f"Player {player} chose {label}.")
    self.state.apply_action(int(action))
    self.advance()

  def _record_action(self, actor: str, player: int, action: int, label: str,
                     bot_name: str | None):
    state = self.state
    self.action_history.append({
        "index": len(self.action_history),
        "actor": actor,
        "player": int(player),
        "role": self.roles[player],
        "botName": bot_name,
        "phase": phase_name(state),
        "phaseId": int(getattr(state, "_phase", -1)),
        "trickNumber": int(getattr(state, "_trick_number", 0)),
        "ledColor": getattr(state, "_led_color", None),
        "legalCount": int(len(state.legal_actions(player))),
        "action": int(action),
        "label": label,
    })

  def _write_human_match_log(self):
    if self.match_logger is None:
      return
    if "human" not in self.roles:
      return
    self.match_logger.write(self._human_match_record())

  def _human_match_record(self):
    state = self.state
    checkpoint_sha = file_sha256(BEST_CHECKPOINT)
    config = {
        "botMode": self.bot_mode,
        "botStatus": self.bot_status,
        "bestCheckpoint": (
            str(BEST_CHECKPOINT.relative_to(REPO_ROOT))
            if BEST_CHECKPOINT.exists() else None
        ),
        "bestCheckpointSha256": checkpoint_sha,
    }
    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schemaVersion": 1,
        "source": "quantum_cat_web",
        "sessionId": self.id,
        "startedAt": self.started_at,
        "completedAt": utc_now_iso(),
        "seed": self.seed,
        "players": len(self.roles),
        "humanCount": sum(1 for role in self.roles if role == "human"),
        "botCount": sum(1 for role in self.roles if role == "bot"),
        "eligibleHumanVsBot": (
            any(role == "human" for role in self.roles)
            and any(role == "bot" for role in self.roles)
        ),
        "roles": list(self.roles),
        "botConfigSha256": config_sha,
        "botConfig": config,
        "rounds": [{
            "roundIndex": 0,
            "returns": [float(x) for x in state.returns()],
            "rawRoundScores": [float(x) for x in getattr(state, "_last_raw_scores", [])],
            "tricksWon": [int(x) for x in getattr(state, "_tricks_won", [])],
            "predictions": [int(x) for x in getattr(state, "_predictions", [])],
            "paradoxed": [bool(x) for x in getattr(state, "_has_paradoxed", [])],
            "anyParadox": any(bool(x) for x in getattr(state, "_has_paradoxed", [])),
        }],
        "final": {
            "returns": [float(x) for x in state.returns()],
            "rawRoundScores": [float(x) for x in getattr(state, "_last_raw_scores", [])],
            "tricksWon": [int(x) for x in getattr(state, "_tricks_won", [])],
            "predictions": [int(x) for x in getattr(state, "_predictions", [])],
            "paradoxed": [bool(x) for x in getattr(state, "_has_paradoxed", [])],
            "anyParadox": any(bool(x) for x in getattr(state, "_has_paradoxed", [])),
            "completedTricks": self._completed_tricks_snapshot(limit=None),
            "boardOwnership": [
                [int(x) for x in row]
                for row in getattr(state, "_board_ownership", np.array([])).tolist()
            ],
        },
        "actions": list(self.action_history),
    }

  def snapshot(self):
    state = self.state
    current = None if state.is_terminal() else int(state.current_player())
    active_human = (
        current if current is not None and self.roles[current] == "human" else None
    )
    legal = state.legal_actions(current) if active_human is not None else []
    legal_set = set(legal)
    return {
        "sessionId": self.id,
        "seed": self.seed,
        "botMode": self.bot_mode,
        "botStatus": self.bot_status,
        "engineLimit": "2-5 total seats",
        "phase": phase_name(state),
        "phaseId": int(getattr(state, "_phase", -1)),
        "terminal": bool(state.is_terminal()),
        "currentPlayer": current,
        "currentRole": None if current is None else self.roles[current],
        "activeHuman": active_human,
        "players": [
            self._seat_snapshot(seat, active_human)
            for seat in range(len(self.roles))
        ],
        "rankCount": int(getattr(state, "_num_card_types", 0)),
        "colors": [
            {"code": code, "name": COLOR_NAMES[code]} for code in COLORS
        ],
        "board": self._board_snapshot(legal_set),
        "currentTrick": self._current_trick(),
        "completedTricks": self._completed_tricks(),
        "actions": [self._action_snapshot(action) for action in legal],
        "returns": [float(x) for x in state.returns()],
        "rawRoundScores": list(getattr(state, "_last_raw_scores", [])),
        "log": list(self.log),
        "summary": {
            "trickNumber": int(getattr(state, "_trick_number", 0)),
            "numTricks": int(getattr(state, "_num_tricks", 0)),
            "ledColor": getattr(state, "_led_color", None),
            "startPlayer": int(getattr(state, "_start_player", 0)),
        },
    }

  def _seat_snapshot(self, seat: int, active_human: int | None):
    state = self.state
    hand = None
    if active_human == seat:
      hand = [int(x) for x in state._hands[seat]]
    return {
        "seat": seat,
        "role": self.roles[seat],
        "isCurrent": (
            not state.is_terminal() and int(state.current_player()) == seat
        ),
        "tricks": int(state._tricks_won[seat]),
        "prediction": (
            None if int(state._predictions[seat]) < 0
            else int(state._predictions[seat])
        ),
        "discardedRank": (
            None if int(state._discarded_cards[seat]) < 0
            else int(state._discarded_cards[seat])
        ),
        "paradoxed": bool(state._has_paradoxed[seat]),
        "tokens": [
            bool(state._color_tokens[seat][idx]) for idx in range(len(COLORS))
        ],
        "hand": hand,
        "handCount": int(np.sum(state._hands[seat])),
        "return": float(state.returns()[seat]) if state.is_terminal() else None,
    }

  def _board_snapshot(self, legal_set: set[int]):
    state = self.state
    board = []
    ranks = int(state._num_card_types)
    for color_idx, color in enumerate(COLORS):
      row = []
      for rank_idx in range(ranks):
        action = color_idx * ranks + rank_idx
        owner = int(state._board_ownership[color_idx, rank_idx])
        row.append({
            "color": color,
            "rank": rank_idx + 1,
            "owner": owner if owner >= 0 else None,
            "blocked": owner == -2,
            "legalAction": action if action in legal_set else None,
        })
      board.append(row)
    return board

  def _current_trick(self):
    state = self.state
    result = []
    for seat, card in enumerate(state._cards_played_this_trick):
      result.append({
          "seat": seat,
          "card": None if card is None else {"rank": int(card[0]), "color": card[1]},
      })
    return result

  def _completed_tricks(self):
    return self._completed_tricks_snapshot(limit=6)

  def _completed_tricks_snapshot(self, limit: int | None):
    tricks = []
    source = self.state._completed_tricks if limit is None else self.state._completed_tricks[-limit:]
    offset = 0 if limit is None else max(0, len(self.state._completed_tricks) - limit)
    for local_idx, trick in enumerate(source):
      plays = []
      for seat, card in trick:
        plays.append({
            "seat": int(seat),
            "card": None if card is None else {"rank": int(card[0]), "color": card[1]},
        })
      tricks.append({"index": offset + local_idx, "plays": plays})
    return tricks

  def _action_snapshot(self, action: int):
    state = self.state
    label = state.action_to_string(state.current_player(), action)
    phase = int(getattr(state, "_phase", -1))
    item = {"id": int(action), "label": label, "kind": phase_name(state)}
    if action == 999:
      item.update({"kind": "paradox"})
    elif phase == 1:
      item.update({"kind": "discard", "rank": action + 1})
    elif phase == 2:
      item.update({"kind": "prediction", "prediction": action - 100})
    elif phase == 3:
      ranks = int(state._num_card_types)
      item.update({
          "kind": "play",
          "rank": int(action % ranks) + 1,
          "color": COLORS[int(action // ranks)],
      })
    return item


def phase_name(state):
  return {
      0: "dealing",
      1: "discard",
      2: "prediction",
      3: "play",
      4: "scoring",
  }.get(int(getattr(state, "_phase", -1)), "unknown")


SESSIONS = {}


class Handler(BaseHTTPRequestHandler):
  def do_GET(self):  # noqa: N802
    parsed = urlparse(self.path)
    if parsed.path == "/api/health":
      return self._json({"ok": True})
    path = "index.html" if parsed.path in ("", "/") else parsed.path.lstrip("/")
    full = (STATIC / path).resolve()
    if not str(full).startswith(str(STATIC.resolve())) or not full.exists():
      self.send_error(404)
      return
    ctype = mimetypes.guess_type(str(full))[0] or "application/octet-stream"
    data = full.read_bytes()
    self.send_response(200)
    self.send_header("Content-Type", ctype)
    self.send_header("Cache-Control", "no-store, max-age=0")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def do_POST(self):  # noqa: N802
    try:
      data = self._read_json()
      if self.path == "/api/new":
        humans = int(data.get("humanPlayers", 1))
        bots = int(data.get("botPlayers", 2))
        seed = int(data.get("seed", 20260605))
        bot_mode = str(data.get("botMode", "best"))
        session = GameSession(humans, bots, seed, bot_mode, HUMAN_MATCH_LOGGER)
        SESSIONS[session.id] = session
        return self._json(session.snapshot())
      if self.path == "/api/action":
        session = SESSIONS.get(str(data.get("sessionId", "")))
        if session is None:
          raise ValueError("Unknown session.")
        session.apply_human_action(int(data["action"]))
        return self._json(session.snapshot())
      self.send_error(404)
    except Exception as exc:
      traceback.print_exc()
      self._json({"error": str(exc)}, status=400)

  def _read_json(self):
    length = int(self.headers.get("Content-Length", "0"))
    if length <= 0:
      return {}
    return json.loads(self.rfile.read(length).decode("utf-8"))

  def _json(self, payload, status=200):
    data = json.dumps(payload).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def log_message(self, fmt, *args):
    sys.stderr.write("%s\n" % (fmt % args))


def main():
  global HUMAN_MATCH_LOGGER
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument(
      "--human-match-log",
      default=str(DEFAULT_HUMAN_MATCH_LOG),
      help="Append completed human visualizer games to this JSONL file.",
  )
  parser.add_argument(
      "--no-human-match-log",
      action="store_true",
      help="Disable completed human game disk logging.",
  )
  args = parser.parse_args()
  if not args.no_human_match_log:
    HUMAN_MATCH_LOGGER = HumanMatchLogger(Path(args.human_match_log))
    print(f"Human match log: {HUMAN_MATCH_LOGGER.path}", flush=True)
  server = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"Quantum Cat web UI: http://{args.host}:{args.port}", flush=True)
  server.serve_forever()


if __name__ == "__main__":
  main()
