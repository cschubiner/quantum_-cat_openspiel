#!/usr/bin/env python3
"""Browser visualizer and play surface for python_quantum_cat."""

from __future__ import annotations

import argparse
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_ai import make_bot
from quantum_cat_alphazero_torch import (
    AZBeliefPolicyBot,
    AZBeliefSearchBot,
    AZPolicyBot,
    load_model_payload,
)


COLORS = ["R", "B", "Y", "G"]
COLOR_NAMES = {"R": "Red", "B": "Blue", "Y": "Yellow", "G": "Green"}


class VisualizerGame:
  def __init__(
      self,
      players=3,
      human_seat=0,
      bot_name="heuristic_target2",
      seed=None,
      checkpoint_path=None,
      belief_samples=4,
      belief_sims=8,
  ):
    self.players = players
    self.human_seat = human_seat
    self.bot_name = bot_name
    self.checkpoint_path = checkpoint_path
    self.belief_samples = belief_samples
    self.belief_sims = belief_sims
    self.seed = random.randrange(1 << 30) if seed is None else seed
    self.game = pyspiel.load_game("python_quantum_cat", {"players": players})
    self.rng = np.random.default_rng(self.seed)
    self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    self.neural_model = None
    self.neural_args = None
    self.neural_players = None
    self._load_neural_checkpoint()
    self.bots = [self._make_bot(seat) for seat in range(players)]
    self.state = self.game.new_initial_state()
    self.history = []
    self._advance_non_human()

  def reset(self, players=None, human_seat=None, bot_name=None, seed=None):
    next_players = self.players if players is None else players
    next_human_seat = self.human_seat if human_seat is None else human_seat
    next_bot_name = self.bot_name if bot_name is None else bot_name
    self._validate_bot_config(next_players, next_bot_name)
    self.players = next_players
    self.human_seat = next_human_seat
    self.bot_name = next_bot_name
    self.seed = random.randrange(1 << 30) if seed is None else seed
    self.game = pyspiel.load_game("python_quantum_cat", {"players": self.players})
    self.rng = np.random.default_rng(self.seed)
    self.bots = [self._make_bot(seat) for seat in range(self.players)]
    self.state = self.game.new_initial_state()
    self.history = []
    self._advance_non_human()

  def apply_human_action(self, action):
    if self.state.is_terminal():
      raise ValueError("Game is already over")
    player = self.state.current_player()
    if player != self.human_seat:
      raise ValueError(f"Waiting on player {player}, not human seat {self.human_seat}")
    legal = self.state.legal_actions(player)
    if action not in legal:
      raise ValueError(f"Illegal action {action}; legal actions are {legal}")
    self._apply_action(action, source="human")
    self._advance_non_human()

  def suggestion(self):
    if self.state.is_terminal() or self.state.is_chance_node():
      return None
    player = self.state.current_player()
    legal = self.state.legal_actions(player)
    if player != self.human_seat or not legal:
      return None
    bot = self._make_bot(self.human_seat, seed_offset=9999 + len(self.history))
    action = bot.step(self.state, player)
    if action not in legal:
      action = legal[0]
    return self._action_payload(player, action)

  def payload(self, reveal=False):
    state = self.state
    current = state.current_player() if not state.is_terminal() else None
    legal = []
    if current == self.human_seat and not state.is_terminal():
      legal = [self._action_payload(current, action)
               for action in state.legal_actions(current)]
    hands = []
    for seat in range(self.players):
      hand = []
      for rank_idx, count in enumerate(state._hands[seat]):
        for _ in range(int(count)):
          hand.append({"rank": rank_idx + 1})
      hands.append({
          "seat": seat,
          "label": "You" if seat == self.human_seat else f"Bot {seat}",
          "isHuman": seat == self.human_seat,
          "cards": hand if reveal or seat == self.human_seat else [],
          "count": len(hand),
      })
    board = []
    for color_idx, color in enumerate(COLORS):
      row = []
      for rank_idx in range(state._num_card_types):
        owner = int(state._board_ownership[color_idx, rank_idx])
        row.append({
            "color": color,
            "rank": rank_idx + 1,
            "owner": owner,
            "blocked": owner == -2,
        })
      board.append(row)
    return {
        "players": self.players,
        "humanSeat": self.human_seat,
        "seed": self.seed,
        "phase": int(state._phase),
        "phaseName": self._phase_name(),
        "currentPlayer": current,
        "terminal": state.is_terminal(),
        "returns": [float(x) for x in state.returns()] if state.is_terminal() else None,
        "tricksWon": [int(x) for x in state._tricks_won],
        "predictions": [int(x) for x in state._predictions],
        "paradoxed": [bool(x) for x in state._has_paradoxed],
        "ledColor": state._led_color,
        "trick": int(state._trick_number),
        "board": board,
        "hands": hands,
        "legalActions": legal,
        "suggestion": self.suggestion(),
        "history": self.history[-80:],
        "completedTricks": [
            [
                {"seat": seat, "rank": card[0], "color": card[1]}
                for seat, card in trick
            ]
            for trick in state._completed_tricks
        ],
    }

  def _advance_non_human(self):
    while not self.state.is_terminal():
      if self.state.is_chance_node():
        outcomes = self.state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = int(self.rng.choice(actions, p=probs))
        self.state.apply_action(action)
        continue
      player = self.state.current_player()
      if player == self.human_seat:
        break
      action = self.bots[player].step(self.state, player)
      self._apply_action(action, source="bot")

  def _load_neural_checkpoint(self):
    if not self.checkpoint_path:
      return
    checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
    saved_args = checkpoint.get("args", {})
    self.neural_players = int(saved_args.get("players", self.players))
    checkpoint_game = pyspiel.load_game(
        "python_quantum_cat",
        {
            "players": self.neural_players,
            "match_context": int(saved_args.get("match_context", False)),
        },
    )
    loader_args = argparse.Namespace(
        players=self.neural_players,
        arch=saved_args.get("arch", "mlp"),
        width=saved_args.get("width", 256),
        depth=saved_args.get("depth", 3),
    )
    self.neural_model, _, saved_args = load_model_payload(
        self.checkpoint_path, checkpoint_game, loader_args, self.device
    )
    self.neural_model.eval()
    self.neural_args = argparse.Namespace(
        value_scale=saved_args.get("value_scale", 20.0),
        c_puct=saved_args.get("c_puct", 1.8),
        sims=self.belief_sims,
    )

  def _validate_bot_config(self, players, bot_name):
    if bot_name not in ("az_policy", "az_belief_policy", "az_belief_search"):
      return
    if self.neural_model is None:
      raise ValueError("No neural checkpoint loaded for neural bot mode")
    if players != self.neural_players:
      raise ValueError(
          f"Neural checkpoint supports {self.neural_players} players; "
          f"requested game has {players}"
      )

  def _make_bot(self, seat, seed_offset=0):
    if self.bot_name in ("az_policy", "az_belief_policy", "az_belief_search"):
      self._validate_bot_config(self.players, self.bot_name)
      if self.bot_name == "az_policy":
        return AZPolicyBot(
            self.neural_model, self.bot_name, self.device,
            self.neural_args.value_scale
        )
      if self.bot_name == "az_belief_policy":
        return AZBeliefPolicyBot(
            self.neural_model, self.bot_name, self.device,
            self.neural_args.value_scale, self.belief_samples
        )
      return AZBeliefSearchBot(
          self.neural_model, self.bot_name, self.device, self.neural_args,
          self.belief_samples, self.belief_sims
      )
    return make_bot(self.bot_name, seed=self.seed + seat * 101 + seed_offset)

  def _apply_action(self, action, source):
    player = self.state.current_player()
    label = self.state.action_to_string(player, action)
    self.state.apply_action(action)
    self.history.append({
        "seat": player,
        "source": source,
        "action": int(action),
        "label": label,
    })

  def _action_payload(self, player, action):
    if self.state._phase == 1:
      decoded = {"kind": "discard", "rank": int(action) + 1}
    elif self.state._phase == 2:
      decoded = {"kind": "prediction", "prediction": int(action) - 100}
    else:
      decoded = decode_action(action, self.state._num_card_types)
    return {
        "action": int(action),
        "label": self.state.action_to_string(player, action),
        **decoded,
    }

  def _phase_name(self):
    return {
        0: "Deal",
        1: "Discard",
        2: "Bid",
        3: "Trick",
        4: "Game Over",
    }.get(int(self.state._phase), str(self.state._phase))


def decode_action(action, num_card_types):
  if action == 999:
    return {"kind": "paradox"}
  if action >= 100:
    return {"kind": "prediction", "prediction": action - 100}
  color_idx = action // num_card_types
  rank_idx = action % num_card_types
  if 0 <= color_idx < len(COLORS):
    return {
        "kind": "play",
        "rank": rank_idx + 1,
        "color": COLORS[color_idx],
        "colorName": COLOR_NAMES[COLORS[color_idx]],
    }
  return {"kind": "discard", "rank": action + 1}


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cat in the Box Bot Lab</title>
<style>
:root {
  color-scheme: light;
  --ink: #20242c;
  --muted: #667085;
  --paper: #f6f4ef;
  --panel: #fffdf8;
  --line: #d8d2c4;
  --red: #db4940;
  --blue: #336fbf;
  --yellow: #dfb43f;
  --green: #329263;
  --black: #171a20;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font: 15px/1.4 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--paper);
}
button, select, input { font: inherit; }
button {
  border: 1px solid var(--black);
  background: var(--black);
  color: white;
  min-height: 36px;
  padding: 0 12px;
  border-radius: 6px;
  cursor: pointer;
}
button.secondary { background: white; color: var(--black); border-color: var(--line); }
button:disabled { opacity: .45; cursor: not-allowed; }
.app { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
.top {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  position: sticky;
  top: 0;
  z-index: 5;
}
.brand { font-weight: 850; font-size: 20px; letter-spacing: 0; margin-right: auto; }
.control { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }
.control select, .control input {
  border: 1px solid var(--line);
  background: white;
  border-radius: 6px;
  min-height: 34px;
  padding: 0 8px;
}
.layout {
  display: grid;
  grid-template-columns: minmax(520px, 1fr) 360px;
  gap: 16px;
  padding: 16px;
}
.surface, .side {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.surface { padding: 14px; min-width: 0; }
.scorebar {
  display: grid;
  grid-template-columns: repeat(var(--players), minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 14px;
}
.seat {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: white;
}
.seat.active { outline: 3px solid #99d5ff; }
.seatName { display: flex; justify-content: space-between; gap: 8px; font-weight: 750; }
.seatStats { color: var(--muted); font-size: 13px; margin-top: 3px; }
.boardWrap { overflow-x: auto; padding-bottom: 4px; }
.board {
  display: grid;
  grid-template-columns: 82px repeat(var(--ranks), 58px);
  gap: 6px;
  min-width: max-content;
}
.cell, .rowHead, .rankHead {
  height: 48px;
  border: 1px solid var(--line);
  border-radius: 6px;
  display: grid;
  place-items: center;
  background: white;
  font-weight: 750;
}
.rankHead, .rowHead { color: var(--muted); background: #faf7ee; }
.rowHead { color: white; }
.rowHead.R { background: var(--red); }
.rowHead.B { background: var(--blue); }
.rowHead.Y { background: var(--yellow); color: #2b2410; }
.rowHead.G { background: var(--green); }
.cell.owner0 { box-shadow: inset 0 0 0 3px #111; }
.cell.owner1 { box-shadow: inset 0 0 0 3px #3b82f6; }
.cell.owner2 { box-shadow: inset 0 0 0 3px #c2410c; }
.cell.owner3 { box-shadow: inset 0 0 0 3px #7c3aed; }
.cell.owner4 { box-shadow: inset 0 0 0 3px #0f766e; }
.cell.blocked { background: repeating-linear-gradient(135deg, #ddd, #ddd 7px, #f8f8f8 7px, #f8f8f8 14px); color: #777; }
.hands { display: grid; grid-template-columns: repeat(var(--players), minmax(0, 1fr)); gap: 8px; margin-top: 14px; }
.hand { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: white; min-height: 96px; }
.cards { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.card {
  width: 34px;
  height: 46px;
  border: 1px solid #222;
  border-radius: 5px;
  display: grid;
  place-items: center;
  font-weight: 850;
  background: white;
}
.card.back { background: #222; color: white; border-color: #111; }
.side { padding: 14px; display: grid; align-content: start; gap: 14px; }
.status { display: grid; gap: 6px; }
.status h2 { margin: 0; font-size: 22px; }
.muted { color: var(--muted); }
.actions { display: grid; gap: 8px; }
.actionGrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.suggestion {
  border: 1px solid #1f7a4d;
  background: #e8f7ef;
  border-radius: 8px;
  padding: 10px;
}
.history {
  max-height: 330px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
}
.historyItem {
  display: grid;
  grid-template-columns: 62px 1fr;
  gap: 8px;
  padding: 8px 10px;
  border-bottom: 1px solid #eee8da;
  font-size: 13px;
}
.historyItem:last-child { border-bottom: 0; }
@media (max-width: 980px) {
  .layout { grid-template-columns: 1fr; }
  .scorebar, .hands { grid-template-columns: 1fr; }
  .top { flex-wrap: wrap; }
}
</style>
</head>
<body>
<div class="app">
  <header class="top">
    <div class="brand">Cat in the Box Bot Lab</div>
    <label class="control">Players <select id="players"><option>3</option><option>4</option><option>5</option><option>2</option></select></label>
    <label class="control">Bot <select id="bot"><option value="az_belief_policy">neural belief policy</option><option value="az_belief_search">neural belief search</option><option value="az_policy">neural policy</option><option value="heuristic_target2">target</option><option value="heuristic">balanced</option><option value="heuristic_adj2">adjacent</option><option value="random">random</option></select></label>
    <label class="control"><input id="reveal" type="checkbox"> Reveal</label>
    <button id="newGame">New</button>
  </header>
  <main class="layout">
    <section class="surface">
      <div id="scorebar" class="scorebar"></div>
      <div class="boardWrap"><div id="board" class="board"></div></div>
      <div id="hands" class="hands"></div>
    </section>
    <aside class="side">
      <div class="status">
        <h2 id="phase">Loading</h2>
        <div id="substatus" class="muted"></div>
      </div>
      <div id="suggestion" class="suggestion"></div>
      <div class="actions">
        <strong>Actions</strong>
        <div id="actions" class="actionGrid"></div>
      </div>
      <div>
        <strong>History</strong>
        <div id="history" class="history"></div>
      </div>
    </aside>
  </main>
</div>
<script>
let current = null;
const $ = (id) => document.getElementById(id);
async function api(path, options = {}) {
  const res = await fetch(path, options);
  const json = await res.json();
  if (!res.ok) throw new Error(json.error || res.statusText);
  return json;
}
async function refresh() {
  current = await api(`/api/state?reveal=${$("reveal").checked ? 1 : 0}`);
  render();
}
function render() {
  document.documentElement.style.setProperty("--ranks", current.board[0].length);
  document.documentElement.style.setProperty("--players", current.players);
  $("phase").textContent = current.terminal ? "Game Over" : current.phaseName;
  $("substatus").textContent = current.terminal
    ? `Returns ${current.returns.map(x => x.toFixed(0)).join(" / ")}`
    : `Trick ${current.trick + 1} · ${current.currentPlayer === current.humanSeat ? "Your turn" : "Bots moving"}`;
  $("scorebar").innerHTML = current.hands.map(h => {
    const active = current.currentPlayer === h.seat && !current.terminal ? " active" : "";
    const bid = current.predictions[h.seat] >= 0 ? current.predictions[h.seat] : "–";
    const boom = current.paradoxed[h.seat] ? " · paradox" : "";
    return `<div class="seat${active}">
      <div class="seatName"><span>${h.label}</span><span>${h.count}</span></div>
      <div class="seatStats">Tricks ${current.tricksWon[h.seat]} · Bid ${bid}${boom}</div>
    </div>`;
  }).join("");
  const rankHeads = ["<div></div>", ...current.board[0].map(c => `<div class="rankHead">${c.rank}</div>`)].join("");
  const rows = current.board.map(row => {
    const color = row[0].color;
    return `<div class="rowHead ${color}">${color}</div>` + row.map(c => {
      const owner = c.owner >= 0 ? ` owner${c.owner}` : "";
      const blocked = c.blocked ? " blocked" : "";
      const mark = c.blocked ? "×" : (c.owner >= 0 ? `P${c.owner}` : "");
      return `<div class="cell${owner}${blocked}">${mark}</div>`;
    }).join("");
  }).join("");
  $("board").innerHTML = rankHeads + rows;
  $("hands").innerHTML = current.hands.map(h => {
    const cards = h.cards.length
      ? h.cards.map(c => `<div class="card">${c.rank}</div>`).join("")
      : Array.from({length: h.count}, () => `<div class="card back">?</div>`).join("");
    return `<div class="hand"><strong>${h.label}</strong><div class="cards">${cards}</div></div>`;
  }).join("");
  $("suggestion").style.display = current.suggestion ? "block" : "none";
  $("suggestion").innerHTML = current.suggestion
    ? `<strong>Suggestion</strong><div>${current.suggestion.label}</div>`
    : "";
  $("actions").innerHTML = current.legalActions.map(a =>
    `<button data-action="${a.action}">${buttonLabel(a)}</button>`
  ).join("") || `<button disabled>${current.terminal ? "Done" : "Waiting"}</button>`;
  $("history").innerHTML = [...current.history].reverse().map(item =>
    `<div class="historyItem"><span>P${item.seat}</span><span>${item.label}</span></div>`
  ).join("");
  document.querySelectorAll("[data-action]").forEach(btn => {
    btn.addEventListener("click", () => play(Number(btn.dataset.action)));
  });
}
function buttonLabel(a) {
  if (a.kind === "prediction") return `Bid ${a.prediction}`;
  if (a.kind === "paradox") return "Paradox";
  if (a.kind === "play") return `${a.color}${a.rank}`;
  if (a.kind === "discard") return `Discard ${a.rank}`;
  return a.label;
}
async function play(action) {
  await api("/api/action", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({action})
  });
  await refresh();
}
$("newGame").addEventListener("click", async () => {
  await api("/api/new", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({players: Number($("players").value), bot: $("bot").value})
  });
  await refresh();
});
$("reveal").addEventListener("change", refresh);
refresh().catch(err => {
  $("phase").textContent = "Error";
  $("substatus").textContent = err.message;
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
  game_lock = threading.Lock()
  game = None

  @classmethod
  def configure(cls, game):
    cls.game = game

  def do_GET(self):
    parsed = urlparse(self.path)
    if parsed.path == "/":
      self._send_html(HTML)
      return
    if parsed.path == "/api/state":
      reveal = parse_qs(parsed.query).get("reveal", ["0"])[0] == "1"
      with self.game_lock:
        if self.game is None:
          self.configure(VisualizerGame())
        payload = self.game.payload(reveal=reveal)
      self._send_json(payload)
      return
    self.send_error(404)

  def do_POST(self):
    parsed = urlparse(self.path)
    data = self._read_json()
    try:
      if parsed.path == "/api/new":
        players = int(data.get("players", 3))
        if players not in (2, 3, 4, 5):
          raise ValueError("players must be 2, 3, 4, or 5")
        bot = data.get("bot", "heuristic_target2")
        with self.game_lock:
          if self.game is None:
            self.configure(VisualizerGame())
          self.game.reset(players=players, human_seat=0, bot_name=bot)
          payload = self.game.payload()
        self._send_json(payload)
        return
      if parsed.path == "/api/action":
        action = int(data["action"])
        with self.game_lock:
          if self.game is None:
            self.configure(VisualizerGame())
          self.game.apply_human_action(action)
          payload = self.game.payload()
        self._send_json(payload)
        return
      self.send_error(404)
    except Exception as exc:  # pylint: disable=broad-except
      self._send_json({"error": str(exc)}, status=400)

  def log_message(self, fmt, *args):
    return

  def _read_json(self):
    length = int(self.headers.get("Content-Length", "0"))
    if not length:
      return {}
    return json.loads(self.rfile.read(length).decode("utf-8"))

  def _send_html(self, body):
    data = body.encode("utf-8")
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def _send_json(self, payload, status=200):
    data = (json.dumps(payload) + "\n").encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8766)
  parser.add_argument("--checkpoint", default="az_runs/best_3p_policy_checkpoint.pt")
  parser.add_argument("--belief-samples", type=int, default=4)
  parser.add_argument("--belief-sims", type=int, default=8)
  args = parser.parse_args()
  checkpoint_path = args.checkpoint
  bot_name = "az_belief_policy"
  try:
    Handler.configure(VisualizerGame(
        players=3,
        bot_name=bot_name,
        checkpoint_path=checkpoint_path,
        belief_samples=args.belief_samples,
        belief_sims=args.belief_sims,
    ))
  except Exception as exc:
    print(f"Neural visualizer init failed ({exc}); falling back to heuristic.")
    Handler.configure(VisualizerGame(players=3, bot_name="heuristic_target2"))
  server = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"Serving Cat visualizer at http://{args.host}:{args.port}")
  server.serve_forever()


if __name__ == "__main__":
  main()
