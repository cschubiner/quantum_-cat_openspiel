#!/usr/bin/env python3
"""Contrastive hidden-world ranker for Quantum Cat belief sampling.

The ranker is deliberately actor-side: it scores candidate determinizations of
an information state. A passing gate can then improve belief search without
changing the champion policy network itself.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_ai import make_bot, shared_discard_action


MAX_PLAYERS = 5
MAX_RANKS = 9
MAX_COLORS = 4
WORLD_FEATURE_SIZE = (
    5  # phase one-hot
    + MAX_PLAYERS  # current-player relative one-hot
    + MAX_PLAYERS * MAX_RANKS  # relative hidden hand counts
    + MAX_PLAYERS * MAX_RANKS  # relative discarded ranks/counts
    + MAX_PLAYERS  # predictions
    + MAX_PLAYERS  # tricks won
    + MAX_PLAYERS * MAX_COLORS  # color tokens
    + MAX_COLORS * MAX_RANKS * 3  # board: free/self/other-or-blocked
    + MAX_PLAYERS  # cards already played this trick
    + 8  # scalar context
)
LIKELIHOOD_FEATURE_SIZE = 4


def _relative_player(player: int, target: int, num_players: int) -> int:
  return (int(target) - int(player)) % max(1, int(num_players))


def belief_world_feature_vector(state, player: int) -> np.ndarray:
  """Observation-safe candidate-world features for a sampled hidden state."""
  num_players = int(getattr(state, "_num_players", 0))
  num_ranks = int(getattr(state, "_num_card_types", 0))
  num_colors = int(getattr(state, "_num_colors", 0))
  num_tricks = max(1, int(getattr(state, "_num_tricks", 1)))
  current_player = int(getattr(state, "_current_player", player))
  phase = int(getattr(state, "_phase", -1))
  cards_initial = max(1, int(getattr(state, "_cards_per_player_initial", num_ranks)))

  features: list[float] = []
  for idx in range(5):
    features.append(1.0 if phase == idx else 0.0)
  current_rel = _relative_player(player, current_player, num_players)
  for idx in range(MAX_PLAYERS):
    features.append(1.0 if idx == current_rel else 0.0)

  hands = getattr(state, "_hands", [])
  for rel in range(MAX_PLAYERS):
    p = (player + rel) % max(1, num_players)
    hand = hands[p] if p < len(hands) else np.zeros(num_ranks, dtype=np.float32)
    for rank_idx in range(MAX_RANKS):
      value = float(hand[rank_idx]) if rank_idx < len(hand) else 0.0
      features.append(value / float(cards_initial))

  discarded = getattr(state, "_discarded_cards", [-1] * num_players)
  for rel in range(MAX_PLAYERS):
    p = (player + rel) % max(1, num_players)
    discard = int(discarded[p]) if p < len(discarded) else -1
    for rank_idx in range(MAX_RANKS):
      features.append(1.0 if discard == rank_idx + 1 else 0.0)

  predictions = getattr(state, "_predictions", [-1] * num_players)
  for rel in range(MAX_PLAYERS):
    p = (player + rel) % max(1, num_players)
    prediction = int(predictions[p]) if p < len(predictions) else -1
    features.append((prediction / float(num_tricks)) if prediction >= 0 else -1.0)

  tricks_won = getattr(state, "_tricks_won", np.zeros(num_players))
  for rel in range(MAX_PLAYERS):
    p = (player + rel) % max(1, num_players)
    tricks = float(tricks_won[p]) if p < len(tricks_won) else 0.0
    features.append(tricks / float(num_tricks))

  color_tokens = getattr(state, "_color_tokens", np.ones((num_players, num_colors)))
  for rel in range(MAX_PLAYERS):
    p = (player + rel) % max(1, num_players)
    for color_idx in range(MAX_COLORS):
      if p < color_tokens.shape[0] and color_idx < color_tokens.shape[1]:
        features.append(1.0 if color_tokens[p, color_idx] else 0.0)
      else:
        features.append(0.0)

  board = getattr(state, "_board_ownership", -1 * np.ones((num_colors, num_ranks)))
  for color_idx in range(MAX_COLORS):
    for rank_idx in range(MAX_RANKS):
      owner = -1
      if color_idx < board.shape[0] and rank_idx < board.shape[1]:
        owner = int(board[color_idx, rank_idx])
      features.extend([
          1.0 if owner == -1 else 0.0,
          1.0 if owner == player else 0.0,
          1.0 if owner not in (-1, player) else 0.0,
      ])

  trick_cards = getattr(state, "_cards_played_this_trick", [None] * num_players)
  for rel in range(MAX_PLAYERS):
    p = (player + rel) % max(1, num_players)
    features.append(1.0 if p < len(trick_cards) and trick_cards[p] is not None else 0.0)

  features.extend([
      float(getattr(state, "_trick_number", 0)) / float(num_tricks),
      float(sum(card is not None for card in trick_cards)) / max(1.0, float(num_players)),
      float(num_players) / float(MAX_PLAYERS),
      float(num_ranks) / float(MAX_RANKS),
      float(num_colors) / float(MAX_COLORS),
      1.0 if getattr(state, "_led_color", None) is None else 0.0,
      1.0 if getattr(state, "_trump_broken", False) else 0.0,
      float(getattr(state, "_match_round", 0)) / max(1.0, float(num_players)),
  ])

  if len(features) != WORLD_FEATURE_SIZE:
    raise ValueError(f"belief world feature size drifted to {len(features)}")
  return np.array(features, dtype=np.float32)


def _likelihood_feature_values(score, group_scores=None) -> list[float]:
  if score is None or not math.isfinite(float(score)):
    return [0.0, -1.0, 0.0, 0.0]
  clipped = max(-60.0, min(0.0, float(score)))
  centered = 0.0
  softmax_prob = 0.0
  if group_scores is not None:
    arr = np.array([
        -1e9 if value is None or not math.isfinite(float(value)) else float(value)
        for value in group_scores
    ], dtype=np.float64)
    finite = np.isfinite(arr) & (arr > -1e8)
    if np.any(finite):
      centered = max(-30.0, min(30.0, float(score) - float(np.mean(arr[finite]))))
      logits = arr - float(np.max(arr[finite]))
      probs = np.exp(np.clip(logits, -60.0, 0.0))
      total = float(probs.sum())
      if total > 0 and math.isfinite(total):
        # This caller appends one candidate at a time, so compute the matching
        # probability by value. Ties get the same feature, which is fine here.
        matches = np.where(np.isclose(arr, float(score)))[0]
        if len(matches):
          softmax_prob = float(probs[int(matches[0])] / total)
  return [
      1.0,
      clipped / 60.0,
      centered / 30.0,
      softmax_prob,
  ]


def ranker_world_feature_vector(
    state,
    player: int,
    expected_size: int = WORLD_FEATURE_SIZE,
    likelihood_score=None,
    group_likelihood_scores=None,
) -> np.ndarray:
  base = belief_world_feature_vector(state, player)
  if expected_size <= WORLD_FEATURE_SIZE:
    return base[:expected_size]
  extra = _likelihood_feature_values(likelihood_score, group_likelihood_scores)
  features = np.concatenate([base, np.array(extra, dtype=np.float32)])
  if expected_size <= features.shape[0]:
    return features[:expected_size].astype(np.float32)
  padded = np.zeros(expected_size, dtype=np.float32)
  padded[:features.shape[0]] = features
  return padded


class BeliefRankerV1(nn.Module):
  def __init__(self, obs_size: int, world_size: int, hidden: int = 256):
    super().__init__()
    self.obs_size = int(obs_size)
    self.world_size = int(world_size)
    self.net = nn.Sequential(
        nn.Linear(self.obs_size + self.world_size, hidden),
        nn.ReLU(),
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )

  def forward(self, obs: torch.Tensor, worlds: torch.Tensor) -> torch.Tensor:
    if obs.ndim == 2 and worlds.ndim == 3:
      obs = obs.unsqueeze(1).expand(-1, worlds.shape[1], -1)
    x = torch.cat([obs, worlds], dim=-1)
    return self.net(x).squeeze(-1)


def load_belief_ranker(path: str | Path, device: torch.device | str = "cpu"):
  checkpoint = torch.load(path, map_location=device)
  config = checkpoint["config"]
  model = BeliefRankerV1(
      int(config["obs_size"]),
      int(config["world_size"]),
      int(config.get("hidden", 256)),
  ).to(device)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()
  return model, config


_RANKER_CACHE: dict[tuple[str, str], tuple[BeliefRankerV1, dict]] = {}


def cached_belief_ranker(path: str | Path, device: torch.device | str):
  key = (str(path), str(device))
  if key not in _RANKER_CACHE:
    _RANKER_CACHE[key] = load_belief_ranker(path, device)
  return _RANKER_CACHE[key]


def ranker_resampled_belief_states(
    state,
    player: int,
    samples: int,
    candidate_count: int,
    ranker_path: str | Path,
    device: torch.device | str,
    temperature: float = 0.7,
    uniform_mix: float = 0.25,
    likelihood_score_fn=None,
    likelihood_scores_fn=None,
):
  """Samples hidden worlds by learned contrastive ranker score."""
  samples = max(1, int(samples))
  candidate_count = max(samples, int(candidate_count))
  candidates = []
  for _ in range(candidate_count):
    try:
      candidates.append(state.resample_from_infostate(player, np.random))
    except Exception:
      candidates.append(state.clone())
  if not candidates:
    return [state.clone()]

  model, config = cached_belief_ranker(ranker_path, device)
  expected_world_size = int(config.get("world_size", WORLD_FEATURE_SIZE))
  likelihood_scores = None
  if expected_world_size > WORLD_FEATURE_SIZE:
    if likelihood_scores_fn is not None:
      likelihood_scores = likelihood_scores_fn(candidates)
    elif likelihood_score_fn is not None:
      likelihood_scores = [likelihood_score_fn(candidate) for candidate in candidates]
  obs_np = np.array(state.observation_tensor(player), dtype=np.float32)
  worlds_np = np.stack([
      ranker_world_feature_vector(
          candidate,
          player,
          expected_world_size,
          None if likelihood_scores is None else likelihood_scores[idx],
          likelihood_scores,
      )
      for idx, candidate in enumerate(candidates)
  ]).astype(np.float32)
  with torch.no_grad():
    obs_t = torch.tensor(obs_np[None, :], dtype=torch.float32, device=device)
    worlds_t = torch.tensor(worlds_np[None, :, :], dtype=torch.float32, device=device)
    scores = model(obs_t, worlds_t).squeeze(0).cpu().numpy().astype(np.float64)

  temperature = max(1e-6, float(temperature))
  logits = (scores - float(np.max(scores))) / temperature
  weights = np.exp(logits)
  total = float(weights.sum())
  if total <= 0 or not math.isfinite(total):
    probs = np.ones(len(candidates), dtype=np.float64) / float(len(candidates))
  else:
    probs = weights / total
  mix = min(1.0, max(0.0, float(uniform_mix)))
  probs = (1.0 - mix) * probs + mix / float(len(candidates))
  probs = probs / float(probs.sum())
  chosen = np.random.choice(len(candidates), size=samples, replace=True, p=probs)
  return [candidates[int(idx)].clone() for idx in chosen]


def _phase_allowed(state, allowed: set[str]) -> bool:
  return not allowed or phase_name_for_state(state) in allowed


def phase_name_for_state(state) -> str:
  return {
      0: "chance",
      1: "discard",
      2: "prediction",
      3: "play",
      4: "terminal",
  }.get(int(getattr(state, "_phase", -1)), "unknown")


def _policy_action(state, player, bot_name: str, neural_actor=None) -> int:
  if neural_actor is not None:
    model, device, value_scale = neural_actor
    from quantum_cat_alphazero_torch import model_policy_value
    policy, _ = model_policy_value(
        model,
        state,
        player,
        int(state.num_distinct_actions()),
        value_scale,
        device,
    )
    legal = state.legal_actions(player)
    return int(max(legal, key=lambda action: float(policy[action])))
  return int(make_bot(bot_name, seed=0).step(state.clone(), player))


def _load_neural_actor(args, game):
  if not args.checkpoint:
    return None
  from quantum_cat_alphazero_torch import load_model_payload
  device = torch.device(args.device)
  model_args = SimpleNamespace(
      players=args.players,
      arch="mlp",
      width=256,
      depth=3,
      value_scale=args.value_scale,
  )
  model, _, saved_args = load_model_payload(args.checkpoint, game, model_args, device)
  model.eval()
  return model, device, float(saved_args.get("value_scale", args.value_scale))


def _candidate_likelihood_scorer(args, neural_actor):
  if not getattr(args, "include_policy_likelihood", False) or neural_actor is None:
    return None
  from quantum_cat_alphazero_torch import _score_observed_trick_likelihood
  model, device, value_scale = neural_actor
  score_args = SimpleNamespace(
      counterfactual_belief_ref_policy_mix=str(args.belief_ref_policy_mix),
      counterfactual_belief_ref_policy_mix_by_phase=str(
          args.belief_ref_policy_mix_by_phase
      ),
      counterfactual_belief_logprob_floor=float(args.belief_logprob_floor),
      value_scale=float(value_scale),
  )
  return lambda candidate: _score_observed_trick_likelihood(
      candidate, model, score_args, device
  )


def _candidate_likelihood_batch_scorer(args, neural_actor):
  if not getattr(args, "include_policy_likelihood", False) or neural_actor is None:
    return None
  from quantum_cat_alphazero_torch import _score_observed_trick_likelihood_batch
  model, device, value_scale = neural_actor
  score_args = SimpleNamespace(
      counterfactual_belief_ref_policy_mix=str(args.belief_ref_policy_mix),
      counterfactual_belief_ref_policy_mix_by_phase=str(
          args.belief_ref_policy_mix_by_phase
      ),
      counterfactual_belief_logprob_floor=float(args.belief_logprob_floor),
      value_scale=float(value_scale),
  )
  return lambda candidates: _score_observed_trick_likelihood_batch(
      candidates, model, score_args, device
  )


def collect_ranker_groups(args) -> dict[str, np.ndarray | dict]:
  random.seed(args.seed)
  np.random.seed(args.seed)
  game = pyspiel.load_game(
      "python_quantum_cat",
      {
          "players": args.players,
          "match_context": int(args.match_context),
          "start_player": args.start_player,
      },
  )
  neural_actor = _load_neural_actor(args, game)
  likelihood_scorer = _candidate_likelihood_scorer(args, neural_actor)
  likelihood_batch_scorer = _candidate_likelihood_batch_scorer(args, neural_actor)
  allowed_phases = {part.strip() for part in args.phases.split(",") if part.strip()}
  obs_rows = []
  world_rows = []
  phase_rows = []
  match_rows = []
  label_rows = []
  groups = 0
  started = time.perf_counter()

  for match_idx in range(args.matches):
    state = game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state.apply_action(int(np.random.choice(actions, p=probs)))
        continue
      player = int(state.current_player())
      legal = state.legal_actions(player)
      if legal and _phase_allowed(state, allowed_phases):
        if random.random() < float(args.state_prob):
          candidates = [state.clone()]
          for _ in range(args.negatives):
            candidates.append(state.resample_from_infostate(player, np.random))
          if likelihood_batch_scorer is not None:
            likelihood_scores = likelihood_batch_scorer(candidates)
          elif likelihood_scorer is not None:
            likelihood_scores = [likelihood_scorer(candidate) for candidate in candidates]
          else:
            likelihood_scores = None
          order = list(range(len(candidates)))
          random.shuffle(order)
          label_index = order.index(0)
          obs_rows.append(np.array(state.observation_tensor(player), dtype=np.float32))
          world_rows.append(np.stack([
              ranker_world_feature_vector(
                  candidates[idx],
                  player,
                  WORLD_FEATURE_SIZE + (
                      LIKELIHOOD_FEATURE_SIZE
                      if likelihood_scores is not None else 0
                  ),
                  None if likelihood_scores is None else likelihood_scores[idx],
                  likelihood_scores,
              )
              for idx in order
          ]))
          phase_rows.append(int(getattr(state, "_phase", -1)))
          match_rows.append(match_idx)
          label_rows.append(label_index)
          groups += 1
          progress_interval = int(getattr(args, "progress_interval", 0))
          if progress_interval > 0 and groups % progress_interval == 0:
            print(json.dumps({
                "groups": groups,
                "match_idx": match_idx,
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }), flush=True)
          if args.max_groups > 0 and groups >= args.max_groups:
            break
      discard_action = shared_discard_action(state, player, legal)
      if discard_action is not None:
        action = int(discard_action)
      elif args.bot == "random":
        action = int(random.choice(legal))
      else:
        action = _policy_action(state, player, args.bot, neural_actor)
      state.apply_action(action)
    if args.max_groups > 0 and groups >= args.max_groups:
      break

  if not obs_rows:
    raise RuntimeError("No ranker groups collected")
  return {
      "obs": np.stack(obs_rows).astype(np.float32),
      "worlds": np.stack(world_rows).astype(np.float32),
      "labels": np.array(label_rows, dtype=np.int64),
      "phases": np.array(phase_rows, dtype=np.int64),
      "match_ids": np.array(match_rows, dtype=np.int64),
      "meta": {
          "players": args.players,
          "negatives": args.negatives,
          "groups": len(obs_rows),
          "elapsed_sec": round(time.perf_counter() - started, 3),
          "world_feature_size": int(world_rows[0].shape[1]),
          "include_policy_likelihood": likelihood_scorer is not None,
      },
  }


def save_generated(args):
  data = collect_ranker_groups(args)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  meta = data.pop("meta")
  np.savez_compressed(out, **data, meta=json.dumps(meta))
  print(json.dumps({"out": str(out), **meta}, indent=2))


def _train_val_split(match_ids: np.ndarray, val_fraction: float, seed: int):
  rng = np.random.default_rng(seed)
  matches = np.array(sorted(set(int(x) for x in match_ids.tolist())))
  if len(matches) < 2:
    indices = np.arange(len(match_ids))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_fraction)))
    val_idx = set(int(x) for x in indices[:val_count])
    val_mask = np.array([idx in val_idx for idx in range(len(match_ids))], dtype=bool)
    return ~val_mask, val_mask
  rng.shuffle(matches)
  val_count = max(1, int(round(len(matches) * val_fraction))) if len(matches) > 1 else 1
  val_matches = set(int(x) for x in matches[:val_count])
  val_mask = np.array([int(x) in val_matches for x in match_ids], dtype=bool)
  train_mask = ~val_mask
  if not np.any(train_mask):
    train_mask[:] = True
    val_mask[:] = False
  return train_mask, val_mask


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
  with torch.no_grad():
    ranks = torch.argsort(logits, dim=1, descending=True)
    label_pos = (ranks == labels.unsqueeze(1)).nonzero()
    positions = label_pos[:, 1].float() + 1.0
    return {
        "top1": float((positions <= 1).float().mean().item()),
        "top4": float((positions <= 4).float().mean().item()),
        "mrr": float((1.0 / positions).mean().item()),
        "nll": float(F.cross_entropy(logits, labels).item()),
        "count": float(labels.numel()),
    }


def train_ranker(args):
  raw = np.load(args.data, allow_pickle=False)
  obs = raw["obs"].astype(np.float32)
  worlds = raw["worlds"].astype(np.float32)
  labels = raw["labels"].astype(np.int64)
  match_ids = raw["match_ids"].astype(np.int64)
  train_mask, val_mask = _train_val_split(match_ids, args.val_fraction, args.seed)
  device = torch.device(args.device)
  model = BeliefRankerV1(obs.shape[1], worlds.shape[2], hidden=args.hidden).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  train_idx = np.flatnonzero(train_mask)
  val_idx = np.flatnonzero(val_mask)
  rng = np.random.default_rng(args.seed)
  best = None
  best_report = None
  for epoch in range(1, args.epochs + 1):
    rng.shuffle(train_idx)
    losses = []
    model.train()
    for start in range(0, len(train_idx), args.batch_size):
      idx = train_idx[start:start + args.batch_size]
      obs_t = torch.tensor(obs[idx], dtype=torch.float32, device=device)
      worlds_t = torch.tensor(worlds[idx], dtype=torch.float32, device=device)
      labels_t = torch.tensor(labels[idx], dtype=torch.long, device=device)
      logits = model(obs_t, worlds_t)
      loss = F.cross_entropy(logits, labels_t)
      opt.zero_grad()
      loss.backward()
      opt.step()
      losses.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
      train_eval_idx = train_idx[:min(len(train_idx), max(args.batch_size, 512))]
      train_report = _metrics(
          model(
              torch.tensor(obs[train_eval_idx], dtype=torch.float32, device=device),
              torch.tensor(worlds[train_eval_idx], dtype=torch.float32, device=device),
          ),
          torch.tensor(labels[train_eval_idx], dtype=torch.long, device=device),
      )
      if len(val_idx):
        val_report = _metrics(
            model(
                torch.tensor(obs[val_idx], dtype=torch.float32, device=device),
                torch.tensor(worlds[val_idx], dtype=torch.float32, device=device),
            ),
            torch.tensor(labels[val_idx], dtype=torch.long, device=device),
        )
      else:
        val_report = dict(train_report)
    report = {
        "epoch": epoch,
        "loss": float(np.mean(losses)) if losses else None,
        "train": train_report,
        "validation": val_report,
    }
    print(json.dumps(report), flush=True)
    score = val_report["mrr"]
    if best is None or score > best:
      best = score
      best_report = report
      out = Path(args.out)
      out.parent.mkdir(parents=True, exist_ok=True)
      torch.save(
          {
              "model_state_dict": model.state_dict(),
              "config": {
                  "obs_size": int(obs.shape[1]),
                  "world_size": int(worlds.shape[2]),
                  "hidden": int(args.hidden),
              },
              "best_report": best_report,
          },
          out,
      )
  report_path = Path(args.out).with_suffix(".validation.json")
  report_path.write_text(json.dumps(best_report, indent=2) + "\n")
  print(json.dumps({"checkpoint": args.out, "validation": str(report_path), "best": best_report}, indent=2))


def main():
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest="cmd", required=True)
  gen = sub.add_parser("generate")
  gen.add_argument("--players", type=int, default=3)
  gen.add_argument("--match-context", action="store_true")
  gen.add_argument("--start-player", type=int, default=0)
  gen.add_argument("--matches", type=int, default=100)
  gen.add_argument("--max-groups", type=int, default=0)
  gen.add_argument("--negatives", type=int, default=31)
  gen.add_argument("--state-prob", type=float, default=1.0)
  gen.add_argument("--phases", default="prediction,play")
  gen.add_argument("--bot", default="heuristic")
  gen.add_argument("--checkpoint", default="")
  gen.add_argument("--device", default="cpu")
  gen.add_argument("--value-scale", type=float, default=20.0)
  gen.add_argument("--include-policy-likelihood", action="store_true")
  gen.add_argument(
      "--belief-ref-policy-mix",
      default="model:1.0",
      help="Reference policy mix used for appended public-history likelihood features.",
  )
  gen.add_argument("--belief-ref-policy-mix-by-phase", default="")
  gen.add_argument("--belief-logprob-floor", type=float, default=1e-6)
  gen.add_argument("--seed", type=int, default=20260604)
  gen.add_argument("--progress-interval", type=int, default=0)
  gen.add_argument("--out", required=True)
  gen.set_defaults(func=save_generated)

  train = sub.add_parser("train")
  train.add_argument("--data", required=True)
  train.add_argument("--out", required=True)
  train.add_argument("--hidden", type=int, default=256)
  train.add_argument("--epochs", type=int, default=10)
  train.add_argument("--batch-size", type=int, default=256)
  train.add_argument("--lr", type=float, default=3e-4)
  train.add_argument("--weight-decay", type=float, default=1e-3)
  train.add_argument("--val-fraction", type=float, default=0.2)
  train.add_argument("--seed", type=int, default=20260604)
  train.add_argument("--device", default="cpu")
  train.set_defaults(func=train_ranker)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
