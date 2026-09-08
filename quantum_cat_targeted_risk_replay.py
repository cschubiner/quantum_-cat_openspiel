"""Mine targeted same-policy paradox-risk labels for action reranking."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import

from quantum_cat_ai import make_bot
from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_RANK_NORM_INDEX,
    APPENDED_ACTION_FEATURE_INDEX,
    action_feature_matrix,
    model_policy_value,
    save_replay,
)
from quantum_cat_full_match_elo import load_neural
from quantum_cat_survival_grammar_replay import safe_feature, structural_scores


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=50)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--seed", type=int, default=20267300)
  parser.add_argument("--initial-start", type=int, default=0)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
  parser.add_argument("--no-match-context", dest="match_context", action="store_false")
  parser.set_defaults(match_context=True)
  parser.add_argument(
      "--phases",
      default="prediction,play",
      help="Comma-separated phases to label: discard,prediction,play.",
  )
  parser.add_argument("--min-trick-number", type=int, default=2)
  parser.add_argument("--leader-min-trick-number", type=int, default=3)
  parser.add_argument("--max-rows", type=int, default=0)
  parser.add_argument("--max-actions-per-state", type=int, default=8)
  parser.add_argument("--top-policy-actions", type=int, default=3)
  parser.add_argument(
      "--include-bot-actions",
      default="heuristic_safe14",
      help="Comma-separated built-in bot actions to always include if legal.",
  )
  parser.add_argument(
      "--continuation",
      choices=("policy", "builtin"),
      default="policy",
      help="Policy used after each candidate action during risk rollouts.",
  )
  parser.add_argument("--continuation-bot", default="heuristic_safe14")
  parser.add_argument("--rollouts-per-action", type=int, default=1)
  parser.add_argument("--continuation-temperature", type=float, default=0.0)
  parser.add_argument(
      "--source-temperature",
      type=float,
      default=0.0,
      help=(
          "Temperature used when advancing the source self-play trajectory. "
          "0 keeps the checkpoint's greedy policy; >0 mines a wider state "
          "distribution around the policy."
      ),
  )
  parser.add_argument(
      "--policy-target",
      choices=("original", "safest", "soft_survival"),
      default="original",
      help=(
          "Policy label to write for each kept row. original preserves the "
          "source checkpoint policy; safest writes a one-hot label on the "
          "lowest-risk labeled action; soft_survival writes a softmax over "
          "1-risk for labeled actions."
      ),
  )
  parser.add_argument(
      "--policy-target-temperature",
      type=float,
      default=0.15,
      help="Temperature for --policy-target=soft_survival.",
  )
  parser.add_argument(
      "--policy-mask",
      choices=("legal", "candidates"),
      default="legal",
      help=(
          "Action mask used by policy distillation. candidates trains only "
          "among labeled rollout candidates so unlabeled legal actions are "
          "not treated as negative examples."
      ),
  )
  parser.add_argument(
      "--risk-scope",
      choices=("any", "acting"),
      default="any",
      help="Risk target is any-player paradox or acting-player paradox.",
  )
  parser.add_argument("--min-risk-spread", type=float, default=0.50)
  parser.add_argument(
      "--row-filter",
      choices=("targeted", "all_allowed"),
      default="targeted",
  )
  parser.add_argument("--progress-interval", type=int, default=10)

  # Reuse structural_scores() knobs for candidate selection only.
  parser.add_argument("--prior-weight", type=float, default=0.02)
  parser.add_argument("--action-value-weight", type=float, default=0.0)
  parser.add_argument("--follow-led-bonus", type=float, default=1.6)
  parser.add_argument("--token-loss-penalty", type=float, default=5.5)
  parser.add_argument("--min-lane-surplus-weight", type=float, default=1.25)
  parser.add_argument("--lane-damage-penalty", type=float, default=1.10)
  parser.add_argument("--lane-pressure-penalty", type=float, default=0.45)
  parser.add_argument("--own-min-colors-weight", type=float, default=0.85)
  parser.add_argument("--own-zero-exit-penalty", type=float, default=2.6)
  parser.add_argument("--own-one-exit-penalty", type=float, default=1.5)
  parser.add_argument("--own-two-less-exit-penalty", type=float, default=0.65)
  parser.add_argument("--future-no-exit-penalty", type=float, default=2.0)
  parser.add_argument("--forced-pressure-penalty", type=float, default=0.90)
  parser.add_argument("--paradox-penalty", type=float, default=8.0)
  parser.add_argument("--discard-relief-weight", type=float, default=1.0)
  parser.add_argument("--discard-no-exit-penalty", type=float, default=1.5)
  parser.add_argument("--prediction-abs-gap-penalty", type=float, default=0.35)
  parser.add_argument("--prediction-over-penalty", type=float, default=0.15)
  return parser.parse_args()


def split_csv(value: str) -> set[str]:
  return {part.strip().lower() for part in str(value).split(",") if part.strip()}


def game_for(players: int, start_player: int, match_context: bool):
  return pyspiel.load_game(
      "python_quantum_cat",
      {
          "players": int(players),
          "start_player": int(start_player),
          "match_context": int(bool(match_context)),
      },
  )


def phase_name(state) -> str:
  return {
      0: "chance",
      1: "discard",
      2: "prediction",
      3: "play",
      4: "terminal",
  }.get(int(getattr(state, "_phase", -1)), "unknown")


def legal_mask(num_actions: int, legal: list[int]) -> np.ndarray:
  mask = np.zeros(num_actions, dtype=np.float32)
  for action in legal:
    mask[int(action)] = 1.0
  return mask


def sample_from_policy(
    policy: np.ndarray,
    legal: list[int],
    temperature: float,
    rng: np.random.Generator,
) -> int:
  if len(legal) == 1:
    return int(legal[0])
  if temperature <= 1e-9:
    return int(max(legal, key=lambda action: float(policy[int(action)])))
  probs = np.array([max(float(policy[int(action)]), 1e-12) for action in legal])
  logits = np.log(probs) / max(1e-6, float(temperature))
  logits -= float(np.max(logits))
  weights = np.exp(logits)
  total = float(np.sum(weights))
  if not math.isfinite(total) or total <= 0.0:
    weights = np.ones_like(weights)
    total = float(np.sum(weights))
  weights /= total
  return int(rng.choice(np.array(legal, dtype=np.int64), p=weights))


def policy_action(model, state, player: int, value_scale: float, device, rng, temp):
  policy, _ = model_policy_value(
      model, state, player, state.num_distinct_actions(), value_scale, device
  )
  return sample_from_policy(policy, list(state.legal_actions(player)), temp, rng)


def continue_to_terminal(
    state,
    model,
    value_scale: float,
    device,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> object:
  builtin = make_bot(str(args.continuation_bot), seed=int(rng.integers(2**31 - 1)))
  while not state.is_terminal():
    if state.is_chance_node():
      actions, probs = zip(*state.chance_outcomes())
      state.apply_action(int(rng.choice(np.array(actions, dtype=np.int64), p=probs)))
      continue
    player = int(state.current_player())
    legal = list(map(int, state.legal_actions(player)))
    if str(args.continuation) == "builtin":
      action = int(builtin.step(state.clone(), player))
      if action not in legal:
        action = int(legal[0])
    else:
      action = policy_action(
          model,
          state,
          player,
          value_scale,
          device,
          rng,
          float(args.continuation_temperature),
      )
    state.apply_action(action)
  return state


def risk_after_action(
    state,
    action: int,
    acting_player: int,
    model,
    value_scale: float,
    device,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> float:
  risks = []
  rollouts = max(1, int(args.rollouts_per_action))
  for _ in range(rollouts):
    shadow = state.clone()
    shadow.apply_action(int(action))
    terminal = continue_to_terminal(
        shadow, model, value_scale, device, args, rng
    )
    paradoxed = [bool(value) for value in getattr(terminal, "_has_paradoxed", [])]
    if str(args.risk_scope) == "acting":
      risks.append(1.0 if paradoxed[int(acting_player)] else 0.0)
    else:
      risks.append(1.0 if any(paradoxed) else 0.0)
  return float(np.mean(np.array(risks, dtype=np.float32)))


def indexed_feature(features: np.ndarray, name: str) -> np.ndarray:
  if name not in APPENDED_ACTION_FEATURE_INDEX:
    return np.zeros(features.shape[0], dtype=np.float32)
  return features[:, APPENDED_ACTION_FEATURE_INDEX[name]]


def add_extreme(
    selected: list[int],
    legal: list[int],
    values: np.ndarray,
    *,
    high: bool,
):
  if not legal:
    return
  action = max(legal, key=lambda a: float(values[int(a)])) if high else min(
      legal, key=lambda a: float(values[int(a)])
  )
  if int(action) not in selected:
    selected.append(int(action))


def candidate_actions(
    state,
    player: int,
    legal: list[int],
    policy: np.ndarray,
    features: np.ndarray,
    args: argparse.Namespace,
    builtin_bots: list,
) -> list[int]:
  if int(args.max_actions_per_state) <= 0 or len(legal) <= int(args.max_actions_per_state):
    return [int(action) for action in legal]
  selected: list[int] = []

  def add(action: int):
    action = int(action)
    if action in legal and action not in selected:
      selected.append(action)

  add(int(max(legal, key=lambda action: float(policy[int(action)]))))
  for action in sorted(legal, key=lambda a: float(policy[int(a)]), reverse=True)[
      :max(0, int(args.top_policy_actions))
  ]:
    add(int(action))
  for bot in builtin_bots:
    try:
      add(int(bot.step(state.clone(), player)))
    except Exception:
      pass

  legal_arr = [int(action) for action in legal]
  structural = structural_scores(features, policy, None, None, args)
  add_extreme(selected, legal_arr, structural, high=True)
  add_extreme(selected, legal_arr, structural, high=False)
  add_extreme(selected, legal_arr, features[:, ACTION_FEATURE_RANK_NORM_INDEX], high=False)
  add_extreme(selected, legal_arr, features[:, ACTION_FEATURE_RANK_NORM_INDEX], high=True)
  add_extreme(selected, legal_arr, features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX], high=True)
  add_extreme(
      selected,
      legal_arr,
      indexed_feature(features, "token_loss_newly_loses_led"),
      high=False,
  )
  add_extreme(
      selected,
      legal_arr,
      indexed_feature(features, "own_future_min_colors_after"),
      high=True,
  )
  add_extreme(
      selected,
      legal_arr,
      indexed_feature(features, "own_future_zero_exit_frac_after"),
      high=False,
  )
  add_extreme(
      selected,
      legal_arr,
      indexed_feature(features, "future_no_exit_after"),
      high=False,
  )
  add_extreme(selected, legal_arr, features[:, ACTION_FEATURE_IS_RED_INDEX], high=False)
  add_extreme(selected, legal_arr, features[:, ACTION_FEATURE_IS_RED_INDEX], high=True)

  limit = max(1, int(args.max_actions_per_state))
  return selected[:limit]


def is_targeted_state(
    state,
    player: int,
    legal: list[int],
    features: np.ndarray,
    args: argparse.Namespace,
) -> tuple[bool, str]:
  phase = phase_name(state)
  if phase not in split_csv(args.phases):
    return False, "phase"
  if len(legal) < 2:
    return False, "forced"
  if str(args.row_filter) == "all_allowed":
    return True, "all_allowed"
  if phase == "prediction":
    return True, "prediction"
  if phase != "play":
    return False, "not_targeted_phase"

  trick = int(getattr(state, "_trick_number", -1))
  led_color = getattr(state, "_led_color", None)
  predictions = list(getattr(state, "_predictions", []))
  tricks_won = list(getattr(state, "_tricks_won", []))
  prediction = int(predictions[player]) if 0 <= player < len(predictions) else -1
  won = int(tricks_won[player]) if 0 <= player < len(tricks_won) else 0
  satisfied = prediction >= 0 and won >= prediction
  lane_contrast = (
      bool(np.any(features[legal, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5))
      and bool(np.any(safe_feature(features, "token_loss_newly_loses_led")[legal] > 0.5))
  )
  bottleneck = (
      float(np.max(indexed_feature(features, "own_future_zero_exit_frac_after")[legal], initial=0.0)) > 0.01
      or float(np.max(indexed_feature(features, "own_future_one_exit_frac_after")[legal], initial=0.0)) > 0.10
  )
  red_choice = bool(np.any(features[legal, ACTION_FEATURE_IS_RED_INDEX] > 0.5))
  if led_color is None and trick >= int(args.leader_min_trick_number):
    if satisfied or bottleneck or red_choice:
      return True, "leader_control"
  if trick >= int(args.min_trick_number) and (lane_contrast or satisfied or bottleneck):
    if lane_contrast:
      return True, "lane_contrast"
    if satisfied:
      return True, "prediction_satisfied"
    return True, "bottleneck"
  return False, "not_targeted"


def replay_row(
    state,
    player: int,
    policy: np.ndarray,
    legal: list[int],
    features: np.ndarray,
    risks: dict[int, float],
    args: argparse.Namespace,
):
  num_actions = state.num_distinct_actions()
  risk_targets = np.zeros(num_actions, dtype=np.float32)
  risk_mask = np.zeros(num_actions, dtype=np.float32)
  value_targets = np.zeros(num_actions, dtype=np.float32)
  value_mask = np.zeros(num_actions, dtype=np.float32)
  for action, risk in risks.items():
    risk_targets[int(action)] = float(risk)
    risk_mask[int(action)] = 1.0
    value_targets[int(action)] = 1.0 - 2.0 * float(risk)
    value_mask[int(action)] = 1.0
  values = np.zeros(int(getattr(state, "_num_players", 3)), dtype=np.float32)
  paradoxes = np.zeros_like(values, dtype=np.float32)
  baseline = int(max(legal, key=lambda action: float(policy[int(action)])))
  policy_target = np.asarray(policy, dtype=np.float32).copy()
  mask_actions = legal
  if risks and str(args.policy_target) != "original":
    candidate_actions = sorted(int(action) for action in risks)
    mask_actions = candidate_actions if str(args.policy_mask) == "candidates" else legal
    policy_target = np.zeros(num_actions, dtype=np.float32)
    if str(args.policy_target) == "safest":
      best_risk = min(float(risk) for risk in risks.values())
      best_actions = [
          int(action)
          for action, risk in risks.items()
          if abs(float(risk) - best_risk) <= 1e-6
      ]
      best = int(max(best_actions, key=lambda action: float(policy[int(action)])))
      policy_target[best] = 1.0
    else:
      temperature = max(1e-6, float(args.policy_target_temperature))
      logits = np.array(
          [(1.0 - float(risks[action])) / temperature for action in candidate_actions],
          dtype=np.float32,
      )
      logits -= float(np.max(logits))
      weights = np.exp(logits)
      total = float(np.sum(weights))
      if not math.isfinite(total) or total <= 0.0:
        weights = np.ones_like(weights, dtype=np.float32)
        total = float(np.sum(weights))
      weights /= total
      for action, weight in zip(candidate_actions, weights):
        policy_target[int(action)] = float(weight)
  return (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask(num_actions, mask_actions),
      policy_target,
      values,
      paradoxes,
      baseline,
      int(player),
      risk_targets,
      risk_mask,
      value_targets,
      value_mask,
      features,
      None,
  )


def collect(args: argparse.Namespace):
  started = time.perf_counter()
  rng = np.random.default_rng(int(args.seed))
  torch.manual_seed(int(args.seed))
  if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("--device=mps requested but unavailable")
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model, model_args = load_neural(args.checkpoint, int(args.players), device)
  if model is None or model_args is None:
    raise ValueError("--checkpoint did not load a model")
  builtin_bots = [
      make_bot(name, seed=int(args.seed) + idx * 31)
      for idx, name in enumerate(split_csv(args.include_bot_actions))
  ]
  rows = []
  stats = Counter()
  phase_counts = Counter()
  reason_counts = Counter()
  risk_spreads = []
  baseline_risks = []
  best_risks = []
  labeled_actions = 0

  for match_idx in range(int(args.matches)):
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      start_player = (
          int(args.initial_start) + int(match_idx) + int(round_idx)
      ) % int(args.players)
      game = game_for(int(args.players), start_player, bool(args.match_context))
      state = game.new_initial_state()
      if bool(args.match_context):
        state.set_match_context(totals, round_idx)
      while not state.is_terminal():
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          state.apply_action(
              int(rng.choice(np.array(actions, dtype=np.int64), p=probs))
          )
          continue
        player = int(state.current_player())
        legal = list(map(int, state.legal_actions(player)))
        policy, _ = model_policy_value(
            model,
            state,
            player,
            state.num_distinct_actions(),
            model_args.value_scale,
            device,
        )
        action = sample_from_policy(
            policy,
            legal,
            float(args.source_temperature),
            rng,
        )
        features = action_feature_matrix(state, player, state.num_distinct_actions())
        keep, reason = is_targeted_state(state, player, legal, features, args)
        reason_counts[reason] += 1
        if keep and (int(args.max_rows) <= 0 or len(rows) < int(args.max_rows)):
          candidates = candidate_actions(
              state, player, legal, policy, features, args, builtin_bots
          )
          risks = {
              int(candidate): risk_after_action(
                  state,
                  int(candidate),
                  player,
                  model,
                  model_args.value_scale,
                  device,
                  args,
                  rng,
              )
              for candidate in candidates
          }
          spread = float(max(risks.values()) - min(risks.values())) if risks else 0.0
          if spread >= float(args.min_risk_spread):
            rows.append(
                replay_row(state, player, policy, legal, features, risks, args)
            )
            phase_counts[phase_name(state)] += 1
            risk_spreads.append(spread)
            baseline_risks.append(float(risks.get(action, 0.0)))
            best_risks.append(float(min(risks.values())))
            labeled_actions += len(risks)
            stats["rows"] += 1
            stats["baseline_risky_rows"] += int(float(risks.get(action, 0.0)) > 0.5)
            stats["improvable_rows"] += int(
                float(risks.get(action, 0.0)) - float(min(risks.values())) >=
                float(args.min_risk_spread)
            )
          else:
            stats["low_risk_spread_rows"] += 1
        state.apply_action(action)
      if bool(args.match_context):
        totals = np.array(state.returns(), dtype=np.float32)
      if (
          int(args.max_rows) > 0
          and len(rows) >= int(args.max_rows)
      ):
        break
    if (
        int(args.progress_interval) > 0
        and (match_idx + 1) % int(args.progress_interval) == 0
    ):
      print(json.dumps({
          "completed_matches": match_idx + 1,
          "rows": len(rows),
          "labeled_actions": int(labeled_actions),
          "elapsed_sec": round(time.perf_counter() - started, 3),
      }), flush=True)
    if int(args.max_rows) > 0 and len(rows) >= int(args.max_rows):
      break

  summary = {
      "source": "quantum_cat_targeted_risk_replay",
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches_requested": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "phases": str(args.phases),
      "row_filter": str(args.row_filter),
      "continuation": str(args.continuation),
      "continuation_bot": str(args.continuation_bot),
      "rollouts_per_action": int(args.rollouts_per_action),
      "source_temperature": float(args.source_temperature),
      "continuation_temperature": float(args.continuation_temperature),
      "risk_scope": str(args.risk_scope),
      "policy_target": str(args.policy_target),
      "policy_target_temperature": float(args.policy_target_temperature),
      "policy_mask": str(args.policy_mask),
      "rows": int(len(rows)),
      "labeled_actions": int(labeled_actions),
      "phase_counts": dict(sorted(phase_counts.items())),
      "reason_counts": dict(sorted(reason_counts.items())),
      "stats": dict(sorted(stats.items())),
      "mean_risk_spread": float(np.mean(risk_spreads)) if risk_spreads else 0.0,
      "mean_baseline_risk": float(np.mean(baseline_risks)) if baseline_risks else 0.0,
      "mean_best_risk": float(np.mean(best_risks)) if best_risks else 0.0,
      "elapsed_sec": round(time.perf_counter() - started, 3),
  }
  return rows, summary


def main() -> None:
  args = parse_args()
  rows, summary = collect(args)
  if not rows:
    raise SystemExit(json.dumps(summary, indent=2, sort_keys=True))
  save_replay(rows, args.out, metadata=summary)
  text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
  summary_path = (
      Path(args.summary_out)
      if args.summary_out else Path(args.out).with_suffix(Path(args.out).suffix + ".summary.json")
  )
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(text)
  print(text, end="")


if __name__ == "__main__":
  main()
