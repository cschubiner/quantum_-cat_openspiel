#!/usr/bin/env python3
"""Generate best-of-rollouts replay rows for Quantum Cat.

This is the first expert-iteration/ReST-style data path for the repo:

  live policy state -> evaluate several legal first actions with repeated
  continuations -> rank the branches -> train the first action from better
  branches upward.

The labels are intentionally trajectory-level. Survival dominates score, so a
high-scoring move that regularly collapses into paradox should not become the
teacher merely because it looks tactically attractive in the current trick.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyspiel
import torch

from quantum_cat_alphazero_torch import action_feature_matrix, save_replay
from quantum_cat_failure_miner import (
    action_string,
    parse_phase_filter,
    phase_name,
    play_round_with_history,
    resolve_actor_bot_args,
    rollout_from_action,
    should_evaluate_decision,
    terminal_round_result,
)
from quantum_cat_full_match_elo import load_neural
from quantum_cat_public_history import public_history_metadata


COLOR_NAMES = ("R", "B", "Y", "G")
PARADOX_ACTION = 999
_NUMPY_SEED_MODULUS = 2**32 - 1


def _numpy_seed(value: int) -> int:
  """Map derived seed blocks into NumPy RandomState's accepted range."""
  return int(int(value) % _NUMPY_SEED_MODULUS)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--matches", type=int, default=4)
  parser.add_argument("--seed", type=int, default=20263710)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--bot-mode", default="policy")
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  parser.add_argument("--out", required=True)
  parser.add_argument("--metadata-out", default="")
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--max-decisions", type=int, default=80)
  parser.add_argument("--max-accepted-rows", type=int, default=0)
  parser.add_argument("--max-decisions-per-round", type=int, default=3)
  parser.add_argument("--min-legal-count", type=int, default=2)
  parser.add_argument("--max-legal-count", type=int, default=0)
  parser.add_argument("--min-trick-number", type=int, default=0)
  parser.add_argument("--max-trick-number", type=int, default=99)
  parser.add_argument("--led-contrast-only", action="store_true")
  parser.add_argument("--require-red-switch-legal", action="store_true")
  parser.add_argument("--rollouts-per-action", type=int, default=5)
  parser.add_argument(
      "--rollout-state-source",
      choices=("infostate", "actual"),
      default="infostate",
      help=(
          "Hidden-state source for candidate continuations. infostate is the "
          "deployment-safe default and evaluates every action on the same "
          "resampled worlds. actual is a full-information oracle audit only."
      ),
  )
  parser.add_argument(
      "--belief-samples",
      type=int,
      default=4,
      help=(
          "Number of actor-information-state worlds per decision when "
          "--rollout-state-source=infostate. --rollouts-per-action is run in "
          "each sampled world."
      ),
  )
  parser.add_argument(
      "--audit-belief-samples",
      type=int,
      default=0,
      help=(
          "Fresh actor-information-state worlds used only after selecting the "
          "teacher-best action. The selected action must independently beat "
          "the source action on these held-out worlds. 0 disables the audit."
      ),
  )
  parser.add_argument(
      "--audit-rollouts-per-action",
      type=int,
      default=0,
      help=(
          "Continuation rollouts per held-out belief world and action. 0 uses "
          "--rollouts-per-action."
      ),
  )
  parser.add_argument("--max-candidate-actions", type=int, default=8)
  parser.add_argument("--policy-top-actions", type=int, default=5)
  parser.add_argument(
      "--include-all-legal",
      action="store_true",
      help="Evaluate every legal action instead of a capped policy/tactical set.",
  )
  parser.add_argument(
      "--require-policy-target-changed",
      action="store_true",
      help="Keep only rows where the rollout-ranked best action differs from policy.",
  )
  parser.add_argument("--require-survival-contrast", action="store_true")
  parser.add_argument("--min-any-paradox-rate-spread", type=float, default=0.25)
  parser.add_argument(
      "--require-forced-paradox-contrast",
      action="store_true",
      help=(
          "Keep only decisions where evaluated actions have materially "
          "different forced-paradox rates. Combine with --min-trick-number 5 "
          "and --max-trick-number 7 for explicit late forced-exit coverage."
      ),
  )
  parser.add_argument("--min-forced-paradox-rate-spread", type=float, default=0.25)
  parser.add_argument(
      "--source-group-id",
      default="",
      help=(
          "Optional provenance id written once per replay row as row_group_ids. "
          "Use a distinct id per independently seeded source shard so held-out "
          "validation can exclude whole sources."
      ),
  )
  parser.add_argument("--min-best-vs-base-score-delta", type=float, default=-1e9)
  parser.add_argument(
      "--min-best-vs-base-any-paradox-improvement",
      type=float,
      default=0.0,
      help=(
          "Require the selected teacher action to reduce paired any-player "
          "paradox rate versus the source policy action by at least this "
          "amount. Positive values protect against winner's-curse overrides."
      ),
  )
  parser.add_argument(
      "--min-audit-best-vs-base-any-paradox-improvement",
      type=float,
      default=0.0,
      help=(
          "Required paired any-player paradox-rate improvement on fresh "
          "--audit-belief-samples after the teacher action was selected."
      ),
  )
  parser.add_argument(
      "--max-audit-paired-binomial-p",
      type=float,
      default=1.0,
      help=(
          "Optional one-sided exact-binomial evidence gate over paired "
          "base/proposal information-state worlds in every fresh audit pass. "
          "Multiple continuations within a world are nested replicates and "
          "count as one inference unit. "
          "1.0 disables it; 0.10 combined with three disjoint passes gives a "
          "fail-closed repeatability screen without relying on rate rounding."
      ),
  )
  parser.add_argument(
      "--audit-rejected-base-out",
      default="",
      help=(
          "Optional separate replay for selection-positive proposals that "
          "fail the fresh-world audit. These rows target the source-policy "
          "action and provide explicit abstention supervision."
      ),
  )
  parser.add_argument(
      "--audit-rejected-base-metadata-out",
      default="",
      help="Optional sidecar path for --audit-rejected-base-out.",
  )
  parser.add_argument(
      "--max-audit-rejected-base-rows",
      type=int,
      default=0,
      help="Optional cap for separately emitted audit-negative rows.",
  )
  parser.add_argument("--target-temperature", type=float, default=0.20)
  parser.add_argument("--target-top-score-gap", type=float, default=2.0)
  parser.add_argument("--survival-weight", type=float, default=10.0)
  parser.add_argument("--actor-survival-weight", type=float, default=2.0)
  parser.add_argument("--forced-paradox-penalty", type=float, default=1.0)
  parser.add_argument("--actor-score-weight", type=float, default=0.20)
  parser.add_argument("--total-score-weight", type=float, default=0.02)
  parser.add_argument(
      "--action-value-safe-target",
      type=float,
      default=0.90,
      help="Bounded auxiliary action-value target for no-paradox branches.",
  )
  parser.add_argument(
      "--action-value-paradox-target",
      type=float,
      default=-0.55,
      help="Bounded auxiliary action-value target for any-paradox branches.",
  )
  parser.add_argument("--action-value-actor-penalty", type=float, default=0.10)
  parser.add_argument("--action-value-forced-penalty", type=float, default=0.25)
  parser.add_argument("--action-value-score-weight", type=float, default=0.05)
  parser.add_argument("--action-value-min", type=float, default=-0.95)
  parser.add_argument("--action-value-max", type=float, default=0.95)
  parser.add_argument("--progress-interval", type=int, default=0)
  parser.add_argument(
      "--write-empty",
      action="store_true",
      help="Write metadata/summary even when no replay rows are accepted.",
  )
  return parser.parse_args()


def _device_from_arg(spec: str) -> torch.device:
  if spec == "auto":
    if torch.backends.mps.is_available():
      return torch.device("mps")
    return torch.device("cpu")
  return torch.device(spec)


def _action_color(state, action: int):
  action = int(action)
  if action == PARADOX_ACTION:
    return None
  num_ranks = int(getattr(state, "_num_card_types", 0) or 0)
  if num_ranks <= 0:
    return None
  color_idx = action // num_ranks
  if 0 <= color_idx < len(COLOR_NAMES):
    return COLOR_NAMES[color_idx]
  return None


def _decision_allowed(decision: dict, allowed_phases: set[str], args) -> bool:
  if not should_evaluate_decision(decision, allowed_phases):
    return False
  legal = [int(action) for action in decision.get("legal", [])]
  if len(legal) < int(args.min_legal_count):
    return False
  max_legal = int(getattr(args, "max_legal_count", 0) or 0)
  if max_legal > 0 and len(legal) > max_legal:
    return False
  trick = int(decision.get("trick_number", -1))
  if trick < int(args.min_trick_number) or trick > int(args.max_trick_number):
    return False
  if not bool(args.led_contrast_only):
    return True
  state = decision["state"]
  led_color = getattr(state, "_led_color", None)
  if led_color not in ("B", "Y", "G"):
    return False
  follows = [
      action for action in legal
      if action != PARADOX_ACTION and _action_color(state, action) == led_color
  ]
  switches = [
      action for action in legal
      if action != PARADOX_ACTION
      and _action_color(state, action) not in (None, led_color)
  ]
  if not follows or not switches:
    return False
  if bool(args.require_red_switch_legal):
    return any(_action_color(state, action) == "R" for action in switches)
  return True


def _sample_round_decisions(
    decisions: list[dict], max_count: int, seed: int
) -> list[dict]:
  """Select decisions across the completed round, not just its first trick."""
  max_count = int(max_count)
  if max_count <= 0 or len(decisions) <= max_count:
    return list(decisions)
  rng = np.random.RandomState(_numpy_seed(seed))
  chosen = sorted(
      int(index)
      for index in rng.choice(len(decisions), size=max_count, replace=False)
  )
  return [decisions[index] for index in chosen]


def _policy_sorted(decision: dict, actions: list[int]) -> list[int]:
  policy = np.asarray(decision.get("policy", []), dtype=np.float32)
  return sorted(
      [int(action) for action in actions],
      key=lambda action: float(policy[action]) if 0 <= action < len(policy) else 0.0,
      reverse=True,
  )


def _candidate_actions(decision: dict, args) -> list[int]:
  state = decision["state"]
  legal = [
      int(action) for action in decision.get("legal", [])
      if int(action) != PARADOX_ACTION
  ]
  if bool(args.include_all_legal):
    return sorted(legal)
  candidates = []

  def add(action: int) -> None:
    action = int(action)
    if action == PARADOX_ACTION or action not in legal:
      return
    if action not in candidates:
      candidates.append(action)

  add(int(decision["action"]))
  for action in _policy_sorted(decision, legal)[:int(args.policy_top_actions)]:
    add(action)

  led_color = getattr(state, "_led_color", None)
  if led_color in ("B", "Y", "G"):
    follows = [
        action for action in legal if _action_color(state, action) == led_color
    ]
    red_switches = [
        action for action in legal if _action_color(state, action) == "R"
    ]
    off_led = [
        action for action in legal
        if _action_color(state, action) not in (None, led_color)
    ]
    for bucket in (follows, red_switches, off_led):
      for action in _policy_sorted(decision, bucket):
        add(action)
        if len(candidates) >= int(args.max_candidate_actions):
          return candidates

  for action in _policy_sorted(decision, legal):
    add(action)
    if len(candidates) >= int(args.max_candidate_actions):
      break
  return candidates


def _branch_score(record: dict, args, value_scale: float) -> float:
  actor_score = float(record["mean_actor_score"])
  total_score = float(record["mean_total_score"])
  score = (
      float(args.survival_weight) * (1.0 - float(record["any_paradox_rate"]))
      + float(args.actor_survival_weight)
      * (1.0 - float(record["actor_paradox_rate"]))
      - float(args.forced_paradox_penalty) * float(record["forced_paradox_rate"])
      + float(args.actor_score_weight)
      * math.tanh(actor_score / max(1e-6, float(value_scale)))
      + float(args.total_score_weight)
      * math.tanh(total_score / max(1e-6, float(value_scale)))
  )
  return float(score)


def _action_value_target(record: dict, args, value_scale: float) -> float:
  if float(record["any_paradox_rate"]) > 0.0:
    target = float(args.action_value_paradox_target)
  else:
    target = float(args.action_value_safe_target)
  target -= float(args.action_value_actor_penalty) * float(record["actor_paradox_rate"])
  target -= float(args.action_value_forced_penalty) * float(record["forced_paradox_rate"])
  target += float(args.action_value_score_weight) * math.tanh(
      float(record["mean_actor_score"]) / max(1e-6, float(value_scale))
  )
  lower = float(args.action_value_min)
  upper = float(args.action_value_max)
  if lower > upper:
    lower, upper = upper, lower
  return float(min(upper, max(lower, target)))


def _evaluate_action(
    model,
    decision: dict,
    action: int,
    rollout_states: list,
    value_scale: float,
    device: torch.device,
    seed_base: int,
    args,
    rollouts_per_action: int | None = None,
) -> dict:
  player = int(decision["player"])
  if rollouts_per_action is None:
    rollouts_per_action = int(args.rollouts_per_action)
  rollouts_per_world = max(1, int(rollouts_per_action))
  outcomes = []
  for world_idx, rollout_state in enumerate(rollout_states):
    for rollout_idx in range(rollouts_per_world):
      seed = _numpy_seed(
          int(seed_base) + world_idx * 104729 + rollout_idx * 7919
      )
      result = rollout_from_action(
          model,
          rollout_state,
          int(action),
          value_scale,
          device,
          seed,
          args,
      )
      outcomes.append(result)
  if not outcomes:
    raise ValueError("best-of-rollouts action evaluation has no rollout worlds")
  any_rate = float(np.mean([bool(item["any_paradox"]) for item in outcomes]))
  actor_rate = float(np.mean([bool(item["paradoxed"][player]) for item in outcomes]))
  forced_rate = float(np.mean([bool(item.get("forced_paradox", False)) for item in outcomes]))
  actor_scores = [float(item["raw_scores"][player]) for item in outcomes]
  total_scores = [float(sum(item["raw_scores"])) for item in outcomes]
  record = {
      "action": int(action),
      "action_string": action_string(decision["state"], player, int(action)),
      "color": _action_color(decision["state"], int(action)),
      "follows_led": bool(
          _action_color(decision["state"], int(action))
          == getattr(decision["state"], "_led_color", None)
      ),
      "is_red_switch": bool(
          _action_color(decision["state"], int(action)) == "R"
          and getattr(decision["state"], "_led_color", None) != "R"
      ),
      "policy_prob": float(
          np.asarray(decision.get("policy", []), dtype=np.float32)[int(action)]
      ),
      "rollouts": int(len(outcomes)),
      "rollout_worlds": int(len(rollout_states)),
      "rollouts_per_world": int(rollouts_per_world),
      "any_paradox_rate": any_rate,
      "any_paradox_outcomes": [
          bool(item["any_paradox"]) for item in outcomes
      ],
      "actor_paradox_rate": actor_rate,
      "forced_paradox_rate": forced_rate,
      "mean_actor_score": float(np.mean(actor_scores)),
      "mean_total_score": float(np.mean(total_scores)),
      "min_actor_score": float(np.min(actor_scores)),
      "max_actor_score": float(np.max(actor_scores)),
      "first_new_paradox_triggers": [
          item.get("first_new_paradox_trigger") for item in outcomes
          if item.get("first_new_paradox_trigger") is not None
      ][:3],
  }
  record["branch_score"] = _branch_score(record, args, value_scale)
  return record


def _one_sided_binomial_tail(successes: int, failures: int) -> float:
  """Exact P[X >= successes] under a fair paired-discordance null."""
  successes = int(successes)
  failures = int(failures)
  total = successes + failures
  if successes < 0 or failures < 0 or total <= 0:
    return 1.0
  numerator = sum(math.comb(total, k) for k in range(successes, total + 1))
  return float(numerator / (2**total))


def _paired_any_paradox_summary(
    base_record: dict,
    proposal_record: dict,
) -> dict | None:
  """Summarize paired evidence at the independent belief-world level.

  Continuations that start from the same resampled information-state world are
  nested replicates, not independent observations.  In particular, Quantum
  Cat's policy continuation is deterministic after the initial deal, so
  ``rollouts_per_world > 1`` can produce byte-for-byte duplicate outcomes.
  Counting those duplicates in the exact sign test would make its p-value
  anti-conservative.  Collapse each world's continuations to a mean first, then
  count a world as saved/harmed according to the sign of that paired mean
  difference.
  """
  base = base_record.get("any_paradox_outcomes")
  proposal = proposal_record.get("any_paradox_outcomes")
  if not isinstance(base, list) or not isinstance(proposal, list):
    return None
  if not base or len(base) != len(proposal):
    return None
  base_worlds = int(base_record.get("rollout_worlds", len(base)))
  proposal_worlds = int(proposal_record.get("rollout_worlds", len(proposal)))
  base_per_world = int(base_record.get("rollouts_per_world", 1))
  proposal_per_world = int(proposal_record.get("rollouts_per_world", 1))
  if (
      base_worlds <= 0
      or base_worlds != proposal_worlds
      or base_per_world <= 0
      or base_per_world != proposal_per_world
      or len(base) != base_worlds * base_per_world
  ):
    return None
  base_array = np.asarray(base, dtype=np.float64).reshape(
      base_worlds, base_per_world
  )
  proposal_array = np.asarray(proposal, dtype=np.float64).reshape(
      proposal_worlds, proposal_per_world
  )
  paired_differences = np.mean(base_array, axis=1) - np.mean(
      proposal_array, axis=1
  )
  saved = int(np.count_nonzero(paired_differences > 1e-12))
  harmed = int(np.count_nonzero(paired_differences < -1e-12))
  tied = int(base_worlds - saved - harmed)
  return {
      "inference_unit": "information_state_world",
      "paired_worlds": int(base_worlds),
      # Keep this field for existing report consumers, but make its unit the
      # independent unit used by the test rather than flattened continuations.
      "paired_outcomes": int(base_worlds),
      "continuation_outcomes": int(len(base)),
      "rollouts_per_world": int(base_per_world),
      "saved_outcomes": int(saved),
      "harmed_outcomes": int(harmed),
      "tied_outcomes": int(tied),
      "net_saved_outcomes": int(saved - harmed),
      "one_sided_binomial_p": float(
          _one_sided_binomial_tail(saved, harmed)
      ),
  }


def _rollout_states_for_decision(
    decision: dict,
    seed_base: int,
    args,
    belief_samples: int | None = None,
) -> list:
  """Returns common hidden worlds for every candidate at one decision.

  The replay row contains only the acting player's observation. Evaluating the
  target on the simulator's true opponents' hands therefore creates an
  unlearnable hidden-information label. The default path resamples those hands
  from the same information state and reuses the identical worlds for all
  candidate actions (common random numbers).
  """
  source = str(getattr(args, "rollout_state_source", "infostate"))
  state = decision["state"]
  player = int(decision["player"])
  if source == "actual":
    return [state]
  if source != "infostate":
    raise ValueError(f"unsupported rollout state source: {source}")

  expected_legal = [int(action) for action in decision["legal"]]
  expected_observation = np.asarray(
      state.observation_tensor(player), dtype=np.float32
  )
  worlds = []
  if belief_samples is None:
    belief_samples = int(getattr(args, "belief_samples", 4))
  for world_idx in range(max(1, int(belief_samples))):
    sampler = np.random.RandomState(
        _numpy_seed(int(seed_base) + world_idx * 15485863)
    )
    sampled = state.resample_from_infostate(player, sampler)
    sampled_legal = [int(action) for action in sampled.legal_actions(player)]
    sampled_observation = np.asarray(
        sampled.observation_tensor(player), dtype=np.float32
    )
    if sampled_legal != expected_legal:
      raise ValueError("belief resample changed actor legal actions")
    if not np.array_equal(sampled_observation, expected_observation):
      raise ValueError("belief resample changed actor observation")
    worlds.append(sampled)
  return worlds


def _target_from_records(records: list[dict], num_actions: int, args):
  if not records:
    return None, None
  best_score = max(float(item["branch_score"]) for item in records)
  gap = float(args.target_top_score_gap)
  if gap >= 0.0:
    target_records = [
        item for item in records if best_score - float(item["branch_score"]) <= gap
    ]
  else:
    target_records = list(records)
  if not target_records:
    return None, None
  logits = np.array(
      [float(item["branch_score"]) for item in target_records],
      dtype=np.float64,
  )
  logits /= max(1e-6, float(args.target_temperature))
  logits -= float(np.max(logits))
  weights = np.exp(logits)
  total = float(weights.sum())
  if total <= 0.0 or not math.isfinite(total):
    return None, None
  policy = np.zeros(num_actions, dtype=np.float32)
  for item, weight in zip(target_records, weights / total):
    policy[int(item["action"])] = float(weight)
  best_action = int(max(records, key=lambda item: float(item["branch_score"]))["action"])
  return policy, best_action


def _base_keep_target(decision: dict, num_actions: int) -> tuple[np.ndarray, int]:
  """Returns a hard abstention target for a fresh-audit-rejected proposal."""
  base_action = int(decision["action"])
  target = np.zeros(int(num_actions), dtype=np.float32)
  target[base_action] = 1.0
  return target, base_action


def _make_replay_row(decision: dict, records: list[dict], target, best_action: int,
                     args, players: int, value_scale: float):
  state = decision["state"]
  player = int(decision["player"])
  num_actions = int(state.num_distinct_actions())
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[[int(action) for action in decision["legal"]]] = 1.0
  risk_targets = np.zeros(num_actions, dtype=np.float32)
  risk_mask = np.zeros(num_actions, dtype=np.float32)
  value_targets = np.zeros(num_actions, dtype=np.float32)
  value_mask = np.zeros(num_actions, dtype=np.float32)
  for record in records:
    action = int(record["action"])
    risk_targets[action] = float(record["any_paradox_rate"])
    risk_mask[action] = 1.0
    value_targets[action] = _action_value_target(record, args, value_scale)
    value_mask[action] = 1.0
  return (
      np.asarray(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      np.asarray(target, dtype=np.float32),
      np.zeros(int(players), dtype=np.float32),
      np.zeros(int(players), dtype=np.float32),
      int(best_action),
      int(player),
      risk_targets,
      risk_mask,
      value_targets,
      value_mask,
      action_feature_matrix(state, player, num_actions),
  )


def _acceptance(decision: dict, records: list[dict], best_action: int, args) -> tuple[bool, str]:
  base_action = int(decision["action"])
  by_action = {int(item["action"]): item for item in records}
  base = by_action.get(base_action)
  best = by_action.get(int(best_action))
  if base is None or best is None:
    return False, "missing_base_or_best"
  any_rates = [float(item["any_paradox_rate"]) for item in records]
  if bool(args.require_survival_contrast):
    spread = max(any_rates) - min(any_rates)
    if spread < float(args.min_any_paradox_rate_spread):
      return False, "weak_survival_contrast"
  if bool(args.require_forced_paradox_contrast):
    forced_rates = [float(item["forced_paradox_rate"]) for item in records]
    forced_spread = max(forced_rates) - min(forced_rates)
    if forced_spread < float(args.min_forced_paradox_rate_spread):
      return False, "weak_forced_paradox_contrast"
  if bool(args.require_policy_target_changed) and int(best_action) == base_action:
    return False, "policy_target_unchanged"
  score_delta = float(best["branch_score"]) - float(base["branch_score"])
  if score_delta < float(args.min_best_vs_base_score_delta):
    return False, "weak_best_vs_base_score_delta"
  paradox_improvement = (
      float(base["any_paradox_rate"]) - float(best["any_paradox_rate"])
  )
  if paradox_improvement < float(
      getattr(args, "min_best_vs_base_any_paradox_improvement", 0.0)
  ):
    return False, "weak_best_vs_base_any_paradox_improvement"
  return True, "accepted"


def _audit_acceptance(
    decision: dict, records: list[dict], best_action: int, args
) -> tuple[bool, str, float]:
  """Require a selected label to survive fresh belief-world evaluation."""
  by_action = {int(item["action"]): item for item in records}
  base = by_action.get(int(decision["action"]))
  best = by_action.get(int(best_action))
  if base is None or best is None:
    return False, "missing_audit_base_or_best", float("-inf")
  improvement = float(base["any_paradox_rate"]) - float(
      best["any_paradox_rate"]
  )
  required = float(
      getattr(args, "min_audit_best_vs_base_any_paradox_improvement", 0.0)
  )
  if improvement < required:
    return (
        False,
        "weak_audit_best_vs_base_any_paradox_improvement",
        improvement,
    )
  max_p = float(getattr(args, "max_audit_paired_binomial_p", 1.0))
  if max_p < 1.0:
    paired = _paired_any_paradox_summary(base, best)
    if paired is None:
      return False, "missing_audit_paired_outcomes", improvement
    if float(paired["one_sided_binomial_p"]) > max_p + 1e-12:
      return False, "weak_audit_paired_evidence", improvement
  return True, "accepted", improvement


def _public_decision(decision: dict) -> dict:
  state = decision["state"]
  player = int(decision["player"])
  legal = [int(action) for action in decision["legal"]]
  led_color = getattr(state, "_led_color", None)
  return {
      "index": int(decision["index"]),
      "phase": str(decision["phase"]),
      "player": player,
      "trick_number": int(decision.get("trick_number", -1)),
      "led_color": led_color,
      "legal_count": int(len(legal)),
      "legal_actions": legal,
      "legal_action_strings": [
          action_string(state, player, action) for action in legal
      ],
      "policy_action": int(decision["action"]),
      "policy_action_string": str(decision["action_string"]),
      "policy_prob": float(decision["policy_prob"]),
  }


def main() -> None:
  args = parse_args()
  if not 0.0 <= float(args.max_audit_paired_binomial_p) <= 1.0:
    raise ValueError("--max-audit-paired-binomial-p must be in [0, 1]")
  # Validate the source actor once before generation. Historically an unknown
  # query-style shield spec silently fell through to raw greedy policy while
  # its metadata still claimed shielded provenance.
  source_bot_mode = str(args.bot_mode)
  args = resolve_actor_bot_args(args)
  if int(args.audit_belief_samples) < 0:
    raise ValueError("--audit-belief-samples must be non-negative")
  if int(args.audit_rollouts_per_action) < 0:
    raise ValueError("--audit-rollouts-per-action must be non-negative")
  if (
      float(args.min_audit_best_vs_base_any_paradox_improvement) > 0.0
      and int(args.audit_belief_samples) == 0
  ):
    raise ValueError(
        "positive --min-audit-best-vs-base-any-paradox-improvement requires "
        "--audit-belief-samples"
    )
  if (
      int(args.audit_belief_samples) > 0
      and str(args.rollout_state_source) != "infostate"
  ):
    raise ValueError(
        "held-out belief validation requires --rollout-state-source=infostate"
    )
  np.random.seed(_numpy_seed(args.seed))
  torch.manual_seed(int(args.seed))
  device = _device_from_arg(str(args.device))
  model, model_args = load_neural(args.checkpoint, int(args.players), device, args)
  if model is None:
    raise ValueError("--checkpoint failed to load")
  value_scale = float(getattr(model_args, "value_scale", 20.0))
  allowed_phases = parse_phase_filter(args.decision_phases)
  rows = []
  metadata_rows = []
  audit_rejected_base_rows = []
  audit_rejected_base_metadata_rows = []
  summary = Counter()
  reject_reasons = Counter()

  def reached_cap() -> bool:
    if int(args.max_decisions) > 0 and summary["decisions_evaluated"] >= int(args.max_decisions):
      return True
    return bool(
        int(args.max_accepted_rows) > 0
        and summary["accepted_rows"] >= int(args.max_accepted_rows)
    )

  for match_idx in range(int(args.matches)):
    if reached_cap():
      break
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      if reached_cap():
        break
      start_player = round_idx % int(args.players)
      game = pyspiel.load_game(
          "python_quantum_cat",
          {
              "players": int(args.players),
              "start_player": int(start_player),
              "match_context": int(bool(args.match_context)),
          },
      )
      prior_totals = np.copy(totals)
      round_seed = int(args.seed) + match_idx * 1000003 + round_idx * 9973
      terminal, decisions, _first_new_paradox = play_round_with_history(
          model,
          game,
          seed=round_seed,
          value_scale=value_scale,
          device=device,
          match_totals=prior_totals if args.match_context else None,
          round_index=round_idx,
          args=args,
      )
      result = terminal_round_result(terminal)
      returns = np.asarray(terminal.returns(), dtype=np.float32)
      totals = returns if args.match_context else totals + returns
      summary["rounds"] += 1
      summary["paradox_rounds"] += int(bool(result["any_paradox"]))
      eligible_decisions = [
          decision for decision in decisions
          if _decision_allowed(decision, allowed_phases, args)
      ]
      selected_decisions = _sample_round_decisions(
          eligible_decisions,
          int(args.max_decisions_per_round),
          seed=round_seed + 32452843,
      )
      for decision in selected_decisions:
        if reached_cap():
          break
        candidates = _candidate_actions(decision, args)
        if len(candidates) < 2:
          reject_reasons["too_few_candidates"] += 1
          continue
        summary["decisions_evaluated"] += 1
        summary[f"decisions_phase_{phase_name(decision['state'])}"] += 1
        summary[
            f"decisions_trick_{int(decision.get('trick_number', -1))}"
        ] += 1
        seed_base = round_seed + int(decision["index"]) * 1009 + 17000019
        try:
          rollout_states = _rollout_states_for_decision(
              decision, seed_base=seed_base + 70000001, args=args
          )
        except Exception as exc:  # Fail closed; never substitute the true deal.
          reject_reasons[f"belief_resample_failed:{type(exc).__name__}"] += 1
          continue
        records = [
            _evaluate_action(
                model,
                decision,
                action,
                rollout_states,
                value_scale,
                device,
                # Use common continuation seeds for every candidate. The old
                # action-dependent seed added label noise to small rollout sets.
                seed_base=seed_base,
                args=args,
            )
            for action in candidates
        ]
        target, best_action = _target_from_records(
            records,
            int(decision["state"].num_distinct_actions()),
            args,
        )
        if target is None:
          reject_reasons["target_failed"] += 1
          continue
        accepted, reason = _acceptance(decision, records, int(best_action), args)
        if not accepted:
          reject_reasons[reason] += 1
          continue
        audit_records = []
        audit_improvement = None
        if int(args.audit_belief_samples) > 0:
          try:
            audit_states = _rollout_states_for_decision(
                decision,
                seed_base=seed_base + 900000007,
                args=args,
                belief_samples=int(args.audit_belief_samples),
            )
          except Exception as exc:  # Fresh belief validation also fails closed.
            reject_reasons[
                f"audit_belief_resample_failed:{type(exc).__name__}"
            ] += 1
            continue
          audit_rollouts = int(args.audit_rollouts_per_action) or int(
              args.rollouts_per_action
          )
          audit_actions = list(dict.fromkeys(
              [int(decision["action"]), int(best_action)]
          ))
          audit_records = [
              _evaluate_action(
                  model,
                  decision,
                  action,
                  audit_states,
                  value_scale,
                  device,
                  seed_base=seed_base + 800000011,
                  args=args,
                  rollouts_per_action=audit_rollouts,
              )
              for action in audit_actions
          ]
          audit_accepted, audit_reason, audit_improvement = _audit_acceptance(
              decision, audit_records, int(best_action), args
          )
          if not audit_accepted:
            reject_reasons[audit_reason] += 1
            negative_cap = int(args.max_audit_rejected_base_rows)
            can_emit_negative = (
                bool(str(args.audit_rejected_base_out or "").strip())
                and audit_reason
                == "weak_audit_best_vs_base_any_paradox_improvement"
                and (
                    negative_cap <= 0
                    or len(audit_rejected_base_rows) < negative_cap
                )
            )
            if can_emit_negative:
              base_target, base_action = _base_keep_target(
                  decision, int(decision["state"].num_distinct_actions())
              )
              negative_row = _make_replay_row(
                  decision,
                  records,
                  base_target,
                  int(base_action),
                  args,
                  int(args.players),
                  value_scale,
              )
              negative_metadata = {
                  "source": "best_of_rollouts_audit_rejected_base",
                  "source_index": int(len(audit_rejected_base_metadata_rows)),
                  "match_index": int(match_idx),
                  "round_index": int(round_idx),
                  "round_seed": int(round_seed),
                  "start_player": int(start_player),
                  "terminal_any_paradox": bool(result["any_paradox"]),
                  "decision": _public_decision(decision),
                  "public_history": public_history_metadata(
                      decision["state"], int(decision["player"])
                  ),
                  "candidate_actions": [int(item) for item in candidates],
                  "best_action": int(base_action),
                  "best_action_string": action_string(
                      decision["state"], int(decision["player"]), int(base_action)
                  ),
                  "proposed_best_action": int(best_action),
                  "proposed_best_action_string": action_string(
                      decision["state"], int(decision["player"]), int(best_action)
                  ),
                  "policy_target_changed": False,
                  "audit_override_rejected": True,
                  "audit_base_keep_validation": True,
                  "independent_belief_validation": True,
                  "audit_best_vs_base_any_paradox_improvement": float(
                      audit_improvement
                  ),
                  "audit_belief_samples": int(len(audit_states)),
                  "audit_rollouts_per_action": int(audit_rollouts),
                  "audit_rollout_records": audit_records,
                  "target_probability_on_policy_action": 1.0,
                  "target_probability_on_best_action": 1.0,
                  "rollout_state_source": str(args.rollout_state_source),
                  "belief_samples": int(len(rollout_states)),
                  "information_safe_rollout_teacher": True,
                  "rollout_records": records,
                  "selection_proposed_best_vs_base_any_paradox_improvement": float(
                      next(
                          item for item in records
                          if int(item["action"]) == int(base_action)
                      )["any_paradox_rate"]
                      - next(
                          item for item in records
                          if int(item["action"]) == int(best_action)
                      )["any_paradox_rate"]
                  ),
              }
              audit_rejected_base_rows.append(negative_row)
              audit_rejected_base_metadata_rows.append(negative_metadata)
              summary["audit_rejected_base_rows"] += 1
            continue
        row = _make_replay_row(
            decision,
            records,
            target,
            int(best_action),
            args,
            int(args.players),
            value_scale,
        )
        metadata = {
            "source": "best_of_rollouts",
            "source_index": int(len(metadata_rows)),
            "match_index": int(match_idx),
            "round_index": int(round_idx),
            "round_seed": int(round_seed),
            "start_player": int(start_player),
            "terminal_any_paradox": bool(result["any_paradox"]),
            "decision": _public_decision(decision),
            "public_history": public_history_metadata(
                decision["state"], int(decision["player"])
            ),
            "candidate_actions": [int(item) for item in candidates],
            "best_action": int(best_action),
            "best_action_string": action_string(
                decision["state"], int(decision["player"]), int(best_action)
            ),
            "policy_target_changed": bool(int(best_action) != int(decision["action"])),
            "best_vs_base_any_paradox_improvement": float(
                next(
                    item for item in records
                    if int(item["action"]) == int(decision["action"])
                )["any_paradox_rate"]
                - next(
                    item for item in records
                    if int(item["action"]) == int(best_action)
                )["any_paradox_rate"]
            ),
            "independent_belief_validation": bool(audit_records),
            "audit_best_vs_base_any_paradox_improvement": (
                None if audit_improvement is None else float(audit_improvement)
            ),
            "audit_belief_samples": int(
                len(audit_states) if audit_records else 0
            ),
            "audit_rollouts_per_action": int(
                (int(args.audit_rollouts_per_action) or int(args.rollouts_per_action))
                if audit_records else 0
            ),
            "audit_rollout_records": audit_records,
            "target_probability_on_policy_action": float(target[int(decision["action"])]),
            "target_probability_on_best_action": float(target[int(best_action)]),
            "rollout_state_source": str(args.rollout_state_source),
            "belief_samples": int(
                len(rollout_states)
                if str(args.rollout_state_source) == "infostate" else 0
            ),
            "information_safe_rollout_teacher": bool(
                str(args.rollout_state_source) == "infostate"
            ),
            "rollout_records": records,
            "any_paradox_rate_spread": float(
                max(float(item["any_paradox_rate"]) for item in records)
                - min(float(item["any_paradox_rate"]) for item in records)
            ),
            "branch_score_spread": float(
                max(float(item["branch_score"]) for item in records)
                - min(float(item["branch_score"]) for item in records)
            ),
            "forced_paradox_rate_spread": float(
                max(float(item["forced_paradox_rate"]) for item in records)
                - min(float(item["forced_paradox_rate"]) for item in records)
            ),
        }
        rows.append(row)
        metadata_rows.append(metadata)
        summary["accepted_rows"] += 1
        summary["policy_target_changed_rows"] += int(
            bool(metadata["policy_target_changed"])
        )
        summary["independent_belief_validated_rows"] += int(
            bool(metadata["independent_belief_validation"])
        )
        summary["forced_paradox_contrast_rows"] += int(
            float(metadata["forced_paradox_rate_spread"])
            >= float(args.min_forced_paradox_rate_spread)
        )
        if int(args.progress_interval) > 0 and (
            summary["decisions_evaluated"] % int(args.progress_interval) == 0
        ):
          print(json.dumps({
              "iteration": "best_of_rollouts_progress",
              "matches": int(match_idx + 1),
              "rounds": int(summary["rounds"]),
              "decisions_evaluated": int(summary["decisions_evaluated"]),
              "accepted_rows": int(summary["accepted_rows"]),
              "paradox_rounds": int(summary["paradox_rounds"]),
          }), flush=True)

  out = Path(args.out)
  metadata_out = (
      Path(args.metadata_out)
      if str(args.metadata_out or "").strip()
      else out.with_suffix(out.suffix + ".metadata.json")
  )
  summary_out = (
      Path(args.summary_out)
      if str(args.summary_out or "").strip()
      else out.with_suffix(out.suffix + ".summary.json")
  )
  metadata = {
      "source": "quantum_cat_best_of_rollouts_replay",
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches_requested": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "bot_mode": source_bot_mode,
      "effective_bot_mode": str(args.bot_mode),
      "device": str(device),
      "value_scale": float(value_scale),
      "decision_phases": str(args.decision_phases),
      "rollouts_per_action": int(args.rollouts_per_action),
      "rollout_state_source": str(args.rollout_state_source),
      "belief_samples": int(args.belief_samples),
      "audit_belief_samples": int(args.audit_belief_samples),
      "audit_rollouts_per_action": int(
          int(args.audit_rollouts_per_action) or int(args.rollouts_per_action)
      ) if int(args.audit_belief_samples) > 0 else 0,
      "information_safe_rollout_teacher": bool(
          str(args.rollout_state_source) == "infostate"
      ),
      "max_candidate_actions": int(args.max_candidate_actions),
      "policy_top_actions": int(args.policy_top_actions),
      "include_all_legal": bool(args.include_all_legal),
      "require_policy_target_changed": bool(args.require_policy_target_changed),
      "require_survival_contrast": bool(args.require_survival_contrast),
      "min_any_paradox_rate_spread": float(args.min_any_paradox_rate_spread),
      "require_forced_paradox_contrast": bool(
          args.require_forced_paradox_contrast
      ),
      "min_forced_paradox_rate_spread": float(
          args.min_forced_paradox_rate_spread
      ),
      "source_group_id": str(args.source_group_id),
      "min_best_vs_base_score_delta": float(args.min_best_vs_base_score_delta),
      "min_best_vs_base_any_paradox_improvement": float(
          args.min_best_vs_base_any_paradox_improvement
      ),
      "min_audit_best_vs_base_any_paradox_improvement": float(
          args.min_audit_best_vs_base_any_paradox_improvement
      ),
      "max_audit_paired_binomial_p": float(
          args.max_audit_paired_binomial_p
      ),
      "target_temperature": float(args.target_temperature),
      "target_top_score_gap": float(args.target_top_score_gap),
      "action_value_safe_target": float(args.action_value_safe_target),
      "action_value_paradox_target": float(args.action_value_paradox_target),
      "action_value_actor_penalty": float(args.action_value_actor_penalty),
      "action_value_forced_penalty": float(args.action_value_forced_penalty),
      "action_value_score_weight": float(args.action_value_score_weight),
      "action_value_min": float(args.action_value_min),
      "action_value_max": float(args.action_value_max),
      "summary": {key: int(value) for key, value in sorted(summary.items())},
      "reject_reasons": {
          key: int(value) for key, value in sorted(reject_reasons.items())
      },
      "rows": int(len(rows)),
      "metadata_out": str(metadata_out),
      "summary_out": str(summary_out),
  }
  if str(args.source_group_id):
    metadata["row_group_ids"] = [str(args.source_group_id)] * len(rows)
    metadata["row_group_id_kind"] = "independent_source_shard"
    metadata["row_group_count"] = 1 if rows else 0
  if rows:
    save_replay(rows, out, metadata=metadata)
  elif not bool(args.write_empty):
    raise RuntimeError(
        "no rows accepted; pass --write-empty to write diagnostics anyway"
    )
  metadata_out.parent.mkdir(parents=True, exist_ok=True)
  metadata_out.write_text(
      json.dumps({**metadata, "rows": metadata_rows}, indent=2, sort_keys=True)
      + "\n"
  )
  if str(args.audit_rejected_base_out or "").strip():
    audit_rejected_base_out = Path(args.audit_rejected_base_out)
    audit_rejected_base_metadata_out = (
        Path(args.audit_rejected_base_metadata_out)
        if str(args.audit_rejected_base_metadata_out or "").strip()
        else audit_rejected_base_out.with_suffix(
            audit_rejected_base_out.suffix + ".metadata.json"
        )
    )
    audit_rejected_base_metadata = {
        "source": "quantum_cat_best_of_rollouts_audit_rejected_base",
        "checkpoint": str(args.checkpoint),
        "players": int(args.players),
        "matches_requested": int(args.matches),
        "seed": int(args.seed),
        "match_context": bool(args.match_context),
        "bot_mode": source_bot_mode,
        "effective_bot_mode": str(args.bot_mode),
        "decision_phases": str(args.decision_phases),
        "rollout_state_source": str(args.rollout_state_source),
        "belief_samples": int(args.belief_samples),
        "audit_belief_samples": int(args.audit_belief_samples),
        "audit_rollouts_per_action": int(
            int(args.audit_rollouts_per_action) or int(args.rollouts_per_action)
        ),
        "information_safe_rollout_teacher": bool(
            str(args.rollout_state_source) == "infostate"
        ),
        "independent_belief_validation": True,
        "target_kind": "hard_keep_source_policy_action_after_audit_rejection",
        "source_group_id": str(args.source_group_id),
        "min_best_vs_base_any_paradox_improvement": float(
            args.min_best_vs_base_any_paradox_improvement
        ),
        "min_audit_best_vs_base_any_paradox_improvement": float(
            args.min_audit_best_vs_base_any_paradox_improvement
        ),
        "rows": int(len(audit_rejected_base_rows)),
        "metadata_out": str(audit_rejected_base_metadata_out),
    }
    if str(args.source_group_id):
      audit_rejected_base_metadata["row_group_ids"] = [
          str(args.source_group_id)
      ] * len(audit_rejected_base_rows)
      audit_rejected_base_metadata["row_group_id_kind"] = (
          "independent_source_shard"
      )
      audit_rejected_base_metadata["row_group_count"] = (
          1 if audit_rejected_base_rows else 0
      )
    if audit_rejected_base_rows:
      save_replay(
          audit_rejected_base_rows,
          audit_rejected_base_out,
          metadata=audit_rejected_base_metadata,
      )
    audit_rejected_base_metadata_out.parent.mkdir(parents=True, exist_ok=True)
    audit_rejected_base_metadata_out.write_text(
        json.dumps(
            {
                **audit_rejected_base_metadata,
                "rows": audit_rejected_base_metadata_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
  summary_out.parent.mkdir(parents=True, exist_ok=True)
  summary_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
