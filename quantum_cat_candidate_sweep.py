"""Run small AlphaZero candidate sweeps and rank snapshot checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


RECIPES = {
    "short_spike": {
        "anchor_kl_weight": "0.90",
        "terminal_paradox_penalty": "0.90",
        "value_loss_weight": "0.85",
        "lr": "2e-6",
        "temperature": "1.0",
        "temperature_drop": "16",
        "self_play_belief_sims": "16",
        "train_steps": "80",
    },
    "pure_policy_explore": {
        "anchor_kl_weight": "0.74",
        "terminal_paradox_penalty": "0.95",
        "value_loss_weight": "0.95",
        "lr": "1.4e-6",
        "temperature": "1.22",
        "temperature_drop": "28",
        "self_play_belief_sims": "28",
        "train_steps": "120",
        "extra_args": [
            "--c-puct=2.25",
        ],
    },
    "pure_policy_balanced": {
        "anchor_kl_weight": "0.82",
        "terminal_paradox_penalty": "1.00",
        "value_loss_weight": "1.00",
        "lr": "1.2e-6",
        "temperature": "1.00",
        "temperature_drop": "22",
        "self_play_belief_sims": "24",
        "train_steps": "120",
        "extra_args": [
            "--c-puct=2.00",
        ],
    },
    "pure_policy_conservative": {
        "anchor_kl_weight": "1.08",
        "terminal_paradox_penalty": "1.05",
        "value_loss_weight": "0.90",
        "lr": "7e-7",
        "temperature": "0.82",
        "temperature_drop": "18",
        "self_play_belief_sims": "24",
        "train_steps": "140",
        "extra_args": [
            "--c-puct=1.65",
        ],
    },
    "score_focus": {
        "anchor_kl_weight": "0.92",
        "terminal_paradox_penalty": "0.80",
        "value_loss_weight": "1.10",
        "lr": "1.5e-6",
        "temperature": "0.9",
        "temperature_drop": "16",
        "self_play_belief_sims": "16",
        "train_steps": "100",
    },
    "safety_diverse": {
        "anchor_kl_weight": "0.84",
        "terminal_paradox_penalty": "1.05",
        "value_loss_weight": "0.90",
        "lr": "1.3e-6",
        "temperature": "1.15",
        "temperature_drop": "24",
        "self_play_belief_sims": "20",
        "train_steps": "100",
    },
    "ordinal_safety": {
        "anchor_kl_weight": "0.86",
        "terminal_paradox_penalty": "1.00",
        "value_loss_weight": "0.75",
        "lr": "1.1e-6",
        "temperature": "1.05",
        "temperature_drop": "20",
        "self_play_belief_sims": "20",
        "train_steps": "100",
        "extra_args": [
            "--ordinal-value-weight=0.45",
            "--paradox-loss-weight=0.05",
        ],
    },
    "official_score": {
        "anchor_kl_weight": "0.74",
        "terminal_paradox_penalty": "0.70",
        "value_loss_weight": "1.10",
        "lr": "1.2e-6",
        "temperature": "0.95",
        "temperature_drop": "18",
        "self_play_belief_sims": "20",
        "train_steps": "120",
        "extra_args": [
            "--ordinal-value-weight=0.20",
            "--official-outcome-value-weight=0.80",
        ],
    },
    "official_safety": {
        "anchor_kl_weight": "0.78",
        "terminal_paradox_penalty": "1.05",
        "value_loss_weight": "1.00",
        "lr": "1.0e-6",
        "temperature": "1.05",
        "temperature_drop": "22",
        "self_play_belief_sims": "20",
        "train_steps": "120",
        "extra_args": [
            "--ordinal-value-weight=0.30",
            "--official-outcome-value-weight=0.70",
            "--paradox-loss-weight=0.04",
        ],
    },
    "official_explore": {
        "anchor_kl_weight": "0.62",
        "terminal_paradox_penalty": "0.85",
        "value_loss_weight": "1.05",
        "lr": "1.5e-6",
        "temperature": "1.20",
        "temperature_drop": "28",
        "self_play_belief_sims": "28",
        "train_steps": "120",
        "extra_args": [
            "--c-puct=2.20",
            "--ordinal-value-weight=0.15",
            "--official-outcome-value-weight=0.70",
        ],
    },
    "search_league_safety": {
        "anchor_kl_weight": "0.90",
        "terminal_paradox_penalty": "1.05",
        "value_loss_weight": "0.95",
        "lr": "9e-7",
        "temperature": "1.05",
        "temperature_drop": "22",
        "self_play_belief_sims": "20",
        "train_steps": "100",
        "league_games": "48",
        "league_train_steps": "80",
        "league_opponent_mode": "mcts",
        "extra_args": [
            "--ordinal-value-weight=0.25",
            "--official-outcome-value-weight=0.45",
            "--paradox-loss-weight=0.04",
        ],
    },
    "search_league_official": {
        "anchor_kl_weight": "0.84",
        "terminal_paradox_penalty": "0.90",
        "value_loss_weight": "1.05",
        "lr": "1.0e-6",
        "temperature": "1.10",
        "temperature_drop": "24",
        "self_play_belief_sims": "24",
        "train_steps": "120",
        "league_games": "48",
        "league_train_steps": "80",
        "league_opponent_mode": "mcts",
        "extra_args": [
            "--c-puct=2.00",
            "--ordinal-value-weight=0.20",
            "--official-outcome-value-weight=0.65",
        ],
    },
    "champ_mcts_league_score": {
        "anchor_kl_weight": "0.88",
        "terminal_paradox_penalty": "0.85",
        "value_loss_weight": "1.10",
        "lr": "8e-7",
        "temperature": "0.95",
        "temperature_drop": "22",
        "self_play_belief_sims": "24",
        "train_steps": "140",
        "league_games": "96",
        "league_train_steps": "120",
        "league_opponent_mode": "mcts",
        "league_checkpoint_count": 2,
        "league_bots": "",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.20",
            "--official-outcome-value-weight=0.75",
        ],
    },
    "champ_mcts_league_balanced": {
        "anchor_kl_weight": "0.94",
        "terminal_paradox_penalty": "0.98",
        "value_loss_weight": "1.05",
        "lr": "7.5e-7",
        "temperature": "0.98",
        "temperature_drop": "22",
        "self_play_belief_sims": "24",
        "train_steps": "140",
        "league_games": "96",
        "league_train_steps": "120",
        "league_opponent_mode": "mcts",
        "league_checkpoint_count": 2,
        "league_bots": "",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.23",
            "--official-outcome-value-weight=0.62",
            "--paradox-loss-weight=0.02",
        ],
    },
    "champ_mcts_league_safety": {
        "anchor_kl_weight": "0.98",
        "terminal_paradox_penalty": "1.10",
        "value_loss_weight": "1.00",
        "lr": "7e-7",
        "temperature": "0.95",
        "temperature_drop": "22",
        "self_play_belief_sims": "24",
        "train_steps": "140",
        "league_games": "96",
        "league_train_steps": "120",
        "league_opponent_mode": "mcts",
        "league_checkpoint_count": 2,
        "league_bots": "",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.25",
            "--official-outcome-value-weight=0.55",
            "--paradox-loss-weight=0.04",
        ],
    },
    "champ_mixed_league_balanced": {
        "anchor_kl_weight": "0.94",
        "terminal_paradox_penalty": "1.00",
        "value_loss_weight": "1.05",
        "lr": "7.5e-7",
        "temperature": "1.00",
        "temperature_drop": "22",
        "self_play_belief_sims": "24",
        "train_steps": "140",
        "league_games": "120",
        "league_train_steps": "140",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.24",
            "--official-outcome-value-weight=0.62",
            "--paradox-loss-weight=0.025",
        ],
    },
    "champ_mixed_league_safety": {
        "anchor_kl_weight": "1.00",
        "terminal_paradox_penalty": "1.12",
        "value_loss_weight": "1.00",
        "lr": "7e-7",
        "temperature": "0.96",
        "temperature_drop": "22",
        "self_play_belief_sims": "24",
        "train_steps": "140",
        "league_games": "120",
        "league_train_steps": "140",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.27",
            "--official-outcome-value-weight=0.56",
            "--paradox-loss-weight=0.05",
        ],
    },
    "champ_mixed_field_balanced": {
        "anchor_kl_weight": "0.93",
        "terminal_paradox_penalty": "1.00",
        "value_loss_weight": "1.05",
        "lr": "7.5e-7",
        "temperature": "1.00",
        "temperature_drop": "22",
        "self_play_belief_sims": "24",
        "train_steps": "140",
        "league_games": "144",
        "league_train_steps": "140",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.24",
            "--official-outcome-value-weight=0.62",
            "--paradox-loss-weight=0.03",
        ],
    },
    "champ_mixed_field_safety": {
        "anchor_kl_weight": "0.99",
        "terminal_paradox_penalty": "1.12",
        "value_loss_weight": "1.00",
        "lr": "7e-7",
        "temperature": "0.96",
        "temperature_drop": "22",
        "self_play_belief_sims": "24",
        "train_steps": "140",
        "league_games": "144",
        "league_train_steps": "140",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.28",
            "--official-outcome-value-weight=0.56",
            "--paradox-loss-weight=0.06",
        ],
    },
    "champ_mixed_field_safety_long": {
        "anchor_kl_weight": "1.02",
        "terminal_paradox_penalty": "1.12",
        "value_loss_weight": "1.00",
        "lr": "6e-7",
        "temperature": "0.94",
        "temperature_drop": "24",
        "self_play_belief_sims": "28",
        "train_steps": "180",
        "league_games": "216",
        "league_train_steps": "180",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.30",
            "--official-outcome-value-weight=0.58",
            "--paradox-loss-weight=0.06",
        ],
    },
    "champ_mixed_field_action_value": {
        "anchor_kl_weight": "0.96",
        "terminal_paradox_penalty": "1.04",
        "value_loss_weight": "0.92",
        "lr": "7e-7",
        "temperature": "1.02",
        "temperature_drop": "24",
        "self_play_belief_sims": "28",
        "train_steps": "160",
        "league_games": "180",
        "league_train_steps": "160",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.26",
            "--official-outcome-value-weight=0.62",
            "--paradox-loss-weight=0.03",
            "--action-value-loss-weight=0.20",
            "--action-value-selection-weight=0.04",
            "--action-value-root-only",
            "--counterfactual-action-value-rollouts=1",
            "--counterfactual-action-value-advantage",
            "--counterfactual-action-max-legal=5",
            "--counterfactual-action-label-prob=0.32",
        ],
    },
    "champ_mixed_field_action_safety": {
        "anchor_kl_weight": "1.00",
        "terminal_paradox_penalty": "1.14",
        "value_loss_weight": "0.88",
        "lr": "6.5e-7",
        "temperature": "1.04",
        "temperature_drop": "24",
        "self_play_belief_sims": "28",
        "train_steps": "160",
        "league_games": "180",
        "league_train_steps": "160",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.28",
            "--official-outcome-value-weight=0.58",
            "--paradox-loss-weight=0.07",
            "--action-paradox-loss-weight=0.16",
            "--action-paradox-selection-penalty=0.12",
            "--action-paradox-root-only",
            "--counterfactual-action-rollouts=1",
            "--counterfactual-action-max-legal=5",
            "--counterfactual-action-label-prob=0.32",
        ],
    },
    "champ_mixed_field_belief_action_value": {
        "anchor_kl_weight": "0.96",
        "terminal_paradox_penalty": "1.04",
        "value_loss_weight": "0.92",
        "lr": "7e-7",
        "temperature": "1.02",
        "temperature_drop": "24",
        "self_play_belief_sims": "28",
        "train_steps": "160",
        "league_games": "180",
        "league_train_steps": "160",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.26",
            "--official-outcome-value-weight=0.62",
            "--paradox-loss-weight=0.03",
            "--action-value-loss-weight=0.20",
            "--action-value-selection-weight=0.04",
            "--action-value-root-only",
            "--counterfactual-action-value-rollouts=1",
            "--counterfactual-action-value-advantage",
            "--counterfactual-belief-samples=4",
            "--counterfactual-action-max-legal=5",
            "--counterfactual-action-label-prob=0.32",
        ],
    },
    "champ_mixed_field_belief_action_safety": {
        "anchor_kl_weight": "1.00",
        "terminal_paradox_penalty": "1.14",
        "value_loss_weight": "0.88",
        "lr": "6.5e-7",
        "temperature": "1.04",
        "temperature_drop": "24",
        "self_play_belief_sims": "28",
        "train_steps": "160",
        "league_games": "180",
        "league_train_steps": "160",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.28",
            "--official-outcome-value-weight=0.58",
            "--paradox-loss-weight=0.07",
            "--action-paradox-loss-weight=0.16",
            "--action-paradox-selection-penalty=0.12",
            "--action-paradox-root-only",
            "--counterfactual-action-rollouts=1",
            "--counterfactual-belief-samples=4",
            "--counterfactual-action-max-legal=5",
            "--counterfactual-action-label-prob=0.32",
        ],
    },
    "champ_mixed_field_belief_action_value_fast": {
        "anchor_kl_weight": "0.96",
        "terminal_paradox_penalty": "1.04",
        "value_loss_weight": "0.92",
        "lr": "7e-7",
        "temperature": "1.02",
        "temperature_drop": "24",
        "self_play_belief_sims": "24",
        "train_steps": "120",
        "league_games": "96",
        "league_train_steps": "100",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.26",
            "--official-outcome-value-weight=0.62",
            "--paradox-loss-weight=0.03",
            "--action-value-loss-weight=0.20",
            "--action-value-selection-weight=0.04",
            "--action-value-root-only",
            "--counterfactual-action-value-rollouts=1",
            "--counterfactual-action-value-advantage",
            "--counterfactual-belief-samples=3",
            "--counterfactual-rollout-opponent-mode=policy",
            "--counterfactual-action-max-legal=4",
            "--counterfactual-action-label-prob=0.24",
        ],
    },
    "champ_mixed_field_fullmatch_action_value_fast": {
        "anchor_kl_weight": "0.92",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "0.95",
        "lr": "6e-7",
        "temperature": "1.02",
        "temperature_drop": "24",
        "self_play_belief_sims": "20",
        "train_steps": "120",
        "league_games": "96",
        "league_train_steps": "120",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.24",
            "--official-outcome-value-weight=0.68",
            "--paradox-loss-weight=0.03",
            "--action-value-loss-weight=0.45",
            "--action-value-selection-weight=0.0",
            "--counterfactual-action-value-rollouts=1",
            "--counterfactual-action-value-advantage",
            "--counterfactual-full-match-rollout",
            "--counterfactual-belief-samples=3",
            "--counterfactual-rollout-opponent-mode=policy",
            "--counterfactual-action-max-legal=4",
            "--counterfactual-action-label-prob=0.22",
            "--c-puct=1.95",
        ],
    },
    "champ_mixed_field_fullmatch_policy_shaping_fast": {
        "anchor_kl_weight": "0.90",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "0.95",
        "lr": "7e-7",
        "temperature": "1.04",
        "temperature_drop": "24",
        "self_play_belief_sims": "20",
        "train_steps": "140",
        "league_games": "96",
        "league_train_steps": "140",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.24",
            "--official-outcome-value-weight=0.68",
            "--paradox-loss-weight=0.03",
            "--action-value-loss-weight=0.25",
            "--action-value-selection-weight=0.0",
            "--counterfactual-action-value-rollouts=1",
            "--counterfactual-action-value-advantage",
            "--counterfactual-full-match-rollout",
            "--counterfactual-policy-target-weight=0.35",
            "--counterfactual-policy-target-temperature=0.08",
            "--counterfactual-belief-samples=3",
            "--counterfactual-rollout-opponent-mode=policy",
            "--counterfactual-action-max-legal=4",
            "--counterfactual-action-label-prob=0.24",
            "--c-puct=2.00",
        ],
    },
    "champ_belief_fullmatch_policy_shaping_fast": {
        "anchor_kl_weight": "0.96",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "0.98",
        "lr": "5e-7",
        "temperature": "1.00",
        "temperature_drop": "22",
        "self_play_belief_sims": "20",
        "train_steps": "120",
        "league_games": "128",
        "league_train_steps": "140",
        "league_opponent_mode": "belief",
        "league_checkpoint_count": 3,
        "league_bots": "",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.22",
            "--official-outcome-value-weight=0.72",
            "--paradox-loss-weight=0.03",
            "--action-value-loss-weight=0.20",
            "--action-value-selection-weight=0.0",
            "--counterfactual-action-value-rollouts=1",
            "--counterfactual-action-value-advantage",
            "--counterfactual-full-match-rollout",
            "--counterfactual-policy-target-weight=0.25",
            "--counterfactual-policy-target-temperature=0.10",
            "--counterfactual-belief-samples=3",
            "--counterfactual-rollout-opponent-mode=policy",
            "--counterfactual-action-max-legal=4",
            "--counterfactual-action-label-prob=0.22",
            "--c-puct=1.90",
        ],
    },
    "champ_belief_fullmatch_policy_shaping_avg2_fast": {
        "anchor_kl_weight": "0.98",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "0.98",
        "lr": "4.5e-7",
        "temperature": "0.98",
        "temperature_drop": "22",
        "self_play_belief_sims": "20",
        "train_steps": "100",
        "league_games": "128",
        "league_train_steps": "120",
        "league_opponent_mode": "belief",
        "league_checkpoint_count": 3,
        "league_bots": "",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.22",
            "--official-outcome-value-weight=0.72",
            "--paradox-loss-weight=0.03",
            "--action-value-loss-weight=0.18",
            "--action-value-selection-weight=0.0",
            "--counterfactual-action-value-rollouts=2",
            "--counterfactual-action-value-advantage",
            "--counterfactual-full-match-rollout",
            "--counterfactual-policy-target-weight=0.25",
            "--counterfactual-policy-target-temperature=0.10",
            "--counterfactual-belief-samples=3",
            "--counterfactual-rollout-opponent-mode=policy",
            "--counterfactual-action-max-legal=3",
            "--counterfactual-action-label-prob=0.18",
            "--c-puct=1.85",
        ],
    },
    "champ_belief_fullmatch_policy_shaping_actual_fast": {
        "anchor_kl_weight": "0.98",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "0.98",
        "lr": "4.5e-7",
        "temperature": "0.98",
        "temperature_drop": "22",
        "self_play_belief_sims": "20",
        "train_steps": "100",
        "league_games": "128",
        "league_train_steps": "120",
        "league_opponent_mode": "belief",
        "league_checkpoint_count": 3,
        "league_bots": "",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.22",
            "--official-outcome-value-weight=0.72",
            "--paradox-loss-weight=0.03",
            "--action-value-loss-weight=0.18",
            "--action-value-selection-weight=0.0",
            "--counterfactual-action-value-rollouts=1",
            "--counterfactual-action-value-advantage",
            "--counterfactual-full-match-rollout",
            "--counterfactual-policy-target-weight=0.25",
            "--counterfactual-policy-target-temperature=0.10",
            "--counterfactual-belief-source=actual",
            "--counterfactual-belief-samples=1",
            "--counterfactual-rollout-opponent-mode=policy",
            "--counterfactual-action-max-legal=4",
            "--counterfactual-action-label-prob=0.22",
            "--c-puct=1.85",
        ],
    },
    "champ_belief_fullmatch_policy_shaping_confident_fast": {
        "anchor_kl_weight": "0.98",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "0.98",
        "lr": "4.5e-7",
        "temperature": "0.98",
        "temperature_drop": "22",
        "self_play_belief_sims": "20",
        "train_steps": "100",
        "league_games": "128",
        "league_train_steps": "120",
        "league_opponent_mode": "belief",
        "league_checkpoint_count": 3,
        "league_bots": "",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.22",
            "--official-outcome-value-weight=0.72",
            "--paradox-loss-weight=0.03",
            "--action-value-loss-weight=0.18",
            "--action-value-selection-weight=0.0",
            "--counterfactual-action-value-rollouts=2",
            "--counterfactual-action-value-advantage",
            "--counterfactual-full-match-rollout",
            "--counterfactual-policy-target-weight=0.25",
            "--counterfactual-policy-target-temperature=0.10",
            "--counterfactual-policy-target-min-actions=3",
            "--counterfactual-policy-target-min-spread=0.04",
            "--counterfactual-belief-source=infostate",
            "--counterfactual-belief-samples=3",
            "--counterfactual-rollout-opponent-mode=policy",
            "--counterfactual-action-max-legal=4",
            "--counterfactual-action-label-prob=0.20",
            "--c-puct=1.85",
        ],
    },
    "champ_action_mlp_heuristic_bootstrap_belief_fast": {
        "anchor_kl_weight": "0.0",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "0.98",
        "lr": "1.2e-6",
        "temperature": "1.00",
        "temperature_drop": "22",
        "self_play_belief_sims": "20",
        "train_steps": "120",
        "league_games": "128",
        "league_train_steps": "120",
        "league_opponent_mode": "belief",
        "league_checkpoint_count": 3,
        "league_bots": "",
        "extra_args": [
            "--resume-architecture=cli",
            "--arch=action_mlp",
            "--width=256",
            "--depth=3",
            "--bootstrap-games=192",
            "--bootstrap-train-steps=220",
            "--bootstrap-bots=heuristic_target2,heuristic,heuristic_adj2",
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.22",
            "--official-outcome-value-weight=0.72",
            "--paradox-loss-weight=0.03",
            "--action-value-loss-weight=0.12",
            "--action-value-selection-weight=0.0",
            "--counterfactual-action-value-rollouts=1",
            "--counterfactual-action-value-advantage",
            "--counterfactual-full-match-rollout",
            "--counterfactual-belief-source=infostate",
            "--counterfactual-belief-samples=2",
            "--counterfactual-rollout-opponent-mode=policy",
            "--counterfactual-action-max-legal=4",
            "--counterfactual-action-label-prob=0.12",
            "--c-puct=1.90",
        ],
    },
    "champ_action_mlp_interaction_bootstrap_first_fast": {
        "anchor_kl_weight": "0.0",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "0.92",
        "lr": "3e-5",
        "temperature": "0.90",
        "temperature_drop": "18",
        "self_play_belief_sims": "16",
        "train_steps": "80",
        "extra_args": [
            "--resume-architecture=cli",
            "--arch=action_mlp",
            "--width=256",
            "--depth=3",
            "--bootstrap-games=384",
            "--bootstrap-train-steps=700",
            "--bootstrap-bots=heuristic_target2,heuristic,heuristic_adj2",
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.18",
            "--official-outcome-value-weight=0.70",
            "--paradox-loss-weight=0.015",
            "--action-value-loss-weight=0.04",
            "--action-value-selection-weight=0.0",
            "--counterfactual-action-value-rollouts=1",
            "--counterfactual-action-value-advantage",
            "--counterfactual-full-match-rollout",
            "--counterfactual-belief-source=infostate",
            "--counterfactual-belief-samples=1",
            "--counterfactual-rollout-opponent-mode=policy",
            "--counterfactual-action-max-legal=3",
            "--counterfactual-action-label-prob=0.06",
            "--c-puct=1.75",
        ],
    },
    "champ_mixed_field_belief_action_safety_fast": {
        "anchor_kl_weight": "1.00",
        "terminal_paradox_penalty": "1.14",
        "value_loss_weight": "0.88",
        "lr": "6.5e-7",
        "temperature": "1.04",
        "temperature_drop": "24",
        "self_play_belief_sims": "24",
        "train_steps": "120",
        "league_games": "96",
        "league_train_steps": "100",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.28",
            "--official-outcome-value-weight=0.58",
            "--paradox-loss-weight=0.07",
            "--action-paradox-loss-weight=0.16",
            "--action-paradox-selection-penalty=0.12",
            "--action-paradox-root-only",
            "--counterfactual-action-rollouts=1",
            "--counterfactual-belief-samples=3",
            "--counterfactual-rollout-opponent-mode=policy",
            "--counterfactual-action-max-legal=4",
            "--counterfactual-action-label-prob=0.24",
        ],
    },
    "champ_residual_mixed_field_balanced": {
        "anchor_kl_weight": "0.82",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "1.00",
        "lr": "8e-7",
        "temperature": "1.05",
        "temperature_drop": "26",
        "self_play_belief_sims": "28",
        "train_steps": "180",
        "league_games": "144",
        "league_train_steps": "150",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--resume-architecture=cli",
            "--arch=residual",
            "--width=384",
            "--depth=4",
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.26",
            "--official-outcome-value-weight=0.62",
            "--paradox-loss-weight=0.04",
            "--c-puct=2.05",
        ],
    },
    "champ_residual_mixed_field_official": {
        "anchor_kl_weight": "0.74",
        "terminal_paradox_penalty": "0.94",
        "value_loss_weight": "1.08",
        "lr": "9e-7",
        "temperature": "1.08",
        "temperature_drop": "28",
        "self_play_belief_sims": "30",
        "train_steps": "180",
        "league_games": "144",
        "league_train_steps": "150",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--resume-architecture=cli",
            "--arch=residual",
            "--width=384",
            "--depth=4",
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.22",
            "--official-outcome-value-weight=0.72",
            "--paradox-loss-weight=0.025",
            "--c-puct=2.15",
        ],
    },
    "champ_residual_mcts_teacher_balanced": {
        "anchor_kl_weight": "0.88",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "1.00",
        "lr": "7e-7",
        "temperature": "1.00",
        "temperature_drop": "24",
        "self_play_belief_sims": "24",
        "train_steps": "120",
        "league_games": "72",
        "league_train_steps": "80",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--resume-architecture=cli",
            "--arch=residual",
            "--width=384",
            "--depth=4",
            "--teacher-checkpoint=az_runs/best_3p_policy_checkpoint.pt",
            "--teacher-mode=mcts",
            "--teacher-sims=24",
            "--teacher-games=72",
            "--teacher-train-steps=160",
            "--teacher-temperature=0.85",
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.26",
            "--official-outcome-value-weight=0.62",
            "--paradox-loss-weight=0.04",
            "--c-puct=2.00",
        ],
    },
    "champ_mlp_mcts_teacher_balanced": {
        "anchor_kl_weight": "0.88",
        "terminal_paradox_penalty": "1.02",
        "value_loss_weight": "1.00",
        "lr": "6e-7",
        "temperature": "0.98",
        "temperature_drop": "24",
        "self_play_belief_sims": "24",
        "train_steps": "120",
        "league_games": "72",
        "league_train_steps": "80",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--teacher-checkpoint=az_runs/best_3p_policy_checkpoint.pt",
            "--teacher-mode=mcts",
            "--teacher-sims=16",
            "--teacher-games=96",
            "--teacher-train-steps=140",
            "--teacher-temperature=0.80",
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.26",
            "--official-outcome-value-weight=0.62",
            "--paradox-loss-weight=0.04",
            "--c-puct=2.00",
        ],
    },
    "champ_mlp_mcts_teacher_official": {
        "anchor_kl_weight": "0.76",
        "terminal_paradox_penalty": "0.96",
        "value_loss_weight": "1.08",
        "lr": "6.5e-7",
        "temperature": "1.00",
        "temperature_drop": "26",
        "self_play_belief_sims": "24",
        "train_steps": "120",
        "league_games": "72",
        "league_train_steps": "80",
        "league_opponent_mode": "mixed",
        "league_checkpoint_count": 3,
        "league_bots": "heuristic",
        "extra_args": [
            "--teacher-checkpoint=az_runs/best_3p_policy_checkpoint.pt",
            "--teacher-mode=mcts",
            "--teacher-sims=16",
            "--teacher-games=96",
            "--teacher-train-steps=140",
            "--teacher-temperature=0.80",
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.22",
            "--official-outcome-value-weight=0.72",
            "--paradox-loss-weight=0.025",
            "--c-puct=2.05",
        ],
    },
    "champ_mlp_mcts_teacher_mcts_league_official": {
        "anchor_kl_weight": "0.82",
        "terminal_paradox_penalty": "0.98",
        "value_loss_weight": "1.08",
        "lr": "6e-7",
        "temperature": "0.98",
        "temperature_drop": "24",
        "self_play_belief_sims": "24",
        "train_steps": "120",
        "league_games": "96",
        "league_train_steps": "100",
        "league_opponent_mode": "mcts",
        "league_checkpoint_count": 2,
        "league_bots": "",
        "extra_args": [
            "--teacher-checkpoint=az_runs/best_3p_policy_checkpoint.pt",
            "--teacher-mode=mcts",
            "--teacher-sims=16",
            "--teacher-games=96",
            "--teacher-train-steps=140",
            "--teacher-temperature=0.80",
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.22",
            "--official-outcome-value-weight=0.72",
            "--paradox-loss-weight=0.025",
            "--c-puct=2.05",
        ],
    },
    "run94_s90_refine_mcts_pressure_official": {
        "anchor_kl_weight": "0.94",
        "terminal_paradox_penalty": "0.98",
        "value_loss_weight": "1.10",
        "lr": "3.5e-7",
        "temperature": "0.92",
        "temperature_drop": "20",
        "self_play_belief_sims": "24",
        "train_steps": "100",
        "league_games": "144",
        "league_train_steps": "120",
        "league_opponent_mode": "mcts",
        "league_checkpoint_count": 2,
        "league_bots": "",
        "extra_args": [
            "--resume-checkpoint=az_runs/candidate_sweep_run94_mcts_teacher_mcts_league_20260602/champ_mlp_mcts_teacher_mcts_league_official/checkpoint_0001_step_0090.pt",
            "--anchor-checkpoint=az_runs/candidate_sweep_run94_mcts_teacher_mcts_league_20260602/champ_mlp_mcts_teacher_mcts_league_official/checkpoint_0001_step_0090.pt",
            "--league-checkpoint=az_runs/best_3p_policy_checkpoint.pt,az_runs/candidate_sweep_run94_mcts_teacher_mcts_league_20260602/champ_mlp_mcts_teacher_mcts_league_official/checkpoint_0001_step_0090.pt",
            "--teacher-checkpoint=az_runs/candidate_sweep_run94_mcts_teacher_mcts_league_20260602/champ_mlp_mcts_teacher_mcts_league_official/checkpoint_0001_step_0090.pt",
            "--teacher-mode=mcts",
            "--teacher-sims=16",
            "--teacher-games=72",
            "--teacher-train-steps=100",
            "--teacher-temperature=0.72",
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.20",
            "--official-outcome-value-weight=0.78",
            "--paradox-loss-weight=0.02",
            "--c-puct=1.95",
        ],
    },
    "run96_s80_refine_belief_pressure_official": {
        "anchor_kl_weight": "0.92",
        "terminal_paradox_penalty": "0.98",
        "value_loss_weight": "1.10",
        "lr": "3e-7",
        "temperature": "0.90",
        "temperature_drop": "18",
        "self_play_belief_sims": "8",
        "train_steps": "120",
        "league_games": "192",
        "league_train_steps": "160",
        "league_opponent_mode": "belief",
        "league_checkpoint_count": 2,
        "league_bots": "",
        "extra_args": [
            "--resume-checkpoint=az_runs/candidate_sweep_run96_run94_s90_refine_20260602/run94_s90_refine_mcts_pressure_official/checkpoint_0001_step_0080.pt",
            "--anchor-checkpoint=az_runs/candidate_sweep_run96_run94_s90_refine_20260602/run94_s90_refine_mcts_pressure_official/checkpoint_0001_step_0080.pt",
            "--league-checkpoint=az_runs/best_3p_policy_checkpoint.pt,az_runs/candidate_sweep_run96_run94_s90_refine_20260602/run94_s90_refine_mcts_pressure_official/checkpoint_0001_step_0080.pt",
            "--value-loss-mode=acting",
            "--ordinal-value-weight=0.18",
            "--official-outcome-value-weight=0.80",
            "--paradox-loss-weight=0.025",
            "--c-puct=1.90",
        ],
    },
    "action_value_cf": {
        "anchor_kl_weight": "0.88",
        "terminal_paradox_penalty": "1.00",
        "value_loss_weight": "0.90",
        "lr": "1.0e-6",
        "temperature": "1.05",
        "temperature_drop": "20",
        "self_play_belief_sims": "20",
        "train_steps": "80",
        "extra_args": [
            "--action-value-loss-weight=0.25",
            "--action-value-selection-weight=0.05",
            "--action-value-root-only",
            "--counterfactual-action-value-rollouts=1",
            "--counterfactual-action-value-advantage",
            "--counterfactual-action-max-legal=5",
            "--counterfactual-action-label-prob=0.35",
        ],
    },
    "action_safety_cf": {
        "anchor_kl_weight": "0.86",
        "terminal_paradox_penalty": "1.10",
        "value_loss_weight": "0.85",
        "lr": "1.0e-6",
        "temperature": "1.10",
        "temperature_drop": "22",
        "self_play_belief_sims": "20",
        "train_steps": "80",
        "extra_args": [
            "--paradox-loss-weight=0.08",
            "--action-paradox-loss-weight=0.22",
            "--action-paradox-selection-penalty=0.18",
            "--action-paradox-root-only",
            "--counterfactual-action-rollouts=1",
            "--counterfactual-action-max-legal=5",
            "--counterfactual-action-label-prob=0.35",
        ],
    },
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-out-dir", default="az_runs/candidate_sweep")
  parser.add_argument("--champion", default="az_runs/best_3p_policy_checkpoint.pt")
  parser.add_argument("--recipes", default="short_spike,score_focus,safety_diverse")
  parser.add_argument("--seed", type=int, default=20260655)
  parser.add_argument("--repeat-count", type=int, default=1)
  parser.add_argument("--games-per-iter", type=int, default=72)
  parser.add_argument("--sims", type=int, default=64)
  parser.add_argument("--self-play-belief-samples", type=int, default=0)
  parser.add_argument("--eval-games", type=int, default=120)
  parser.add_argument(
      "--eval-full-match",
      action="store_true",
      help="Use full-match official scoring for trainer snapshot evals.",
  )
  parser.add_argument("--snapshot-interval", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
  parser.add_argument("--self-play-workers", type=int, default=0)
  parser.add_argument("--worker-torch-threads", type=int, default=1)
  parser.add_argument("--auto-worker-min-games", type=int, default=32)
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--collect-only", action="store_true")
  parser.add_argument(
      "--gate-champion",
      action="store_true",
      help="Run direct full-match champion gates for the top sweep candidates.",
  )
  parser.add_argument("--gate-top-n", type=int, default=1)
  parser.add_argument("--gate-matches", type=int, default=300)
  parser.add_argument(
      "--gate-repeat-count",
      type=int,
      default=1,
      help=(
          "Run each champion gate this many times with independent seeds and "
          "aggregate the pairwise score-rate evidence."
      ),
  )
  parser.add_argument(
      "--gate-repeat-seed-stride",
      type=int,
      default=100000,
      help="Seed offset between repeated gates for the same candidate.",
  )
  parser.add_argument("--gate-opponent", default="heuristic")
  parser.add_argument(
      "--gate-champion-mode",
      choices=("policy", "mcts", "belief", "belief_policy"),
      default="policy",
      help="Inference mode for the champion in direct gates.",
  )
  parser.add_argument(
      "--gate-candidate-mode",
      choices=("policy", "mcts", "belief", "belief_policy"),
      default="policy",
      help="Inference mode for candidates in direct gates.",
  )
  parser.add_argument(
      "--gate-mcts-sims",
      type=int,
      default=8,
      help="MCTS simulations used when either gate bot runs in MCTS mode.",
  )
  parser.add_argument(
      "--gate-belief-samples",
      type=int,
      default=4,
      help="Belief samples used when either gate bot runs in belief mode.",
  )
  parser.add_argument(
      "--gate-belief-sims",
      type=int,
      default=8,
      help="Per-belief-state MCTS simulations for belief-mode gates.",
  )
  parser.add_argument("--gate-workers", type=int, default=0)
  parser.add_argument("--gate-worker-torch-threads", type=int, default=1)
  parser.add_argument("--gate-auto-worker-min-games", type=int, default=32)
  parser.add_argument(
      "--gate-one-per-recipe",
      action="store_true",
      help="Gate only the highest-ranked candidate from each recipe.",
  )
  parser.add_argument(
      "--gate-max-per-recipe",
      type=int,
      default=1,
      help=(
          "When --gate-one-per-recipe is set, allow this many candidates from "
          "each recipe before moving on."
      ),
  )
  parser.add_argument(
      "--gate-max-per-run",
      type=int,
      default=0,
      help=(
          "Optional cap on gated candidates from the same run/out_dir. A value "
          "of 0 disables this cap."
      ),
  )
  parser.add_argument(
      "--gate-include-best-paradox-per-recipe",
      action="store_true",
      help=(
          "Also gate the lowest-paradox screen candidate from each recipe when "
          "it is not already selected by Elo ranking."
      ),
  )
  return parser.parse_args()


def recipe_names(raw):
  names = [name.strip() for name in raw.split(",") if name.strip()]
  unknown = [name for name in names if name not in RECIPES]
  if unknown:
    raise ValueError(f"Unknown recipes: {unknown}; known={sorted(RECIPES)}")
  return names


def trainer_command(args, recipe_name, recipe, out_dir, seed):
  cmd = [
      sys.executable,
      "quantum_cat_alphazero_torch.py",
      f"--players={args.players}",
      f"--device={args.device}",
      "--match-context",
      "--full-match-training",
      "--random-start-player",
      "--discard-resume-metrics",
      f"--seed={seed}",
      f"--resume-checkpoint={args.champion}",
      f"--anchor-checkpoint={args.champion}",
      f"--anchor-kl-weight={recipe['anchor_kl_weight']}",
      f"--terminal-paradox-penalty={recipe['terminal_paradox_penalty']}",
      f"--value-loss-weight={recipe['value_loss_weight']}",
      "--iterations=1",
      f"--games-per-iter={args.games_per_iter}",
      f"--sims={args.sims}",
      f"--self-play-belief-samples={args.self_play_belief_samples}",
      f"--self-play-belief-sims={recipe['self_play_belief_sims']}",
      f"--self-play-workers={args.self_play_workers}",
      f"--worker-torch-threads={args.worker_torch_threads}",
      f"--auto-worker-min-games={args.auto_worker_min_games}",
      f"--temperature={recipe['temperature']}",
      f"--temperature-drop={recipe['temperature_drop']}",
      f"--train-steps={recipe['train_steps']}",
      f"--train-snapshot-interval={args.snapshot_interval}",
      f"--batch-size={args.batch_size}",
      f"--lr={recipe['lr']}",
      f"--eval-games={args.eval_games}",
      "--eval-rotate-start-player",
      f"--out-dir={out_dir}",
  ]
  if args.eval_full_match:
    cmd.append("--eval-full-match")
  if "league_games" in recipe:
    league_checkpoint = args.champion
    checkpoint_count = int(recipe.get("league_checkpoint_count", 1))
    if checkpoint_count > 1:
      league_checkpoint = ",".join([args.champion] * checkpoint_count)
    cmd.extend([
        f"--league-games={recipe['league_games']}",
        f"--league-checkpoint={league_checkpoint}",
        f"--league-train-steps={recipe.get('league_train_steps', 0)}",
        f"--league-opponent-mode={recipe.get('league_opponent_mode', 'policy')}",
    ])
    if recipe.get("league_bots") is not None:
      cmd.append(f"--league-bots={recipe.get('league_bots')}")
  cmd.extend(recipe.get("extra_args", []))
  return cmd


def rating_from_row(row):
  return (
      row.get("eval", {})
      .get("ratings", {})
      .get("az_policy")
  )


def avg_return_from_row(row):
  return (
      row.get("eval", {})
      .get("avg_returns", {})
      .get("az_policy")
  )


def paradox_from_row(row):
  return (
      row.get("eval", {})
      .get("paradoxes_per_match", {})
      .get("az_policy")
  )


def checkpoint_belongs_to_out_dir(checkpoint, out_dir):
  if not checkpoint:
    return False
  checkpoint_path = Path(checkpoint)
  try:
    checkpoint_path.relative_to(out_dir)
    return True
  except ValueError:
    return str(checkpoint_path).startswith(str(out_dir) + "/")


def row_has_local_snapshot(row, out_dir):
  for snapshot in row.get("train_snapshots", []):
    if checkpoint_belongs_to_out_dir(snapshot.get("checkpoint"), out_dir):
      return True
  return False


def collect_candidates(recipe_name, out_dir):
  metrics_path = out_dir / "metrics.json"
  if not metrics_path.exists():
    return []
  rows = json.loads(metrics_path.read_text())
  candidates = []
  for row in rows:
    checkpoint = row.get("checkpoint")
    if checkpoint:
      if not checkpoint_belongs_to_out_dir(checkpoint, out_dir):
        continue
    elif row_has_local_snapshot(row, out_dir):
      checkpoint = str(out_dir / "checkpoint_0001.pt")
    else:
      continue
    rating = rating_from_row(row)
    if rating is None:
      continue
    candidates.append({
        "recipe": recipe_name,
        "iteration": row.get("iteration"),
        "train_steps_completed": row.get("train_steps_completed"),
        "checkpoint": checkpoint,
        "az_policy": rating,
        "az_avg_return": avg_return_from_row(row),
        "az_paradoxes": paradox_from_row(row),
        "out_dir": str(out_dir),
    })
  return candidates


def safe_name(value):
  return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def gate_command(args, candidate, index, out_path, repeat_index=0):
  candidate_name = f"candidate_{index + 1}"
  champion_spec = args.champion
  candidate_spec = candidate["checkpoint"]
  uses_mcts = args.gate_champion_mode == "mcts" or args.gate_candidate_mode == "mcts"
  uses_belief = (
      args.gate_champion_mode == "belief"
      or args.gate_candidate_mode == "belief"
  )
  if args.gate_champion_mode != "policy":
    champion_spec = f"{champion_spec}:{args.gate_champion_mode}"
  if args.gate_candidate_mode != "policy":
    candidate_spec = f"{candidate_spec}:{args.gate_candidate_mode}"
  cmd = [
      sys.executable,
      "quantum_cat_full_match_elo.py",
      f"--players={args.players}",
      f"--matches={args.gate_matches}",
      "--device=cpu",
      "--schedule=permutations",
      "--bots",
      "champion",
      candidate_name,
      args.gate_opponent,
      "--neural-bot",
      f"champion={champion_spec}",
      "--neural-bot",
      f"{candidate_name}={candidate_spec}",
      f"--workers={args.gate_workers}",
      f"--worker-torch-threads={args.gate_worker_torch_threads}",
      f"--auto-worker-min-games={args.gate_auto_worker_min_games}",
      f"--seed={args.seed + 5000 + index + repeat_index * args.gate_repeat_seed_stride}",
      f"--out={out_path}",
  ]
  if uses_mcts:
    cmd.append(f"--mcts-sims={args.gate_mcts_sims}")
  if uses_belief:
    cmd.extend([
        f"--belief-samples={args.gate_belief_samples}",
        f"--belief-sims={args.gate_belief_sims}",
    ])
  return cmd


def gate_summary(
    result,
    candidate_name="candidate_1",
    champion_mode="policy",
    candidate_mode="policy",
    mcts_sims=8,
    belief_samples=4,
    belief_sims=8,
):
  pairwise = result.get("pairwise", {})
  direct = pairwise.get(candidate_name, {}).get("champion")
  reverse = pairwise.get("champion", {}).get(candidate_name)
  stats = result.get("stats", {})
  return {
      "champion_mode": champion_mode,
      "candidate_mode": candidate_mode,
      "mcts_sims": mcts_sims if champion_mode == "mcts" or candidate_mode == "mcts" else None,
      "belief_samples": (
          belief_samples
          if champion_mode == "belief" or candidate_mode == "belief"
          else None
      ),
      "belief_sims": (
          belief_sims
          if champion_mode == "belief" or candidate_mode == "belief"
          else None
      ),
      "candidate_over_champion": direct,
      "champion_over_candidate": reverse,
      "online_ratings": result.get("online_ratings", result.get("ratings")),
      "pairwise_elo_ratings": result.get("pairwise_elo_ratings"),
      "candidate_paradoxes_per_match": (
          stats.get(candidate_name, {}).get("paradoxes_per_match")
      ),
      "champion_paradoxes_per_match": (
          stats.get("champion", {}).get("paradoxes_per_match")
      ),
  }


def score_rate_ci95(score, matches):
  if matches <= 0:
    return None
  rate = score / matches
  half_width = 1.96 * (rate * (1.0 - rate) / matches) ** 0.5
  return [round(max(0.0, rate - half_width), 4), round(min(1.0, rate + half_width), 4)]


def elo_from_score_rate(rate):
  rate = min(0.999999, max(0.000001, rate))
  return round(400.0 * math.log10(rate / (1.0 - rate)), 2)


def aggregate_gate_summaries(summaries):
  summaries = [summary for summary in summaries if summary]
  if not summaries:
    return None
  direct_rows = [
      summary.get("candidate_over_champion")
      for summary in summaries
      if summary.get("candidate_over_champion")
  ]
  reverse_rows = [
      summary.get("champion_over_candidate")
      for summary in summaries
      if summary.get("champion_over_candidate")
  ]
  if not direct_rows:
    return summaries[-1]

  def combine(rows):
    matches = sum(int(row.get("matches", 0)) for row in rows)
    wins = sum(int(row.get("wins", 0)) for row in rows)
    losses = sum(int(row.get("losses", 0)) for row in rows)
    ties = sum(int(row.get("ties", 0)) for row in rows)
    score = wins + 0.5 * ties
    score_rate = score / matches if matches else 0.0
    avg_score_diff = (
        sum(float(row.get("avg_score_diff", 0.0)) * int(row.get("matches", 0))
            for row in rows)
        / matches
        if matches else 0.0
    )
    return {
        "matches": matches,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(wins / matches, 4) if matches else 0.0,
        "non_loss_rate": round((wins + ties) / matches, 4) if matches else 0.0,
        "score_rate": round(score_rate, 4),
        "score_rate_ci95": score_rate_ci95(score, matches),
        "head_to_head_elo_diff": elo_from_score_rate(score_rate) if matches else 0.0,
        "avg_score_diff": round(avg_score_diff, 4),
        "avg_score_diff_ci95": None,
    }

  candidate_paradoxes = [
      summary.get("candidate_paradoxes_per_match")
      for summary in summaries
      if summary.get("candidate_paradoxes_per_match") is not None
  ]
  champion_paradoxes = [
      summary.get("champion_paradoxes_per_match")
      for summary in summaries
      if summary.get("champion_paradoxes_per_match") is not None
  ]
  first = summaries[0]
  return {
      "repeat_count": len(summaries),
      "champion_mode": first.get("champion_mode"),
      "candidate_mode": first.get("candidate_mode"),
      "mcts_sims": first.get("mcts_sims"),
      "belief_samples": first.get("belief_samples"),
      "belief_sims": first.get("belief_sims"),
      "candidate_over_champion": combine(direct_rows),
      "champion_over_candidate": combine(reverse_rows) if reverse_rows else None,
      "candidate_paradoxes_per_match": (
          round(float(np.mean(candidate_paradoxes)), 4)
          if candidate_paradoxes else None
      ),
      "champion_paradoxes_per_match": (
          round(float(np.mean(champion_paradoxes)), 4)
          if champion_paradoxes else None
      ),
      "repeats": summaries,
  }


def gate_candidates(args, candidates):
  if not getattr(args, "gate_one_per_recipe", False):
    recipe_filtered = candidates[:]
  else:
    recipe_filtered = []
    seen_counts = {}
    max_per_recipe = max(1, int(getattr(args, "gate_max_per_recipe", 1)))
    for candidate in candidates:
      recipe_key = candidate.get("recipe_name") or candidate.get("recipe")
      seen_count = seen_counts.get(recipe_key, 0)
      if seen_count >= max_per_recipe:
        continue
      seen_counts[recipe_key] = seen_count + 1
      recipe_filtered.append(candidate)
      if (
          int(getattr(args, "gate_max_per_run", 0)) <= 0
          and len(recipe_filtered) >= args.gate_top_n
      ):
        break

  max_per_run = int(getattr(args, "gate_max_per_run", 0))
  if max_per_run <= 0:
    selected = recipe_filtered[:args.gate_top_n]
    return add_best_paradox_candidates(args, candidates, selected)
  selected = []
  run_counts = {}
  for candidate in recipe_filtered:
    run_key = candidate.get("out_dir") or candidate.get("recipe")
    run_count = run_counts.get(run_key, 0)
    if run_count >= max_per_run:
      continue
    run_counts[run_key] = run_count + 1
    selected.append(candidate)
    if len(selected) >= args.gate_top_n:
      break
  return add_best_paradox_candidates(args, candidates, selected)


def candidate_identity(candidate):
  return (
      candidate.get("checkpoint"),
      candidate.get("recipe_name") or candidate.get("recipe"),
      candidate.get("train_steps_completed"),
      candidate.get("iteration"),
  )


def add_best_paradox_candidates(args, candidates, selected):
  if not getattr(args, "gate_include_best_paradox_per_recipe", False):
    return selected
  selected = list(selected)
  selected_keys = {candidate_identity(candidate) for candidate in selected}
  by_recipe = {}
  for candidate in candidates:
    if candidate.get("az_paradoxes") is None:
      continue
    recipe_key = candidate.get("recipe_name") or candidate.get("recipe")
    by_recipe.setdefault(recipe_key, []).append(candidate)
  paradox_picks = []
  for recipe_key in sorted(by_recipe):
    paradox_picks.append(
        min(
            by_recipe[recipe_key],
            key=lambda item: (
                item.get("az_paradoxes", float("inf")),
                -(item.get("az_policy") or -float("inf")),
            ),
        )
    )
  for candidate in paradox_picks:
    if len(selected) >= args.gate_top_n:
      break
    key = candidate_identity(candidate)
    if key in selected_keys:
      continue
    selected.append(candidate)
    selected_keys.add(key)
  return selected


def tactical_preflight_command(args, candidate, out_path):
  candidate_spec = (
      f"{candidate['checkpoint']}:{args.gate_candidate_mode}"
  )
  return [
      sys.executable,
      "quantum_cat_curated_validation.py",
      "--players", str(args.players),
      "--candidate", candidate_spec,
      "--belief-samples", str(args.gate_belief_samples),
      "--belief-sims", str(args.gate_belief_sims),
      "--mcts-sims", str(args.gate_mcts_sims),
      "--out", str(out_path),
  ]


def run_candidate_tactical_preflight(args, candidate, out_path):
  suite_path = (
      Path(__file__).resolve().parent
      / "experiments/quantum_cat_curated_tactical_validation.json"
  )
  payload = json.loads(suite_path.read_text(encoding="utf-8"))
  matching = [
      scenario for scenario in payload.get("scenarios", [])
      if int(scenario.get("state", {}).get("numPlayers", -1))
      == int(args.players)
  ]
  if not matching:
    return {
        "available": False,
        "players": int(args.players),
        "scenario_family_count": 0,
        "passed": None,
    }

  command = tactical_preflight_command(args, candidate, out_path)
  result = {
      "available": True,
      "players": int(args.players),
      "scenario_family_count": len(matching),
      "candidate": f"{candidate['checkpoint']}:{args.gate_candidate_mode}",
      "command": command,
      "out": str(out_path),
      "passed": None,
  }
  if args.dry_run:
    return result

  completed = subprocess.run(
      command, check=False, capture_output=True, text=True
  )
  if completed.stdout:
    print(completed.stdout, flush=True)
  if completed.stderr:
    print(completed.stderr, file=sys.stderr, flush=True)
  if completed.returncode not in (0, 1) or not out_path.is_file():
    raise RuntimeError(
        f"Candidate tactical preflight failed to run: exit "
        f"{completed.returncode}, artifact={out_path}"
    )
  report = json.loads(out_path.read_text(encoding="utf-8"))
  if report.get("candidate") != result["candidate"]:
    raise ValueError("Candidate tactical report does not match gated selector")
  result.update({
      "scenario_count": int(report.get("scenarioCount") or 0),
      "passed_count": int(report.get("passedCount") or 0),
      "passed": bool(report.get("passed")),
  })
  if bool(completed.returncode == 0) != bool(result["passed"]):
    raise ValueError("Candidate tactical report contradicts its exit status")
  return result


def run_candidate_bidding_preflight(args, candidate, out_path):
  suite_path = (
      Path(__file__).resolve().parent
      / "experiments/quantum_cat_curated_bidding_validation.json"
  )
  payload = json.loads(suite_path.read_text(encoding="utf-8"))
  matching = [
      scenario for scenario in payload.get("scenarios", [])
      if int(scenario.get("state", {}).get("numPlayers", -1))
      == int(args.players)
  ]
  if not matching:
    raise ValueError("Every supported player count requires locked bidding hands")
  candidate_spec = f"{candidate['checkpoint']}:{args.gate_candidate_mode}"
  command = [
      sys.executable,
      "quantum_cat_curated_bidding_validation.py",
      "--players", str(args.players),
      "--candidate", candidate_spec,
      "--quiet",
      "--out", str(out_path),
  ]
  result = {
      "available": True,
      "players": int(args.players),
      "scenario_family_count": len(matching),
      "candidate": candidate_spec,
      "command": command,
      "out": str(out_path),
      "passed": None,
  }
  if args.dry_run:
    return result
  completed = subprocess.run(
      command, check=False, capture_output=True, text=True
  )
  if completed.stdout:
    print(completed.stdout, flush=True)
  if completed.stderr:
    print(completed.stderr, file=sys.stderr, flush=True)
  if completed.returncode not in (0, 1) or not out_path.is_file():
    raise RuntimeError(
        f"Candidate bidding preflight failed to run: exit "
        f"{completed.returncode}, artifact={out_path}"
    )
  report = json.loads(out_path.read_text(encoding="utf-8"))
  if (
      report.get("candidate") != candidate_spec
      or report.get("suiteID") != "quantum-cat-curated-bidding-validation-v1"
      or report.get("mode") != "candidate_bid_validation"
  ):
    raise ValueError("Candidate bidding report does not match gated selector")
  result.update({
      "scenario_count": int(report.get("scenarioCount") or 0),
      "passed_count": int(report.get("passedCount") or 0),
      "passed": bool(report.get("passed")),
  })
  if bool(completed.returncode == 0) != bool(result["passed"]):
    raise ValueError("Candidate bidding report contradicts its exit status")
  return result


def run_candidate_bid_regime_preflight(args, candidate, out_path):
  suite_path = (
      Path(__file__).resolve().parent
      / "experiments/quantum_cat_curated_bid_regime_validation.json"
  )
  payload = json.loads(suite_path.read_text(encoding="utf-8"))
  matching = [
      scenario for scenario in payload.get("scenarios", [])
      if int(scenario.get("state", {}).get("numPlayers", -1))
      == int(args.players)
  ]
  if not matching:
    raise ValueError("Every supported player count requires locked bid-regime moves")
  candidate_spec = f"{candidate['checkpoint']}:{args.gate_candidate_mode}"
  command = tactical_preflight_command(args, candidate, out_path)
  command[2:2] = ["--suite", str(suite_path)]
  result = {
      "available": True,
      "players": int(args.players),
      "scenario_family_count": len(matching),
      "candidate": candidate_spec,
      "command": command,
      "out": str(out_path),
      "passed": None,
  }
  if args.dry_run:
    return result
  completed = subprocess.run(
      command, check=False, capture_output=True, text=True
  )
  if completed.stdout:
    print(completed.stdout, flush=True)
  if completed.stderr:
    print(completed.stderr, file=sys.stderr, flush=True)
  if completed.returncode not in (0, 1) or not out_path.is_file():
    raise RuntimeError(
        f"Candidate bid-regime preflight failed to run: exit "
        f"{completed.returncode}, artifact={out_path}"
    )
  report = json.loads(out_path.read_text(encoding="utf-8"))
  if (
      report.get("candidate") != candidate_spec
      or report.get("suiteID") != "quantum-cat-curated-bid-regime-validation-v1"
      or any(
          int(row.get("players") or 0) != int(args.players)
          for row in report.get("results", [])
      )
  ):
    raise ValueError("Candidate bid-regime report does not match gated selector")
  result.update({
      "scenario_count": int(report.get("scenarioCount") or 0),
      "passed_count": int(report.get("passedCount") or 0),
      "passed": bool(report.get("passed")),
  })
  if bool(completed.returncode == 0) != bool(result["passed"]):
    raise ValueError("Candidate bid-regime report contradicts its exit status")
  return result


def run_champion_gates(args, base_out, candidates):
  if not args.gate_champion or args.gate_top_n <= 0:
    return []
  gates_dir = base_out / "champion_gates"
  gates_dir.mkdir(parents=True, exist_ok=True)
  gated = []
  for index, candidate in enumerate(gate_candidates(args, candidates)):
    mode_suffix = ""
    if args.gate_champion_mode != "policy" or args.gate_candidate_mode != "policy":
      mode_suffix = (
          f"_{args.gate_candidate_mode}_vs_{args.gate_champion_mode}"
      )
      if args.gate_champion_mode == "mcts" or args.gate_candidate_mode == "mcts":
        mode_suffix += f"_sims{args.gate_mcts_sims}"
      if args.gate_champion_mode == "belief" or args.gate_candidate_mode == "belief":
        mode_suffix += (
            f"_belief{args.gate_belief_samples}x{args.gate_belief_sims}"
        )
    slug = safe_name(
        f"{index + 1:02d}_{candidate['recipe']}_{candidate.get('train_steps_completed')}"
        f"{mode_suffix}"
    )
    repeat_count = max(1, int(getattr(args, "gate_repeat_count", 1)))
    out_path = gates_dir / f"{slug}.json"
    if repeat_count > 1:
      out_path = gates_dir / f"{slug}_aggregate.json"
    gate_row = {
        "candidate": candidate,
        "repeat_count": repeat_count,
        "out": str(out_path),
        "repeats": [],
    }
    tactical_out = gates_dir / f"{slug}_tactical.json"
    tactical = run_candidate_tactical_preflight(
        args, candidate, tactical_out
    )
    gate_row["curated_tactical"] = tactical
    candidate["curated_tactical"] = tactical
    if tactical["available"] and tactical["passed"] is False:
      gate_row["skipped"] = "locked_tactical_failure"
      candidate["champion_gate"] = {
          "passed": False,
          "reason": "locked_tactical_failure",
          "curated_tactical": tactical,
      }
      gated.append(gate_row)
      print(json.dumps({
          "event": "champion_gate_rejected_tactical",
          "rank": index + 1,
          "candidate": candidate,
          "tactical": tactical,
      }, indent=2), flush=True)
      continue
    bidding_out = gates_dir / f"{slug}_bidding.json"
    bidding = run_candidate_bidding_preflight(args, candidate, bidding_out)
    gate_row["curated_bidding"] = bidding
    candidate["curated_bidding"] = bidding
    if bidding["passed"] is False:
      gate_row["skipped"] = "locked_bidding_failure"
      candidate["champion_gate"] = {
          "passed": False,
          "reason": "locked_bidding_failure",
          "curated_bidding": bidding,
      }
      gated.append(gate_row)
      print(json.dumps({
          "event": "champion_gate_rejected_bidding",
          "rank": index + 1,
          "candidate": candidate,
          "bidding": bidding,
      }, indent=2), flush=True)
      continue
    bid_regime_out = gates_dir / f"{slug}_bid_regime.json"
    bid_regime = run_candidate_bid_regime_preflight(
        args, candidate, bid_regime_out
    )
    gate_row["curated_bid_regime"] = bid_regime
    candidate["curated_bid_regime"] = bid_regime
    if bid_regime["passed"] is False:
      gate_row["skipped"] = "locked_bid_regime_failure"
      candidate["champion_gate"] = {
          "passed": False,
          "reason": "locked_bid_regime_failure",
          "curated_bid_regime": bid_regime,
      }
      gated.append(gate_row)
      print(json.dumps({
          "event": "champion_gate_rejected_bid_regime",
          "rank": index + 1,
          "candidate": candidate,
          "bid_regime": bid_regime,
      }, indent=2), flush=True)
      continue
    print(json.dumps({
        "event": "start_champion_gate",
        "rank": index + 1,
        "candidate": candidate,
        "repeat_count": repeat_count,
    }, indent=2), flush=True)
    repeat_summaries = []
    for repeat_index in range(repeat_count):
      repeat_out_path = out_path
      if repeat_count > 1:
        repeat_out_path = gates_dir / f"{slug}_r{repeat_index + 1:02d}.json"
      cmd = gate_command(args, candidate, index, repeat_out_path, repeat_index)
      repeat_row = {
          "repeat": repeat_index + 1,
          "command": cmd,
          "out": str(repeat_out_path),
      }
      gate_row["repeats"].append(repeat_row)
      print(json.dumps({
          "event": "start_champion_gate_repeat",
          "rank": index + 1,
          "repeat": repeat_index + 1,
          "cmd": cmd,
      }, indent=2), flush=True)
      if not args.dry_run:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if completed.stdout:
          print(completed.stdout, flush=True)
        result = json.loads(repeat_out_path.read_text())
        repeat_summary = gate_summary(
            result,
            f"candidate_{index + 1}",
            args.gate_champion_mode,
            args.gate_candidate_mode,
            args.gate_mcts_sims,
            args.gate_belief_samples,
            args.gate_belief_sims,
        )
        repeat_row["result"] = repeat_summary
        repeat_summaries.append(repeat_summary)
      print(json.dumps({
          "event": "champion_gate_repeat_done",
          "rank": index + 1,
          "repeat": repeat_index + 1,
          "summary": repeat_row.get("result"),
      }, indent=2), flush=True)
    if not args.dry_run:
      gate_row["result"] = aggregate_gate_summaries(repeat_summaries)
      candidate["champion_gate"] = gate_row["result"]
      if repeat_count > 1:
        out_path.write_text(json.dumps(gate_row["result"], indent=2) + "\n")
    gated.append(gate_row)
    print(json.dumps({
        "event": "champion_gate_done",
        "rank": index + 1,
        "summary": gate_row.get("result"),
    }, indent=2), flush=True)
  return gated


def main():
  args = parse_args()
  if args.repeat_count < 1:
    raise ValueError("--repeat-count must be >= 1")
  base_out = Path(args.base_out_dir)
  base_out.mkdir(parents=True, exist_ok=True)
  names = recipe_names(args.recipes)
  all_candidates = []
  runs = []
  run_index = 0
  for repeat in range(args.repeat_count):
    for name in names:
      suffix = name if args.repeat_count == 1 else f"{name}_r{repeat + 1:02d}"
      out_dir = base_out / suffix
      recipe = RECIPES[name]
      seed = args.seed + run_index
      run_index += 1
      run_label = suffix
      cmd = trainer_command(args, name, recipe, out_dir, seed)
      runs.append({
          "recipe": name,
          "run_label": run_label,
          "repeat": repeat + 1,
          "seed": seed,
          "out_dir": str(out_dir),
          "cmd": cmd,
      })
      print(json.dumps({
          "event": "start_recipe",
          "recipe": name,
          "run_label": run_label,
          "repeat": repeat + 1,
          "seed": seed,
          "out_dir": str(out_dir),
          "cmd": cmd,
      }, indent=2), flush=True)
      if not args.dry_run and not args.collect_only:
        subprocess.run(cmd, check=True)
      candidates = collect_candidates(run_label, out_dir)
      for candidate in candidates:
        candidate["recipe_name"] = name
        candidate["repeat"] = repeat + 1
        candidate["seed"] = seed
      all_candidates.extend(candidates)
      best = max(candidates, key=lambda item: item["az_policy"], default=None)
      print(json.dumps({"event": "recipe_done", "recipe": name,
                        "run_label": run_label, "best": best},
                       indent=2), flush=True)
  all_candidates.sort(key=lambda item: item["az_policy"], reverse=True)
  champion_gates = run_champion_gates(args, base_out, all_candidates)
  summary = {
      "runs": runs,
      "candidates": all_candidates,
      "champion_gates": champion_gates,
  }
  summary_path = base_out / "sweep_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps({"event": "sweep_done", "summary_path": str(summary_path),
                    "best": all_candidates[:5]}, indent=2), flush=True)


if __name__ == "__main__":
  main()
