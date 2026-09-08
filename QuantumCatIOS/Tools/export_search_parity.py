#!/usr/bin/env python3
"""Independent, actor-view fixtures for the native Run1052 implementation.

Does not modify the frozen Python engine, model, selector, or confirmation files.
The full-world rows are synthetic sampler test expectations, never live inputs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pyspiel
import torch
import quantum_cat_full_match_elo as native
from export_move_parity_fixtures import live_state_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--matches', type=int, default=2)
    options = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    confirmed = json.loads((ROOT / 'research/run1052/CONFIRMED.json').read_text())
    selector = str(ROOT / 'research/run1052/champion.pt') + ':' + confirmed['candidate'].split(':', 1)[1]
    entry = native.load_neural_bots(['champion=' + selector], 5, torch.device('cpu'))['champion']
    base = native.AZCompetitiveTacticsBot(entry['model'], 'base', torch.device('cpu'), copy.deepcopy(entry['args']))
    root = native.AZRootRolloutBot(entry['model'], 'root', torch.device('cpu'), copy.deepcopy(entry['args']))
    rows, searches, worlds, deals = [], [], [], []
    buckets = set()
    started = time.monotonic()
    rng = np.random.RandomState(8127)
    random_words = [int(rng.randint(0, 2**32, dtype=np.uint32)) for _ in range(1300)]
    for rnd in range(5):
        rng = np.random.RandomState(917 + rnd)
        state = pyspiel.load_game('python_quantum_cat', {'players': 5, 'start_player': rnd, 'match_context': 1}).new_initial_state()
        state.set_match_context([1, 2, 3, 4, 5], rnd)
        while state.is_chance_node():
            actions, probabilities = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(actions, p=probabilities)))
        deals.append({'seed': 917 + rnd, 'round': rnd, 'hands': [list(map(int, hand)) for hand in state._hands],
                      'observation': list(state.observation_tensor(state.current_player())), 'state': live_state_snapshot(state, state.current_player())})
    for match in range(options.matches):
        totals = np.zeros(5, dtype=np.float32)
        for rnd in range(5):
            rng = np.random.RandomState(2026090807 + match * 100003 + rnd * 9973)
            state = pyspiel.load_game('python_quantum_cat', {'players': 5, 'start_player': (match + rnd) % 5, 'match_context': 1}).new_initial_state()
            state.set_match_context(totals, rnd)
            while not state.is_terminal():
                if state.is_chance_node():
                    actions, probabilities = zip(*state.chance_outcomes())
                    state.apply_action(int(rng.choice(actions, p=probabilities)))
                    continue
                player = state.current_player()
                legal = list(map(int, state.legal_actions(player)))
                action = int(base.step(state.clone(), player))
                rows.append({'state': live_state_snapshot(state, player), 'legal': legal, 'action': action,
                             'observation': list(state.observation_tensor(player))})
                if root._enabled_for_phase(state) and len(legal) > 1:
                    policy, _ = native.model_policy_value(root.model, state, player, 1000, root.args.value_scale, root.device, use_shared_prediction=False)
                    baseline = root._base_action(state, player, legal, policy)
                    protected = root._preserves_tactical_override()
                    seed = root._reset_search_rng(state, player, legal)
                    candidates = root._candidate_actions(state, player, legal, policy, baseline_action=baseline)
                    safe = root._safe_candidate_actions(state, player, candidates, baseline)
                    if len(worlds) < 25:
                        sample_rng = np.random.RandomState(seed)
                        world = state.resample_from_infostate(player, sample_rng)
                        worlds.append({'state': live_state_snapshot(state, player), 'seed': seed, 'legal': legal,
                                       'hands': [list(map(int, hand)) for hand in world._hands],
                                       'discards': list(map(int, world._discarded_cards)), 'safe': safe, 'candidates': candidates, 'baseline': baseline})
                    bucket = (rnd, 'multiple' if len(safe) > 1 else 'single', protected)
                    if bucket not in buckets and not protected:
                        buckets.add(bucket)
                        scores = []
                        original = root._score_actions
                        def capture(states, actor, actions, rollouts):
                            result = original(states, actor, actions, rollouts)
                            scores.append({str(k): [v['paradox'], v['score']] for k, v in result.items()})
                            return result
                        root._score_actions = capture
                        selected = int(root.step(state.clone(), player))
                        root._score_actions = original
                        searches.append({'state': live_state_snapshot(state, player), 'legal': legal,
                                         'seed': seed, 'baseline': int(baseline), 'candidates': safe,
                                         'action': selected, 'searchScores': scores[0] if scores else {},
                                         'auditScores': scores[1] if len(scores) > 1 else {}})
                        print(f'root round={rnd} safe={safe} base={baseline} selected={selected} elapsed={time.monotonic()-started:.1f}', flush=True)
                state.apply_action(action)
            totals = np.asarray(state.returns(), dtype=np.float32)
    result = {'checkpointSHA256': confirmed['checkpoint_sha256'], 'selector': selector,
              'randomSeed': 8127, 'randomWords': random_words, 'deals': deals, 'worlds': worlds,
              'continuations': rows, 'searches': searches, 'elapsedSeconds': time.monotonic() - started}
    out = Path(options.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, separators=(',', ':'), default=lambda value: value.item()) + '\n')
    print(json.dumps({'out': str(out), 'continuations': len(rows), 'searches': len(searches), 'seconds': result['elapsedSeconds']}), flush=True)


if __name__ == '__main__':
    main()
