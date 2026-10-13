"""Exact own-hand liveness oracle for Quantum Cat (Cat in the Box).

Computes, from ONLY the acting player's own hand plus public state (board
ownership, color tokens, trick count), whether the player can still guarantee
a legal play for every remaining trick, and how robust that guarantee is to
adversarial cell consumption by opponents.

Information hygiene: nothing here reads opponent hands. Inputs are the
player's own rank counts (their private knowledge), the public board, and the
public token grid.

Key structural fact: a card of rank r can only be played into board column r
(any color the player still holds a token for). The card->cell compatibility
graph therefore decomposes into independent per-rank components, so

    max_matching = sum_r min(count[r], cap[r])

where count[r] is the player's copies of rank r and cap[r] is the number of
open cells (c, r) over the player's live token colors c. No general matching
algorithm is needed.

Soundness guarantee: open cells are only ever consumed, tokens only ever
burned, and hands only shrink during the play phase. Therefore if
max_matching < plays_remaining at any point, the player is guaranteed to
paradox in every continuation (unless another player paradoxes first and ends
the round). The converse does not hold: "alive" is necessary, not sufficient,
because opponents will keep consuming cells and may force token burns by
leading colors the player cannot follow.

Known approximations (all conservative or empirically negligible):
  - Opponents are modeled as able to consume any open cell in the player's
    compatible set ("adversarial deletions"), ignoring their own rank/token
    legality constraints.
  - The red-lead restriction (cannot lead red before the red row is occupied
    unless red is the only option) is ignored. It never forces a paradox by
    itself - red leads are always allowed when red is the only remaining
    color - so it only adds choice restrictions in non-critical states.
  - Future forced token burns (opponents leading colors the player cannot
    follow) are not searched; `margin_without_token` exposes the sensitivity
    so callers can penalize voluntarily burning a critical token.
"""

import numpy as np

PARADOX_ACTION = 999
_NUM_COLORS = 4  # R, B, Y, G - matches _COLORS in quantum_cat.py


# ---------------------------------------------------------------------------
# Pure core: per-rank counts/caps arithmetic.
# ---------------------------------------------------------------------------


def max_matching(counts, caps):
  """Maximum number of the player's cards placeable on open compatible cells.

  Args:
    counts: sequence, counts[r] = player's copies of rank index r.
    caps: sequence, caps[r] = open cells in column r over live token colors.

  Returns:
    int matching size.
  """
  return int(sum(min(int(c), int(k)) for c, k in zip(counts, caps)))


def adversarial_margin(counts, caps, plays_remaining):
  """Max adversarial cell deletions the survival guarantee absorbs.

  The adversary deletes open compatible cells one at a time to minimize the
  matching. Deleting in column r only reduces the matching once
  cap[r] < count[r], so the optimal adversary first spends surplus
  (cap[r]-count[r])+ deletions "for free" in whichever column it attacks.
  The player survives K deletions iff the post-deletion matching still
  covers plays_remaining.

  Greedy exact computation: the player's slack S = matching - plays_remaining
  allows S whole units of matching loss. The adversary's cheapest way to
  cause m units of loss is to attack columns in increasing order of surplus,
  paying (surplus_r + units taken from r). We return the total number of
  deletions needed to cause S+1 units of loss, minus 1: the largest K that is
  guaranteed survivable.

  Returns:
    int margin >= -1. Negative means already dead (matching < plays needed).
    Large positive values are clipped only by total open compatible cells.
  """
  matching = max_matching(counts, caps)
  plays_remaining = int(plays_remaining)
  if matching < plays_remaining:
    return -1
  slack = matching - plays_remaining
  # Columns where the player actually uses cells: contribution min(count,cap).
  # Adversary attack cost per column r: surplus_r deletions before losses
  # start, then 1 deletion per unit of loss, up to min(count,cap) units.
  attacks = []  # (surplus, available_loss_units)
  for c, k in zip(counts, caps):
    c = int(c)
    k = int(k)
    used = min(c, k)
    if used <= 0:
      continue
    attacks.append((max(0, k - c), used))
  # Cheapest way to accumulate slack+1 units of loss: choose a SUBSET of
  # columns minimizing total surplus paid, subject to their combined loss
  # units covering the requirement; each loss unit then costs 1 deletion.
  # Small knapsack DP: dp[j] = min surplus to secure >= j loss units.
  needed = slack + 1
  infinity = float("inf")
  dp = [0.0] + [infinity] * needed
  for surplus, units in attacks:
    for j in range(needed, 0, -1):
      src = max(0, j - units)
      if dp[src] + surplus < dp[j]:
        dp[j] = dp[src] + surplus
  if dp[needed] == infinity:
    # Adversary cannot destroy enough matching even deleting everything.
    return int(sum(int(k) for k in caps))
  return int(dp[needed] + needed - 1)


def margin_without_token(counts, caps_by_color, tokens, plays_remaining,
                         color_idx):
  """Adversarial margin if token `color_idx` were lost right now."""
  if not tokens[color_idx]:
    caps = _caps_from_colors(caps_by_color, tokens)
    return adversarial_margin(counts, caps, plays_remaining)
  reduced = list(tokens)
  reduced[color_idx] = False
  caps = _caps_from_colors(caps_by_color, reduced)
  return adversarial_margin(counts, caps, plays_remaining)


def _caps_from_colors(caps_by_color, tokens):
  """caps[r] summed over live token colors. caps_by_color: [color][rank]."""
  num_ranks = len(caps_by_color[0])
  caps = [0] * num_ranks
  for c_idx, live in enumerate(tokens):
    if not live:
      continue
    row = caps_by_color[c_idx]
    for r in range(num_ranks):
      caps[r] += int(row[r])
  return caps


# ---------------------------------------------------------------------------
# OpenSpiel state adapter. Reads only own hand + public state.
# ---------------------------------------------------------------------------


class LivenessView(object):
  """Snapshot of one player's liveness-relevant state."""

  __slots__ = (
      "counts", "caps_by_color", "tokens", "plays_remaining", "led_color_idx",
      "num_ranks",
  )

  def __init__(self, counts, caps_by_color, tokens, plays_remaining,
               led_color_idx):
    self.counts = counts
    self.caps_by_color = caps_by_color
    self.tokens = tokens
    self.plays_remaining = plays_remaining
    self.led_color_idx = led_color_idx
    self.num_ranks = len(counts)

  def caps(self):
    return _caps_from_colors(self.caps_by_color, self.tokens)

  def matching(self):
    return max_matching(self.counts, self.caps())

  def alive(self):
    return self.matching() >= self.plays_remaining

  def margin(self):
    return adversarial_margin(self.counts, self.caps(), self.plays_remaining)


def view_from_state(state, player):
  """Build a LivenessView for `player` from an OpenSpiel quantum_cat state.

  Reads state._hands[player] (the player's own private hand - the same
  information their observation tensor contains) plus public members.
  """
  num_ranks = int(state._num_card_types)
  counts = [int(x) for x in state._hands[player]]
  board = state._board_ownership
  tokens = [bool(state._color_tokens[player][c]) for c in range(_NUM_COLORS)]
  caps_by_color = []
  for c in range(_NUM_COLORS):
    caps_by_color.append(
        [1 if int(board[c, r]) == -1 else 0 for r in range(num_ranks)])
  plays = int(state._num_tricks) - int(state._trick_number)
  if state._cards_played_this_trick[player] is not None:
    plays -= 1
  led = state._led_color
  led_idx = None
  if led is not None:
    led_idx = ["R", "B", "Y", "G"].index(led)
  return LivenessView(counts, caps_by_color, tokens, plays, led_idx)


def post_play_margin(view, action):
  """Adversarial margin after the player makes play `action` (c*T + r).

  Models exactly the state effects of the player's own play:
    - rank leaves the hand;
    - the cell (c, r) closes;
    - if there is a led color and the play is off-led, the led token burns;
    - plays_remaining decreases by one.
  """
  if action == PARADOX_ACTION:
    return -1
  num_ranks = view.num_ranks
  c_idx = action // num_ranks
  r_idx = action % num_ranks
  counts = list(view.counts)
  counts[r_idx] -= 1
  caps_by_color = [list(row) for row in view.caps_by_color]
  caps_by_color[c_idx][r_idx] = 0
  tokens = list(view.tokens)
  if view.led_color_idx is not None and c_idx != view.led_color_idx:
    tokens[view.led_color_idx] = False
  caps = _caps_from_colors(caps_by_color, tokens)
  return adversarial_margin(counts, caps, view.plays_remaining - 1)


def post_discard_margin(view, rank_idx):
  """Adversarial margin after discarding one copy of `rank_idx`.

  During the discard phase plays_remaining equals num_tricks and the trick
  state is empty, so only the hand shrinks.
  """
  counts = list(view.counts)
  counts[rank_idx] -= 1
  caps = _caps_from_colors(view.caps_by_color, view.tokens)
  return adversarial_margin(counts, caps, view.plays_remaining)


def discard_view_from_state(state, player):
  """LivenessView during the discard phase (before any tricks)."""
  num_ranks = int(state._num_card_types)
  counts = [int(x) for x in state._hands[player]]
  board = state._board_ownership
  tokens = [bool(state._color_tokens[player][c]) for c in range(_NUM_COLORS)]
  caps_by_color = []
  for c in range(_NUM_COLORS):
    caps_by_color.append(
        [1 if int(board[c, r]) == -1 else 0 for r in range(num_ranks)])
  return LivenessView(counts, caps_by_color, tokens, int(state._num_tricks),
                      None)


def legal_play_cells(view):
  """Immediately legal (color, rank) plays implied by the view.

  Mirrors quantum_cat._trick_legal_actions WITHOUT the red-lead filter (the
  caller intersects with state.legal_actions anyway).
  """
  acts = []
  for r in range(view.num_ranks):
    if view.counts[r] <= 0:
      continue
    for c in range(_NUM_COLORS):
      if not view.tokens[c]:
        continue
      if view.caps_by_color[c][r] <= 0:
        continue
      acts.append(c * view.num_ranks + r)
  return sorted(acts)


def select_action_with_oracle(state, player, policy, min_margin=1,
                              apply_to_discard=True):
  """Oracle shield action selection.

  Keeps the policy's preferred legal action unless its post-action margin
  falls below `min_margin` while a strictly safer legal alternative exists;
  in that case picks the highest-margin action, tie-broken by policy mass.
  Dead moves (post margin < 0, i.e. guaranteed future paradox) are always
  vetoed when any alive move exists.

  Args:
    state: quantum_cat state at a decision node for `player`.
    policy: array over the 1000-action space (need not be normalized over
      legal actions).
    min_margin: tolerated post-action adversarial margin before overriding.
    apply_to_discard: also pick margin-maximizing discards (tie-break by
      policy).

  Returns:
    The selected action, or None if this phase is not handled (caller falls
    back to its base behavior, e.g. prediction phase).
  """
  legal = state.legal_actions(player)
  if not legal:
    return None
  phase = int(state._phase)
  if phase == 1 and apply_to_discard:
    view = discard_view_from_state(state, player)
    scored = []
    for a in legal:
      m = post_discard_margin(view, int(a))
      scored.append((m, float(policy[a]), a))
    best_margin = max(s[0] for s in scored)
    if best_margin <= min_margin:
      # All discards equally pressured - keep the policy's choice.
      return max(legal, key=lambda a: float(policy[a]))
    keep = [s for s in scored if s[0] >= min(best_margin, min_margin)]
    if not keep:
      keep = [s for s in scored if s[0] == best_margin]
    return max(keep, key=lambda s: (s[1], s[0]))[2]
  if phase != 3:
    return None
  if legal == [PARADOX_ACTION]:
    return PARADOX_ACTION
  view = view_from_state(state, player)
  scored = []
  for a in legal:
    if a == PARADOX_ACTION:
      continue
    m = post_play_margin(view, int(a))
    scored.append((m, float(policy[a]), a))
  if not scored:
    return max(legal, key=lambda a: float(policy[a]))
  best_margin = max(s[0] for s in scored)
  base = max(scored, key=lambda s: s[1])
  threshold = min(min_margin, best_margin)
  if base[0] >= threshold:
    return base[2]
  keep = [s for s in scored if s[0] >= threshold]
  if not keep:
    keep = [s for s in scored if s[0] == best_margin]
  return max(keep, key=lambda s: (s[1], s[0]))[2]
