import numpy as np
import pyspiel
from collections import deque

import numpy as np
import pyspiel
from collections import deque

# ---------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------

_DEFAULT_NUM_PLAYERS = 2  # If no param is given, defaults to 5. Now supports 2..5.

# Suits/Colors for trick-taking strength and adjacency board.
_COLORS = ["R", "B", "Y", "G"]
_NUM_COLORS = len(_COLORS)

# Paradox action code
_ACTION_PARADOX = 999

# For predictions, we only allow 1..4 in the 3–5 player version.
# In 2-player mode, we skip predictions entirely.
_ACTION_PREDICT_OFFSET = 100  # so "Predict=1" -> 101, "Predict=4" -> 104

# Special marker for "blocked" squares (2p leftover reveal)
_BLOCKED_MARKER = -2  # If board_ownership[color, rank] == -2, that (color,rank) is blocked.

# ---------------------------------------------------------------------
# Game Type & Info
# ---------------------------------------------------------------------

_MAX_GAME_LENGTH = 500  # Enough to cover dealing, discarding, bidding, trick-taking

_GAME_TYPE = pyspiel.GameType(
    short_name="python_quantum_cat",
    long_name="Quantum Cat Trick-Taking (One-Round, Adjacency Bonus)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,  # Will be overridden per-instance
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=5,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={"players": _DEFAULT_NUM_PLAYERS}
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=1000,   # Large enough for all moves + paradox
    max_chance_outcomes=45,      # Max deck size for 5 players
    num_players=_DEFAULT_NUM_PLAYERS,
    min_utility=-50.0,
    max_utility=50.0,
    utility_sum=0.0,  # We'll fix the sum to 0 only in 2p mode within the code
    max_game_length=_MAX_GAME_LENGTH
)

# ---------------------------------------------------------------------
# QuantumCatGame
# ---------------------------------------------------------------------
class QuantumCatGame(pyspiel.Game):
  """
  One-round 'Quantum Cat Trick-Taking' with adjacency scoring:
    - 2–5 players.
    - If 2 players => zero-sum payoffs; special rules:
      * Ranks 1..5, 25 total cards.
      * Deal 10 each => leftover 5 => reveal 3 & block them in Green column.
      * Skip discard & prediction, then play 9 tricks.
      * If a player wins <=4 tricks, they also get adjacency bonus.
      * Final payoffs made zero-sum: (score0 - score1, score1 - score0).
    - If 3–5 players => general-sum:
      * 3: ranks 1..6 (30 cards)
      * 4: ranks 1..8 (40 cards)
      * 5: ranks 1..9 (45 cards)
      * Each discards 1, then predicts (1..4), then plays trick-taking.
      * Adjacency bonus only if you match your prediction exactly (and not paradox).
  """

  def __init__(self, game_type, game_info, params=None):
    """Initialize the QuantumCat game.
    
    Args:
      game_type: The registered game type
      game_info: The game info
      params: Optional params dict with e.g. "players"
    """
    if params is None:
      params = {}
    num_players = int(params.get("players", _DEFAULT_NUM_PLAYERS))
    
    if not (2 <= num_players <= 5):
      raise ValueError("QuantumCatGame only supports 2..5 players.")

    # Decide rank range by #players
    if num_players == 2:
      self.max_card_value = 5   # ranks 1..5
    elif num_players == 3:
      self.max_card_value = 6   # ranks 1..6
    elif num_players == 4:
      self.max_card_value = 8   # ranks 1..8
    else:  # num_players == 5
      self.max_card_value = 9   # ranks 1..9

    # 5 copies of each rank
    self.total_cards = 5 * self.max_card_value
    self.num_card_types = self.max_card_value
    self.num_colors = len(_COLORS)

    # Override game_info for this specific instance
    game_info = pyspiel.GameInfo(
        num_distinct_actions=1000,
        max_chance_outcomes=45,
        num_players=num_players,
        min_utility=-50.0,
        max_utility=50.0,
        utility_sum=0.0,
        max_game_length=_MAX_GAME_LENGTH
    )

    # For 2p, use a zero-sum game type
    if num_players == 2:
      game_type = game_type._replace(
          utility=pyspiel.GameType.Utility.ZERO_SUM
      )

    super().__init__(game_type, game_info, params)

    # Cards per player initially + #tricks
    if num_players == 2:
      # 25 total => each gets 10 => leftover=5 => reveal 3 in Green => skip discard/pred
      # => 9 total tricks
      self.cards_per_player_initial = 10
      self.num_tricks = 9
    elif num_players == 3:
      # 30 total => each gets 10 => discards 1 => 9 => 8 tricks
      self.cards_per_player_initial = 10
      self.num_tricks = 8
    elif num_players == 4:
      # 40 total => each gets 10 => discards 1 => 9 => 8 tricks
      self.cards_per_player_initial = 10
      self.num_tricks = 8
    else:  # 5 players
      # 45 total => each gets 9 => discards 1 => 8 => 7 tricks
      self.cards_per_player_initial = 9
      self.num_tricks = 7

  def new_initial_state(self):
    return QuantumCatGameState(self)

  def num_distinct_actions(self):
    """
    Must be >= the largest action index used for any player action.
    We use 1000 because the code sets PARADOX = 999 explicitly,
    and color-rank actions can occupy lower indices.
    """
    return 1000

  def observation_tensor_shape(self):
    """
    Return a concrete 1D shape for the single-player viewpoint.
    This matches the pieces defined in QuantumCatObserver
    (with PrivateInfoType.SINGLE_PLAYER).
    """
    num_players = self.num_players()
    num_colors = 4
    num_card_types = self.num_card_types

    # Sum of public pieces:
    #  1) current_player -> size = num_players
    #  2) phase -> size = 5
    #  3) led_color -> size = (num_colors + 1)
    #  4) trick_number -> size = 1
    #  5) start_player -> size = num_players
    #  6) cards_played_in_trick -> size = 2 * num_players
    #  7) predictions -> size = num_players
    #  8) tricks_won -> size = num_players
    #  9) board_ownership -> size = num_colors * num_card_types
    # 10) color_tokens -> size = num_players * num_colors
    # And single-player private pieces:
    # 11) hand -> size = num_card_types
    # 12) discarded_rank -> size = 1
    # 
    # So total = 6 * num_players for #1 + #5 + #6 + #7 + #8
    #          + 5 + (num_colors + 1) + 1
    #          + (num_colors * num_card_types)
    #          + (num_players * num_colors)
    #          + num_card_types + 1
    #
    # We'll compute it directly:

    total_size = (
        6 * num_players
        + 5
        + (num_colors + 1)
        + 1
        + (num_colors * num_card_types)
        + (num_players * num_colors)
        + num_card_types
        + 1
    )
    return [total_size]

  def make_py_observer(self, iig_obs_type=None, params=None):
    return QuantumCatObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        self.num_players(),
        self.num_card_types,
        self.num_colors,
        params
    )

# ---------------------------------------------------------------------
# QuantumCatGameState
# ---------------------------------------------------------------------
class QuantumCatGameState(pyspiel.State):
  """
  Phases:
    0) Dealing (chance)
    1) Discard (skipped entirely if 2 players)
    2) Prediction (skipped entirely if 2 players)
    3) Trick-taking
    4) Scoring (terminal)

  2-player special logic:
    - Ranks=1..5 => 25 total.
    - After dealing 10 each, leftover 5 => reveal top 3 => block them in Green => skip discard/pred => 9 tricks total.
    - Adjacency bonus if (tricks_won <= 4) and no paradox.
    - Final payoffs forced to sum to 0 => (score0, -score0).

  3–5 player logic:
    - Same as original Cat in the Box: discard 1, predict 1..4, then do normal trick-taking.
    - Adjacency bonus if you match your prediction exactly and do not paradox.
    - General-sum payoffs.
  """

  def __init__(self, game: QuantumCatGame):
    super().__init__(game)
    self._game = game

    # Basic parameters
    self._num_players = game.num_players()
    self._num_card_types = game.num_card_types
    self._num_colors = game.num_colors
    self._cards_per_player_initial = game.cards_per_player_initial
    self._total_cards = game.total_cards
    self._num_tricks = game.num_tricks

    # Phase: 0..4
    self._phase = 0

    # Create & shuffle deck
    self._deck = self._create_deck()
    np.random.shuffle(self._deck)
    self._cards_dealt = 0
    self._deal_player = 0

    # Each player's hand => 1D array [num_card_types], storing how many copies of each rank
    self._hands = [np.zeros(self._num_card_types, dtype=int) for _ in range(self._num_players)]

    # Discard tracking
    self._has_discarded = [False] * self._num_players
    self._discarded_cards = [-1] * self._num_players  # rank or -1

    # Predictions in [1..4], or -1 if not yet made
    self._predictions = [-1] * self._num_players

    # Trick info
    self._trick_number = 0
    self._start_player = 0
    self._current_player = pyspiel.PlayerId.CHANCE
    self._led_color = None
    self._cards_played_this_trick = [None] * self._num_players  # each slot is (rank_val, color_str)
    self._tricks_won = np.zeros(self._num_players, dtype=int)

    # Board ownership => [color_idx, rank_idx] = player_id, -1 if empty, -2 if blocked
    self._board_ownership = -1 * np.ones((self._num_colors, self._num_card_types), dtype=int)

    # Paradox tracking
    self._has_paradoxed = [False] * self._num_players

    # Color tokens => [player][color_idx] bool
    # True means the player can still declare that color in the trick-taking phase
    self._color_tokens = np.ones((self._num_players, self._num_colors), dtype=bool)

    # Terminal and rewards
    self._game_over = False
    self._returns = [0.0] * self._num_players
    self._rewards = [0.0] * self._num_players
    self._player_adjacency_bonus = [0.0] * self._num_players

    # Keep track if trump (Red) is broken
    self._trump_broken = False

    # Keep record of completed tricks (for info, tracking, etc.)
    self._completed_tricks = []

  # --------------
  # Deck creation
  # --------------
  def _create_deck(self):
    deck = []
    for val in range(1, self._game.max_card_value + 1):
      for _ in range(5):
        deck.append(val)
    return np.array(deck, dtype=int)

  # -------------------------------------------------------------------
  # PySpiel interface methods
  # -------------------------------------------------------------------
  def current_player(self):
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    return self._current_player

  def is_terminal(self):
    return self._game_over

  def returns(self):
    return list(self._returns)

  def rewards(self):
    return list(self._rewards)

  def legal_actions(self, player=None):
    if player is None:
      player = self.current_player()
    return self._legal_actions(player)

  def chance_outcomes(self):
    # Dealing
    num_left = self._total_cards - self._cards_dealt
    if num_left <= 0:
      return []
    p = 1.0 / num_left
    return [(i, p) for i in range(num_left)]

  def _action_to_string(self, player, action):
    if player == pyspiel.PlayerId.CHANCE:
      return f"DealCard(index={action})"
    if action == _ACTION_PARADOX:
      return "PARADOX"

    if self._phase == 1:  # discard
      return f"Discard: rank={action+1}"
    elif self._phase == 2:  # predict
      pred = action - _ACTION_PREDICT_OFFSET
      return f"Prediction={pred}"
    elif self._phase == 3:  # trick-taking
      color_idx = action // self._num_card_types
      rank_idx = action % self._num_card_types
      return f"Play: (rank={rank_idx+1}, color={_COLORS[color_idx]})"

    return f"UnknownAction={action}"

  def __str__(self):
    return (
      f"Phase={self._phase}, Trick={self._trick_number}, CurrentPlayer={self._current_player}\n"
      f"Hands={self._hands}\n"
      f"HasDiscarded={self._has_discarded}, Predictions={self._predictions}\n"
      f"TricksWon={self._tricks_won}, LedColor={self._led_color}\n"
      f"Paradoxed={self._has_paradoxed}, GameOver={self._game_over}\n"
      f"ColorTokens={self._color_tokens}\n"
      "BoardOwnership=\n"
      + "\n".join(
          f"{_COLORS[c_idx]} {self._board_ownership[c_idx]}"
          for c_idx in range(self._num_colors)
        )
      + "\n"
      f"CompletedTricks={self._completed_tricks}"
    )

  def _apply_action(self, action):
    if self.is_chance_node():
      self._apply_deal(action)
    else:
      if self._phase == 1:
        self._apply_discard(action)
      elif self._phase == 2:
        self._apply_prediction(action)
      elif self._phase == 3:
        if action == _ACTION_PARADOX:
          self._apply_paradox()
        else:
          self._apply_trick_action(action)

  def is_chance_node(self):
    return (self._current_player == pyspiel.PlayerId.CHANCE)

  # -------------------------------------------------------------------
  # Phase 0: Dealing (chance)
  # -------------------------------------------------------------------
  def _apply_deal(self, outcome_index):
    chosen_idx = self._cards_dealt + outcome_index
    chosen_card = self._deck[chosen_idx]
    # swap
    self._deck[chosen_idx], self._deck[self._cards_dealt] = \
        self._deck[self._cards_dealt], self._deck[chosen_idx]

    # add to player's hand
    self._hands[self._deal_player][chosen_card - 1] += 1
    self._cards_dealt += 1
    self._deal_player = (self._deal_player + 1) % self._num_players

    # Once each player has initial cards, move to next phase
    total_needed = self._num_players * self._cards_per_player_initial
    if self._cards_dealt >= total_needed:
      if self._num_players == 2:
        # 2p => reveal leftover 3 in green => skip discard & prediction => jump phase=3
        self._handle_two_player_leftover()
        self._phase = 3
        self._trick_number = 0
        self._start_player = 0
        self._current_player = self._start_player
        self._led_color = None
        self._cards_played_this_trick = [None] * self._num_players
      else:
        # 3..5 => proceed to discard
        self._phase = 1
        self._current_player = 0
    else:
      # Still dealing
      self._current_player = pyspiel.PlayerId.CHANCE

  def _handle_two_player_leftover(self):
    """
    In 2p: leftover is 5 cards (25 total minus 2×10=20).
    Reveal top 3 => block them in Green => remove them from deck.
    """
    leftover = self._deck[self._cards_dealt:]
    if len(leftover) != 5:
      raise ValueError("Expected exactly 5 leftover cards in 2p mode.")
    green_idx = 3  # G
    # Reveal top 3 => block them in (green, rank-1)
    for i in range(3):
      rank_val = leftover[i]
      self._board_ownership[green_idx, rank_val - 1] = _BLOCKED_MARKER

    # Remove them from the deck => leftover is no longer used
    self._deck = self._deck[:self._cards_dealt]

  # -------------------------------------------------------------------
  # Phase 1: Discard
  # -------------------------------------------------------------------
  def _apply_discard(self, action):
    player = self._current_player
    self._hands[player][action] -= 1
    self._discarded_cards[player] = action + 1
    self._has_discarded[player] = True
    self._advance_discard_phase()

  def _advance_discard_phase(self):
    if all(self._has_discarded):
      # move to predictions
      self._phase = 2
      self._current_player = 0
    else:
      self._current_player = (self._current_player + 1) % self._num_players

  # -------------------------------------------------------------------
  # Phase 2: Prediction
  # -------------------------------------------------------------------
  def _apply_prediction(self, action):
    player = self._current_player
    pred = action - _ACTION_PREDICT_OFFSET
    self._predictions[player] = pred
    self._advance_prediction_phase()

  def _advance_prediction_phase(self):
    next_p = (self._current_player + 1) % self._num_players
    if all(x >= 1 for x in self._predictions):
      # Everyone predicted => start trick-taking
      self._phase = 3
      self._trick_number = 0
      self._start_player = 0
      self._current_player = self._start_player
      self._led_color = None
      self._cards_played_this_trick = [None] * self._num_players
    else:
      self._current_player = next_p

  # -------------------------------------------------------------------
  # Phase 3: Trick-taking
  # -------------------------------------------------------------------
  def _legal_actions(self, player):
    # CHANCE node (dealing)
    if player == pyspiel.PlayerId.CHANCE:
      if self._phase == 0:
        num_left = self._total_cards - self._cards_dealt
        return list(range(num_left))
      return []

    # Phase 1 => discard (skipped if 2p)
    if self._phase == 1:
      if not self._has_discarded[player]:
        return self._discard_actions(player)
      else:
        return []

    # Phase 2 => predictions (skipped if 2p)
    if self._phase == 2:
      if self._predictions[player] < 0:
        return [101, 102, 103, 104]  # i.e. 1..4
      else:
        return []

    # Phase 3 => trick-taking
    if self._phase == 3:
      return self._trick_legal_actions(player)

    # Phase 4 => terminal => no actions
    return []

  def _discard_actions(self, player):
    hand_vec = self._hands[player]
    acts = []
    for rank_idx in range(self._num_card_types):
      if hand_vec[rank_idx] > 0:
        acts.append(rank_idx)
    return sorted(acts)

  def _trick_legal_actions(self, player):
    """
    If no legal actions => must PARADOX.
    - A color-rank is legal if:
      * Not already claimed or blocked (board_ownership != -1 and != -2).
      * The player's hand has that rank > 0.
      * The player still has the color token available.
    - If leading and trump not broken => can't lead red unless it's only color left.
    """
    hand_vec = self._hands[player]
    actions = []
    for rank_idx in range(self._num_card_types):
      if hand_vec[rank_idx] <= 0:
        continue
      for c_idx in range(self._num_colors):
        if not self._color_tokens[player][c_idx]:
          continue
        if self._board_ownership[c_idx, rank_idx] != -1:  # either claimed or blocked
          continue
        # valid
        act = c_idx * self._num_card_types + rank_idx
        actions.append(act)

    if self._led_color is None and not self._trump_broken:
      # remove red leads if we can lead something else
      red_actions = [a for a in actions if (a // self._num_card_types) == 0]
      if red_actions and len(actions) > len(red_actions):
        actions = [a for a in actions if (a // self._num_card_types) != 0]

    if not actions:
      return [_ACTION_PARADOX]
    return sorted(actions)

  def _apply_trick_action(self, action):
    color_idx = action // self._num_card_types
    rank_idx = action % self._num_card_types
    player = self._current_player

    # Remove from hand
    self._hands[player][rank_idx] -= 1
    # Mark ownership
    self._board_ownership[color_idx, rank_idx] = player

    rank_val = rank_idx + 1
    color_str = _COLORS[color_idx]
    self._cards_played_this_trick[player] = (rank_val, color_str)

    # If not leading color and it's red => trump broken
    if self._led_color is not None and color_str == "R" and color_str != self._led_color:
      self._trump_broken = True

    # If no lead, set it
    if self._led_color is None:
      self._led_color = color_str
    else:
      # If did not follow lead => lose that color token
      if color_str != self._led_color:
        led_idx = _COLORS.index(self._led_color)
        self._color_tokens[player][led_idx] = False

    # Next
    self._current_player = (self._current_player + 1) % self._num_players
    if self._current_player == self._start_player:
      # Trick ends
      winner = self._evaluate_trick_winner()
      self._tricks_won[winner] += 1

      # record the trick
      completed = []
      for p_idx, cardinfo in enumerate(self._cards_played_this_trick):
        completed.append((p_idx, cardinfo))
      self._completed_tricks.append(completed)

      self._trick_number += 1
      self._start_player = winner
      self._current_player = winner
      self._led_color = None
      self._cards_played_this_trick = [None] * self._num_players

      # check if done
      if self._trick_number >= self._num_tricks:
        self._phase = 4
        self._game_over = True
        self._compute_final_scores()

  def _apply_paradox(self):
    player = self._current_player
    self._has_paradoxed[player] = True
    self._phase = 4
    self._game_over = True
    self._compute_final_scores()

  # -------------------------------------------------------------------
  # Trick evaluation
  # -------------------------------------------------------------------
  def _evaluate_trick_winner(self):
    # if any red => highest red wins
    red_plays = [(p, v) for p,(v,c) in enumerate(self._cards_played_this_trick) if c == "R"]
    if red_plays:
      return max(red_plays, key=lambda x: x[1])[0]

    # else highest in led color
    if self._led_color is None:
      all_plays = [(p, v) for p,(v,c) in enumerate(self._cards_played_this_trick)]
      return max(all_plays, key=lambda x: x[1])[0]
    led_plays = [(p,v) for p,(v,c) in enumerate(self._cards_played_this_trick)
                 if c == self._led_color]
    if not led_plays:
      all_plays = [(p,v) for p,(v,c) in enumerate(self._cards_played_this_trick)]
      return max(all_plays, key=lambda x: x[1])[0]
    return max(led_plays, key=lambda x: x[1])[0]

  # -------------------------------------------------------------------
  # Phase 4: Scoring
  # -------------------------------------------------------------------
  def _compute_final_scores(self):
    raw_scores = [0.0] * self._num_players

    for p in range(self._num_players):
      tricks = self._tricks_won[p]
      if self._has_paradoxed[p]:
        # paradox => negative your trick count
        raw_scores[p] = -float(tricks)
        self._player_adjacency_bonus[p] = 0.0
      else:
        base = float(tricks)
        if self._num_players == 2:
          # If not paradox, adjacency bonus if <= 4
          if tricks <= 4:
            cluster_bonus = self._largest_cluster_for_player(p)
            raw_scores[p] = base + cluster_bonus
            self._player_adjacency_bonus[p] = cluster_bonus
          else:
            raw_scores[p] = base
            self._player_adjacency_bonus[p] = 0.0
        else:
          # 3..5 => adjacency bonus if matched your prediction exactly
          pred = self._predictions[p]
          if pred == tricks:
            cluster_bonus = self._largest_cluster_for_player(p)
            raw_scores[p] = base + cluster_bonus
            self._player_adjacency_bonus[p] = cluster_bonus
          else:
            raw_scores[p] = base
            self._player_adjacency_bonus[p] = 0.0

    # If 2 players => convert to zero-sum
    if self._num_players == 2:
      # Suppose raw_scores=[r0, r1].
      # We'll define final[0] = (r0 - r1), final[1] = (r1 - r0).
      diff = raw_scores[0] - raw_scores[1]
      raw_scores[0] = diff
      raw_scores[1] = -diff

    # Add these to self._returns & self._rewards
    for p in range(self._num_players):
      self._returns[p] += raw_scores[p]
      self._rewards[p] += raw_scores[p]

  # -------------------------------------------------------------------
  # BFS for adjacency
  # -------------------------------------------------------------------
  def _largest_cluster_for_player(self, player):
    visited = np.zeros((self._num_colors, self._num_card_types), dtype=bool)
    max_cluster = 0

    def neighbors(c, r):
      for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]:
        cc, rr = c+dc, r+dr
        if 0 <= cc < self._num_colors and 0 <= rr < self._num_card_types:
          yield (cc, rr)

    for c_idx in range(self._num_colors):
      for r_idx in range(self._num_card_types):
        if (self._board_ownership[c_idx, r_idx] == player) and (not visited[c_idx, r_idx]):
          # BFS/DFS
          size = 0
          queue = deque([(c_idx, r_idx)])
          visited[c_idx, r_idx] = True
          while queue:
            c0, r0 = queue.popleft()
            size += 1
            for (c1, r1) in neighbors(c0, r0):
              if not visited[c1, r1] and self._board_ownership[c1, r1] == player:
                visited[c1, r1] = True
                queue.append((c1, r1))
          if size > max_cluster:
            max_cluster = size
    return max_cluster

  def _count_cards_played_by(self, player_id):
    """
    Utility to see how many cards a player has played so far,
    for e.g. partial info / resampling.
    """
    count = 0
    for trick_list in self._completed_tricks:
      for (p, cardinfo) in trick_list:
        if p == player_id and cardinfo is not None:
          count += 1
    # Also check the in-progress trick
    if self._cards_played_this_trick[player_id] is not None:
      count += 1
    return count

  def observation_tensor(self, player=None):
    """Return the flattened 1D observation for the specified player (or current)."""
    if player is None:
      player = self._current_player

    # Allocate a 1D numpy array of the same size that observation_tensor_shape() says.
    obs_size = self._game.observation_tensor_shape()[0]
    obs = np.zeros(obs_size, dtype=np.float32)

    # Build a temporary observer and ask it to fill its .tensor
    observer = self._game.make_py_observer(
        pyspiel.IIGObservationType(perfect_recall=False)
    )
    observer.set_from(self, player)

    # Copy the observer's data into our obs array:
    obs[:] = observer.tensor
    return obs

  def resample_from_infostate(self, player_id, sampler):
    """
    Returns a new state with all hidden information re-sampled
    consistently with 'player_id's' perspective of the game so far.
    - The player's own hand & discard are fully known to them.
    - Other players' discards, if face-down, are unknown => sample them.
    - All publicly played cards are removed from the unknown deck.
    - We re-deal the unknown cards to each other player's hidden hand and discard.
    """

    # 1) Clone this full state to overwrite hidden areas
    cloned = self.clone()

    # 2) Build a multiset of all possible ranks [1..max_card_value], each with 5 copies
    max_val = self._game.max_card_value
    card_counts = [5] * max_val  # card_counts[r-1] => how many copies of rank r remain

    # 3) Remove from card_counts all cards that 'player_id' definitely knows are used
    #    or out of the deck.
    #    (A) The player's own hand
    for rank_idx, count_in_hand in enumerate(self._hands[player_id]):
      card_counts[rank_idx] -= count_in_hand

    #    (B) The player's own face-down discard (they know it)
    my_discard_rank = self._discarded_cards[player_id]
    if my_discard_rank != -1:
      card_counts[my_discard_rank - 1] -= 1

    #    (C) All publicly known trick plays
    #        For each completed trick + current trick, those ranks are visible.
    for trick_list in self._completed_tricks:
      for (p, cardinfo) in trick_list:
        if cardinfo is not None:
          rank_val, color_str = cardinfo
          card_counts[rank_val - 1] -= 1
    # Also the in-progress trick
    for p, cardinfo in enumerate(self._cards_played_this_trick):
      if cardinfo is not None:
        rank_val, color_str = cardinfo
        card_counts[rank_val - 1] -= 1

    # 4) Figure out how many unknown cards each other player must still hold in-hand,
    #    plus whether we must sample their discard (face-down).
    unknown_allocations = [0] * self._num_players
    needs_discard = [False] * self._num_players

    # Determine how many each started with (after discarding).
    # For 5p => 8 in hand, for 4p => 9 in hand, for 3p => 10 in hand, etc.
    if self._num_players == 5:
      starting_in_hand_after_discard = 8
    elif self._num_players == 4:
      starting_in_hand_after_discard = 9
    elif self._num_players == 3:
      starting_in_hand_after_discard = 9
    else:
      raise ValueError("Only 3..5 players supported in Cat in the Box")

    for p in range(self._num_players):
      if p == player_id:
        # We know our own exact hand + discard
        unknown_allocations[p] = 0
        needs_discard[p] = False
        continue

      # If discard is still -1, that means from *player_id*'s perspective,
      # p's discard is unknown => we must sample it.
      if self._discarded_cards[p] == -1:
        needs_discard[p] = True
      else:
        needs_discard[p] = False

      # Count how many times p has played a card so far
      played_count = self._count_cards_played_by(p)
      # Then p must have (starting_in_hand_after_discard - played_count) cards left hidden
      unknown_allocations[p] = starting_in_hand_after_discard - played_count

    # 5) Create the unknown deck list from card_counts
    unknown_deck = []
    for rank_idx, ccount in enumerate(card_counts):
      if ccount < 0:
        raise ValueError(f"Contradictory card counts for rank {rank_idx+1}")
      unknown_deck.extend([rank_idx] * ccount)
    np.random.shuffle(unknown_deck)

    # 6) Assign unknown discards + unknown hands to each other player
    deck_pos = 0
    for p in range(self._num_players):
      if p == player_id:
        # We already have our known data
        continue

      # 6a) If p's discard is unknown => pick 1 random rank for it
      if needs_discard[p]:
        discard_rank_idx = unknown_deck[deck_pos]
        deck_pos += 1
        cloned._discarded_cards[p] = discard_rank_idx + 1

      # 6b) Now re-deal p's unknown portion of their hand
      needed = unknown_allocations[p]
      cloned._hands[p].fill(0)
      for _ in range(needed):
        rank_idx = unknown_deck[deck_pos]
        deck_pos += 1
        cloned._hands[p][rank_idx] += 1

    return cloned

# ---------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------
class QuantumCatObserver:
  """Observer, conforming to the PyObserver interface."""

  def __init__(self, iig_obs_type, num_players, num_card_types, num_colors, params=None):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    self.iig_obs_type = iig_obs_type
    self.num_players = num_players
    self.num_card_types = num_card_types
    self.num_colors = num_colors

    pieces = [
        ("current_player", num_players, (num_players,)),
        ("phase", 5, (5,)),
        ("led_color", num_colors + 1, (num_colors + 1,)),
        ("trick_number", 1, (1,)),
        ("start_player", num_players, (num_players,)),
        ("cards_played_in_trick", 2 * num_players, (2 * num_players,)),
        ("predictions", num_players, (num_players,)),
        ("tricks_won", num_players, (num_players,)),
        ("board_ownership", num_colors * num_card_types, (num_colors, num_card_types)),
        ("color_tokens", num_players * num_colors, (num_players, num_colors)),
    ]
    # If perfect recall or private info is SINGLE_PLAYER, we store the player's hand & discard
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("hand", num_card_types, (num_card_types,)))
      pieces.append(("discarded_rank", 1, (1,)))

    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, dtype=np.float32)

    # map each piece name -> slice in self.tensor
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index+size].reshape(shape)
      index += size

  def set_from(self, state, player):
    self.tensor.fill(0)
    cp = state.current_player()

    if cp not in [pyspiel.PlayerId.TERMINAL, pyspiel.PlayerId.CHANCE]:
      self.dict["current_player"][cp] = 1.0

    if 0 <= state._phase <= 4:
      self.dict["phase"][state._phase] = 1.0

    self._encode_led_color(state._led_color)
    self.dict["trick_number"][0] = state._trick_number
    if not state.is_terminal():
      self.dict["start_player"][state._start_player] = 1.0

    # cards played in current trick
    arr = self.dict["cards_played_in_trick"]
    color_map = {"R": 0, "B": 1, "Y": 2, "G": 3}
    for p in range(self.num_players):
      cardinfo = state._cards_played_this_trick[p]
      if cardinfo is not None:
        rank_val, color_str = cardinfo
        arr[2*p] = rank_val
        arr[2*p + 1] = color_map[color_str]
      else:
        arr[2*p] = -1
        arr[2*p + 1] = -1

    # predictions, tricks won
    for p in range(self.num_players):
      self.dict["predictions"][p] = state._predictions[p]
      self.dict["tricks_won"][p] = state._tricks_won[p]

    # board ownership
    for c_idx in range(self.num_colors):
      for r_idx in range(self.num_card_types):
        self.dict["board_ownership"][c_idx][r_idx] = float(state._board_ownership[c_idx, r_idx])

    # color tokens
    color_tokens_arr = self.dict["color_tokens"]
    for p in range(self.num_players):
      for c_idx in range(self.num_colors):
        color_tokens_arr[p, c_idx] = float(state._color_tokens[p, c_idx])

    # single-player private info
    if self.iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      for i in range(self.num_card_types):
        self.dict["hand"][i] = state._hands[player][i]
      self.dict["discarded_rank"][0] = float(state._discarded_cards[player])

  def _encode_led_color(self, led_color):
    color_map = {"R": 0, "B": 1, "Y": 2, "G": 3}
    if led_color is None:
      self.dict["led_color"][-1] = 1.0
    else:
      self.dict["led_color"][color_map[led_color]] = 1.0

  def string_from(self, state, player):
    """Human-readable string for debugging."""
    pieces = []
    cp = state.current_player()
    if cp not in (pyspiel.PlayerId.TERMINAL, pyspiel.PlayerId.CHANCE):
      pieces.append(f"current_player=p{cp}")
    pieces.append(f"phase={state._phase}")
    pieces.append(f"predictions={list(state._predictions)}")
    if self.iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(f"hand={state._hands[player]}")
      pieces.append(f"led_color={state._led_color}")

    pieces.append("color_tokens=" + "\n".join(
        f"p{p}: {state._color_tokens[p]}" for p in range(self.num_players)
    ))

    # Current trick's plays
    trick_plays = []
    for p in range(self.num_players):
      card = state._cards_played_this_trick[p]
      if card is not None:
        rank_val, color_str = card
        trick_plays.append(f"p{p}=>{rank_val}{color_str}")
    if trick_plays:
      pieces.append("Trick:[" + ",".join(trick_plays) + "]")

    # Board ownership
    board_str = [
        f"{_COLORS[c_idx]}: {state._board_ownership[c_idx]}"
        for c_idx in range(self.num_colors)
    ]
    pieces.append("board_ownership=\n" + "\n".join(board_str))
    return " | ".join(str(p) for p in pieces)

# ---------------------------------------------------------------------
# Register the game with PySpiel
# ---------------------------------------------------------------------
pyspiel.register_game(_GAME_TYPE, QuantumCatGame)

