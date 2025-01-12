import numpy as np
import pyspiel
from collections import deque

# ---------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------

_DEFAULT_NUM_PLAYERS = 5  # If no param is given, defaults to 5. Supports 3..5.

# Suits/Colors for trick-taking strength and adjacency board.
_COLORS = ["R", "B", "Y", "G"]
_NUM_COLORS = len(_COLORS)

# Paradox action code
_ACTION_PARADOX = 999

# For predictions, we only allow 1..4 (no 0, no >4).
# We place them at action codes [101..104].
_ACTION_PREDICT_OFFSET = 100  # so "Predict=1" -> 101, "Predict=4" -> 104

# ---------------------------------------------------------------------
# GameType & GameInfo
# ---------------------------------------------------------------------
_GAME_TYPE = pyspiel.GameType(
    short_name="python_quantum_cat",
    long_name="Quantum Cat Trick-Taking (One-Round, Adjacency Bonus)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=5,
    min_num_players=3,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "players": _DEFAULT_NUM_PLAYERS
    }
)

_MAX_GAME_LENGTH = 500  # Enough to cover dealing, discarding, bidding, trick-taking

_QUANTUM_CAT_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=1000,   # Large enough for all moves + paradox
    max_chance_outcomes=45,      # Max deck size for 5 players
    num_players=_DEFAULT_NUM_PLAYERS,
    min_utility=-50.0,
    max_utility=50.0,
    utility_sum=0.0,
    max_game_length=_MAX_GAME_LENGTH,
)

# ---------------------------------------------------------------------
# QuantumCatGame
# ---------------------------------------------------------------------
class QuantumCatGame(pyspiel.Game):
  """
  One-round 'Quantum Cat Trick-Taking' with adjacency scoring for correct bids.
  - 3..5 players
  - Ranks: 5 players => 1..9, 4 => 1..8, 3 => 1..7
  - Discard once, then predict (1..4), then play 'num_tricks' tricks, then score.
  """

  def __init__(self, params=None):
    game_parameters = params or dict()
    num_players = game_parameters.get("players", _DEFAULT_NUM_PLAYERS)
    game_info = pyspiel.GameInfo(
        num_distinct_actions=1000,   # Large enough for all moves + paradox
        max_chance_outcomes=45,      # Max deck size for 5 players
        num_players=num_players,
        min_utility=-50.0,
        max_utility=50.0,
        utility_sum=0.0,
        max_game_length=_MAX_GAME_LENGTH,
    )
    super().__init__(_GAME_TYPE, game_info, game_parameters)
    assert 3 <= num_players <= 5, "Only 3..5 players supported."

    # Card range depends on #players
    self.max_card_value = {3: 7, 4: 8, 5: 9}[num_players]
    # 5 copies of each rank
    self.total_cards = 5 * self.max_card_value
    self.num_card_types = self.max_card_value
    self.num_colors = _NUM_COLORS

    # Cards per player initially + #tricks
    if num_players == 5:
      # 45 total => each gets 9 => discards 1 => 8 => we play 7 tricks
      self.cards_per_player_initial = 9
      self.num_tricks = 7
    elif num_players == 4:
      # 40 total => each gets 10 => discards 1 => 9 => we play 8 tricks
      self.cards_per_player_initial = 10
      self.num_tricks = 8
    else:  # 3 players => for example, can deal 11 each, discard 1 => 10 => etc.
      self.cards_per_player_initial = 11
      self.num_tricks = 10

  def new_initial_state(self):
    return QuantumCatGameState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return QuantumCatObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        self.num_players(), self.num_card_types, self.num_colors, params
    )

# ---------------------------------------------------------------------
# QuantumCatGameState
# ---------------------------------------------------------------------
class QuantumCatGameState(pyspiel.State):
  """
  Phases:
    0) Dealing (chance)
    1) Discard (each player discards 1 card)
    2) Prediction (must pick 1..4)
    3) Trick-taking (num_tricks)
    4) Scoring (terminal)

  Adjacency Board => _board_ownership[color][rank] = player_id or -1 if empty.
  We place tokens whenever a player declares (color, rank).
  Then at game end, if player didn't paradox and matched their bid,
  they get adjacency bonus = size of largest connected cluster of squares that belong to them.

  New “No forced follow” logic:
    - Each player has color-tokens for R,B,Y,G. Initially all True.
    - If the lead color is not None and a player declares a *different* color,
      that player must remove the token for the lead color (set it False),
      thus can never again declare that lead color in the round.
    - If a player's token for a color is False, that player cannot declare that color anymore.
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

    # Each player's hand => length-_num_card_types vector
    self._hands = [np.zeros(self._num_card_types, dtype=int)
                   for _ in range(self._num_players)]

    # Discard tracking
    self._has_discarded = [False] * self._num_players
    self._discarded_cards = [-1] * self._num_players

    # Predictions
    self._predictions = [-1] * self._num_players

    # Trick info
    self._trick_number = 0
    self._start_player = 0
    self._current_player = pyspiel.PlayerId.CHANCE
    self._led_color = None
    self._cards_played_this_trick = [None] * self._num_players
    self._tricks_won = np.zeros(self._num_players, dtype=int)

    # Board ownership (color, rank) => which player placed a token? -1 => empty
    self._board_ownership = -1 * np.ones((self._num_colors, self._num_card_types), dtype=int)

    # Paradox tracking
    self._has_paradoxed = [False] * self._num_players

    # Each player’s color tokens: True means they can still declare that color
    # Indexed by [player][color_idx]
    self._color_tokens = np.ones((self._num_players, self._num_colors), dtype=bool)

    # Terminal and rewards
    self._game_over = False
    self._returns = [0.0] * self._num_players
    self._rewards = [0.0] * self._num_players  # per-step rewards
    
    # Track if trump has been broken
    self._trump_broken = False

  # --------------
  # Deck creation
  # --------------
  def _create_deck(self):
    deck = []
    for val in range(1, self._game.max_card_value+1):
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
    """Total reward for each player over the course of the game so far."""
    return list(self._returns)

  def rewards(self):
    """Returns rewards for the current step."""
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

    # Phase-based
    if self._phase == 1:
      # Discard => action in [0..num_card_types-1]
      return f"Discard: rank={action+1}"
    elif self._phase == 2:
      # Predictions => [101..104] => means 1..4
      pred = action - _ACTION_PREDICT_OFFSET
      return f"Prediction={pred}"
    elif self._phase == 3:
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
      f"BoardOwnership=\n{self._board_ownership}"
    )

  def _apply_action(self, action):
    if self.is_chance_node():
      self._apply_deal(action)
    else:
      if self._phase == 1:
        self._apply_discard(action)
      elif self._phase == 2:
        # Must be in [101..104], i.e. 1..4
        self._apply_prediction(action)
      elif self._phase == 3:
        if action == _ACTION_PARADOX:
          self._apply_paradox()
        else:
          self._apply_trick_action(action)

  # -------------------------------------------------------------------
  # Phase 0: Dealing (chance)
  # -------------------------------------------------------------------
  def _apply_deal(self, outcome_index):
    chosen_idx = self._cards_dealt + outcome_index
    chosen_card = self._deck[chosen_idx]
    # swap
    self._deck[chosen_idx], self._deck[self._cards_dealt] = \
      self._deck[self._cards_dealt], self._deck[chosen_idx]
    self._hands[self._deal_player][chosen_card - 1] += 1

    self._cards_dealt += 1
    self._deal_player = (self._deal_player + 1) % self._num_players

    # Once everyone has the required initial cards, go to discard phase
    if self._cards_dealt >= (self._num_players * self._cards_per_player_initial):
      self._phase = 1
      self._current_player = 0
    else:
      self._current_player = pyspiel.PlayerId.CHANCE

  # -------------------------------------------------------------------
  # Phase 1: Discard
  # -------------------------------------------------------------------
  def _apply_discard(self, action):
    player = self._current_player
    # print(f"[DEBUG] _apply_discard: player={player}, discard_action={action}, hand_before={self._hands[player].tolist()}")
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
    pred = action - _ACTION_PREDICT_OFFSET  # e.g. 101 => 1
    self._predictions[player] = pred
    self._advance_prediction_phase()

  def _advance_prediction_phase(self):
    next_p = (self._current_player + 1) % self._num_players
    if all(x >= 1 for x in self._predictions):  # i.e., everyone predicted 1..4
      # Move to trick-taking
      self._phase = 3
      self._trick_number = 0
      self._start_player = 0
      self._current_player = self._start_player
      self._led_color = None
      self._cards_played_this_trick = [None]*self._num_players
    else:
      self._current_player = next_p

  # -------------------------------------------------------------------
  # Phase 3: Trick-taking
  # -------------------------------------------------------------------
  def _legal_actions(self, player):
    # Phase-based logic
    if player == pyspiel.PlayerId.CHANCE:
      if self._phase == 0:
        num_left = self._total_cards - self._cards_dealt
        return list(range(num_left))
      return []
    if self._phase == 1:
      # Discard
      if not self._has_discarded[player]:
        return self._discard_actions(player)
      else:
        return []
    elif self._phase == 2:
      # Predictions in [1..4] => action codes [101..104]
      if self._predictions[player] < 0:
        return [101, 102, 103, 104]  # Predict 1..4
      else:
        return []
    elif self._phase == 3:
      return self._trick_legal_actions(player)
    # Phase 4 => no actions
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
    Each color can be declared only if player’s token for that color is still True.
    The player can always declare any color for which they have a token,
    even if there's a led color. If they pick a different color than the lead,
    they must forfeit that lead color token (done inside _apply_trick_action).
    """
    hand_vec = self._hands[player]

    actions = []
    for rank_idx in range(self._num_card_types):
      if hand_vec[rank_idx] <= 0:
        continue
      for c_idx in range(self._num_colors):
        if not self._color_tokens[player][c_idx]:
          # player no longer has that color token
          continue
        if self._board_ownership[c_idx][rank_idx] != -1:
          # that color-rank is already claimed
          continue
        # If all checks pass, action is valid
        act = c_idx * self._num_card_types + rank_idx
        actions.append(act)

    # If leading the trick (led_color is None) and trump not broken, 
    # disallow R if there's another color
    if self._led_color is None and not self._trump_broken:
        non_trump_actions = [a for a in actions if (a // self._num_card_types) != 0]
        if non_trump_actions:
            # Remove trump actions
            actions = non_trump_actions

    if not actions:
      # no moves => paradox
      return [_ACTION_PARADOX]
    return sorted(actions)

  def _apply_trick_action(self, action):
    color_idx = action // self._num_card_types
    rank_idx = action % self._num_card_types

    player = self._current_player
    # Remove from hand
    self._hands[player][rank_idx] -= 1
    # Mark board ownership => adjacency
    self._board_ownership[color_idx][rank_idx] = player

    # Check for adjacency reward by finding largest block before this placement
    self._board_ownership[color_idx][rank_idx] = -1  # Temporarily remove
    old_largest = self._largest_cluster_coords_for_player(player)
    self._board_ownership[color_idx][rank_idx] = player  # Restore

    rank_val = rank_idx + 1
    color_str = _COLORS[color_idx]
    self._cards_played_this_trick[player] = (rank_val, color_str)

    # If this is not the lead color, and the chosen color is "R",
    # then we've broken trump.
    if self._led_color is not None and color_str == "R" and color_str != self._led_color:
      self._trump_broken = True

    # If no color led, set it
    if self._led_color is None:
      self._led_color = color_str
    else:
      # If player *did not* follow the lead color => remove lead-color token
      if color_str != self._led_color:
        led_idx = _COLORS.index(self._led_color)
        self._color_tokens[player][led_idx] = False

    # Next
    self._current_player = (self._current_player + 1) % self._num_players
    # If we loop back to start_player => trick ends
    if self._current_player == self._start_player:
      winner = self._evaluate_trick_winner()
      
      self._tricks_won[winner] += 1
      self._trick_number += 1
      self._start_player = winner
      self._current_player = winner
      self._led_color = None
      self._cards_played_this_trick = [None]*self._num_players

      # Check if done
      if self._trick_number >= self._num_tricks:
        self._phase = 4
        self._compute_final_scores()
        self._game_over = True

  def resample_from_infostate(self, player_id, sampler):
    """Returns a new state with all hidden information re-sampled
    consistently with 'player_id's' perspective."""
    # 1) Create a full deep clone of this state (just like 'clone()').
    cloned = self.clone()

    # 2) Identify which cards are KNOWN to player_id:
    #    - The player's own hand is fully known.
    #    - Possibly the discards if your game reveals them? Or partially known?
    #    - The color tokens on the board do not reveal ranks unless your logic does so.
    #    Keep track of these known cards so they don't get re-randomized.

    # Let's gather a list/array of all possible ranks:
    # ranks = range(1, self._game.max_card_value+1)
    # Make a multi-set of how many copies exist globally. For example, 5 copies per rank.

    # 3) Remove from that multi-set all the cards that 'player_id' definitely knows.
    #    This includes your own hand in 'cloned._hands[player_id]'.

    # 4) Also remove from that multi-set any public knowledge. If you reveal discards to everyone,
    #    remove those. If the discards are hidden, they remain part of the unknown distribution.

    # 5) Now we have a set of "unassigned" cards. We also have to figure out how many cards
    #    each other player must still hold (and how many are in the discard) in order to match
    #    the visible counts or other constraints.

    # 6) We'll do a random shuffle of that "unassigned" deck and then distribute them
    #    among the other players and discards, consistent with the known counts.
    #    For example, if we know that player2 has 6 cards left, we assign them 6 random cards
    #    from the unknown set. If we know 1 card was discarded face-down by player2
    #    but we don't know which it was, we choose from the unknown set, etc.

    # This is the main logic you must fill. The goal is that from 'player_id's perspective,
    # everything in 'cloned' is consistent with the partial info they had.

    # 7) Return the final 'cloned' state.
    return cloned


  def _evaluate_trick_winner(self):
    # If at least one red => highest red
    red_plays = [(p, v) for p,(v,c) in enumerate(self._cards_played_this_trick) if c == "R"]
    if red_plays:
      return max(red_plays, key=lambda x: x[1])[0]
    # else highest among led color
    if self._led_color is None:
      # corner case
      all_plays = [(p,v) for p,(v,c) in enumerate(self._cards_played_this_trick)]
      return max(all_plays, key=lambda x: x[1])[0]
    led_plays = [(p,v) for p,(v,c) in enumerate(self._cards_played_this_trick)
                 if c == self._led_color]
    if not led_plays:
      # fallback => highest overall
      all_plays = [(p,v) for p,(v,c) in enumerate(self._cards_played_this_trick)]
      return max(all_plays, key=lambda x: x[1])[0]
    return max(led_plays, key=lambda x: x[1])[0]

  # -------------------------------------------------------------------
  # Paradox
  # -------------------------------------------------------------------
  def _apply_paradox(self):
    player = self._current_player
    self._has_paradoxed[player] = True

    # End game
    self._phase = 4
    self._game_over = True
    self._compute_final_scores()

  # -------------------------------------------------------------------
  # Scoring (Phase 4)
  # -------------------------------------------------------------------
  def _compute_final_scores(self):
    # First compute raw scores
    raw_scores = [0.0] * self._num_players
    for p in range(self._num_players):
      tricks = self._tricks_won[p]
      if self._has_paradoxed[p]:
        # If paradox => score = -(tricks)
        raw_scores[p] = -float(tricks)
      else:
        base = float(tricks)
        pred = self._predictions[p]
        if pred == tricks:
          # get adjacency bonus = largest connected cluster
          cluster_bonus = self._largest_cluster_for_player(p)
          raw_scores[p] = base + cluster_bonus
        else:
          raw_scores[p] = base

    # Apply reward scaling and add to final step rewards
    for p in range(self._num_players):
      final_reward = 5.0 * raw_scores[p]
      self._returns[p] += final_reward
      self._rewards[p] += final_reward  # Add to any existing step reward

  # -------------------------------------------------------------------
  # Adjacency Logic
  # -------------------------------------------------------------------
  def _largest_cluster_for_player(self, player):
    """
    Compute the size of the largest connected group of squares
    for which _board_ownership[color][rank] == player.
    Adjacency = up/down/left/right in (color, rank) grid.
    color in [0.._num_colors-1], rank in [0.._num_card_types-1].
    """
    visited = np.zeros((self._num_colors, self._num_card_types), dtype=bool)
    max_cluster = 0

    def neighbors(c, r):
      for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]:
        cc, rr = c+dc, r+dr
        if 0 <= cc < self._num_colors and 0 <= rr < self._num_card_types:
          yield (cc, rr)

    for c_idx in range(self._num_colors):
      for r_idx in range(self._num_card_types):
        if (self._board_ownership[c_idx][r_idx] == player) and (not visited[c_idx][r_idx]):
          # BFS or DFS to find connected cluster
          size = 0
          queue = deque([(c_idx, r_idx)])
          visited[c_idx][r_idx] = True
          while queue:
            c0, r0 = queue.popleft()
            size += 1
            # check neighbors
            for (c1, r1) in neighbors(c0, r0):
              if not visited[c1][r1] and self._board_ownership[c1][r1] == player:
                visited[c1][r1] = True
                queue.append((c1, r1))

          if size > max_cluster:
            max_cluster = size

    return max_cluster

  def _largest_cluster_coords_for_player(self, player):
    """
    Returns a list of (color_idx, rank_idx) squares that form the
    single largest connected cluster owned by `player`.
    """
    visited = np.zeros((self._num_colors, self._num_card_types), dtype=bool)
    best_cluster = []
    best_size = 0

    def neighbors(c, r):
      for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]:
        cc, rr = c+dc, r+dr
        if 0 <= cc < self._num_colors and 0 <= rr < self._num_card_types:
          yield (cc, rr)

    for c_idx in range(self._num_colors):
      for r_idx in range(self._num_card_types):
        if self._board_ownership[c_idx][r_idx] == player and not visited[c_idx][r_idx]:
          # BFS or DFS to find connected cluster from this square
          queue = [(c_idx, r_idx)]
          visited[c_idx][r_idx] = True
          cluster_coords = []
          while queue:
            c0, r0 = queue.pop()
            cluster_coords.append((c0, r0))
            for (cc, rr) in neighbors(c0, r0):
              if not visited[cc][rr] and self._board_ownership[cc][rr] == player:
                visited[cc][rr] = True
                queue.append((cc, rr))
          # Check if this cluster is bigger than the current best
          if len(cluster_coords) > best_size:
            best_size = len(cluster_coords)
            best_cluster = cluster_coords

    return best_cluster

# ---------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------
class QuantumCatObserver:
  """Observer, conforming to the PyObserver interface."""

  def __init__(self, iig_obs_type, num_players, num_card_types, num_colors, params=None):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    self.iig_obs_type = iig_obs_type
    self.num_players = num_players
    self.num_card_types = num_card_types
    self.num_colors = num_colors

    # Determine which observation pieces we want to include
    pieces = [
        ("current_player", num_players, (num_players,)),
        ("phase", 5, (5,)),  # 0..4
        ("led_color", num_colors + 1, (num_colors + 1,)),  # 1-hot for each color + None
        ("trick_number", 1, (1,)),
        ("start_player", num_players, (num_players,)),
        ("cards_played_in_trick", 2 * num_players, (2 * num_players,)),  # pairs of (rank,color_idx)
        ("predictions", num_players, (num_players,)),
        ("tricks_won", num_players, (num_players,)),
        ("board_ownership", num_colors * num_card_types, (num_colors, num_card_types)),
        ("color_tokens", num_players * num_colors, (num_players, num_colors)),  # Public info for all players
    ]
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("hand", num_card_types, (num_card_types,)))
      pieces.append(("discarded_rank", 1, (1,)))

    # Build the single flat tensor
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, dtype=np.float32)

    # Build the named & reshaped views of the bits of the flat tensor
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def _encode_led_color(self, led_color):
    """
    led_color: 'R', 'B', 'Y', 'G', or None.
    We set one-hot for R/B/Y/G. If None, we set the last slot.
    """
    color_map = {"R": 0, "B": 1, "Y": 2, "G": 3}
    if led_color is None:
      self.dict["led_color"][-1] = 1.0
    else:
      self.dict["led_color"][color_map[led_color]] = 1.0

  def set_from(self, state, player):
    """Updates the observer's data to reflect `state` from the POV of `player`."""
    self.tensor.fill(0)
    cp = state.current_player()
    if cp != pyspiel.PlayerId.TERMINAL and cp != pyspiel.PlayerId.CHANCE:
      self.dict["current_player"][cp] = 1

    if 0 <= state._phase <= 4:
      self.dict["phase"][state._phase] = 1

    # Encode led color
    self._encode_led_color(state._led_color)

    # Store trick info
    self.dict["trick_number"][0] = state._trick_number
    
    # One-hot for start player (if not terminal)
    if not state.is_terminal():
        self.dict["start_player"][state._start_player] = 1.0

    # Cards played in current trick as (rank,color_idx) pairs
    arr = self.dict["cards_played_in_trick"]
    color_map = {"R": 0, "B": 1, "Y": 2, "G": 3}
    for p in range(self.num_players):
        if state._cards_played_this_trick[p] is not None:
            rank_val, color_str = state._cards_played_this_trick[p]
            arr[2*p] = rank_val
            arr[2*p + 1] = color_map[color_str]
        else:
            arr[2*p] = -1  # indicates "no card"
            arr[2*p + 1] = -1

    # Predictions and tricks won
    for p in range(self.num_players):
        self.dict["predictions"][p] = state._predictions[p]
        self.dict["tricks_won"][p] = state._tricks_won[p]

    # Board ownership
    for c_idx in range(state._num_colors):
        for r_idx in range(state._num_card_types):
            self.dict["board_ownership"][c_idx][r_idx] = float(state._board_ownership[c_idx][r_idx])

    # Store all players' color tokens (public info)
    color_tokens_mat = self.dict["color_tokens"]
    for p in range(self.num_players):
        for c_idx in range(self.num_colors):
            color_tokens_mat[p, c_idx] = float(state._color_tokens[p][c_idx])

    # If single-player private info => store your hand
    if self.iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      for i in range(self.num_card_types):
        self.dict["hand"][i] = state._hands[player][i]
      self.dict["discarded_rank"][0] = float(state._discarded_cards[player])

  def string_from(self, state, player):
    """Observation of `state` from the POV of `player`, as a string."""
    pieces = []
    cp = state.current_player()
    if cp != pyspiel.PlayerId.TERMINAL and cp != pyspiel.PlayerId.CHANCE:
      pieces.append(f"p{cp}")
    pieces.append(f"phase={state._phase}")
    if self.iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(f"hand={state._hands[player]}")
      pieces.append(f"my_prediction={state._predictions[player]}")
      pieces.append(f"led_color={state._led_color}")
    
    # Always show all players' color tokens (public info)
    pieces.append("color_tokens=" + "\n".join(
        f"p{p}: {state._color_tokens[p]}" for p in range(self.num_players)
    ))
    
    # Show any partial trick plays by other players
    trick_plays = []
    for p in range(self.num_players):
      card = state._cards_played_this_trick[p]
      if card is not None:
        rank_val, color_str = card
        trick_plays.append(f"p{p}=>{rank_val}{color_str}")
    if trick_plays:
      pieces.append("Trick:" + ",".join(trick_plays))
        
    pieces.append(f"board_ownership=\n{state._board_ownership}")
    return " ".join(str(p) for p in pieces)


pyspiel.register_game(_GAME_TYPE, QuantumCatGame)
