#!/usr/bin/env python3
"""Tests for evaluate_quantum_cat helpers."""

from absl.testing import absltest
import pyspiel

from evaluate_quantum_cat import make_policy
from evaluate_quantum_cat import pick_follow_suit_action
from open_spiel.python.games import quantum_cat  # pylint:disable=unused-import


class EvaluateQuantumCatTest(absltest.TestCase):

  def test_pick_follow_suit_action_uses_real_card_type_count(self):
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    legal = [
        color * game.num_card_types
        for color in range(4)
    ]

    for led_color in range(4):
      obs = [0.0] * 64
      led_color_start = game.num_players() + 5
      obs[led_color_start + led_color] = 1.0

      picks = {
          pick_follow_suit_action(
              legal, obs, game.num_card_types, game.num_players())
          for _ in range(20)
      }

      self.assertEqual(picks, {led_color * game.num_card_types})

  def test_follow_suit_bot_uses_game_num_card_types(self):
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    bot = make_policy("follow_suit", game, player_id=0)

    self.assertEqual(bot._num_card_types, game.num_card_types)  # pylint:disable=protected-access
    self.assertNotEqual(bot._num_card_types, game.num_distinct_actions() // 4)  # pylint:disable=protected-access


if __name__ == "__main__":
  absltest.main()
