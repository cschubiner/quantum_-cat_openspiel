// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/yacht/yacht.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace yacht {
namespace {

const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    std::pair<Action, double>(1, 1.0 / 6),
    std::pair<Action, double>(2, 1.0 / 6),
    std::pair<Action, double>(3, 1.0 / 6),
    std::pair<Action, double>(4, 1.0 / 6),
    std::pair<Action, double>(5, 1.0 / 6),
    std::pair<Action, double>(6, 1.0 / 6),
};

const std::vector<int> kChanceOutcomeValues = {1, 2, 3, 4, 5, 6};

// Facts about the game
const GameType kGameType{/*short_name=*/"yacht",
                         /*long_name=*/"Yacht",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*min_num_players=*/2,
                         /*max_num_players=*/2,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new YachtGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

std::string CurPlayerToString(Player cur_player) { return "Some dice"; }

std::string PositionToStringHumanReadable(int pos) { return "Pos"; }

std::string YachtState::ActionToString(Player player, Action move_id) const {
  return "actionToString";
}

std::string YachtState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

YachtState::YachtState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId),
      prev_player_(kChancePlayerId),
      turns_(-1),
      dice_({}),
      scores_({0, 0}),
      scoring_sheets_({ScoringSheet(), ScoringSheet()}) {}

Player YachtState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

int YachtState::Opponent(int player) const { return 1 - player; }

void YachtState::RollDie(int outcome) {
  dice_.push_back(kChanceOutcomeValues[outcome - 1]);
}

int YachtState::DiceValue(int i) const {
  SPIEL_CHECK_GE(i, 0);
  SPIEL_CHECK_LT(i, dice_.size());

  if (dice_[i] >= 1 && dice_[i] <= 6) {
    return dice_[i];
  } else if (dice_[i] >= 7 && dice_[i] <= 12) {
    // This die is marked as chosen, so return its proper value.
    // Note: dice are only marked as chosen during the legal moves enumeration.
    return dice_[i] - 6;
  } else {
    SpielFatalError(absl::StrCat("Bad dice value: ", dice_[i]));
  }
}

void YachtState::DoApplyAction(Action move) {
  // Apply Action
  int i = 0;
  i++;
}

void YachtState::UndoAction(int player, Action action) {
  // Probably delete this. No undo's in yacht.
  int i = 0;
  i++;
}

Action YachtState::EncodedBarMove() const { return 24; }

Action YachtState::EncodedPassMove() const { return 25; }

bool YachtState::IsPosInHome(int player, int pos) const { return true; }

int YachtState::HighestUsableDiceOutcome() const {
  if (UsableDiceOutcome(dice_[1])) {
    return dice_[1];
  } else if (UsableDiceOutcome(dice_[0])) {
    return dice_[0];
  } else {
    return -1;
  }
}

bool YachtState::UsableDiceOutcome(int outcome) const {
  return (outcome >= 1 && outcome <= 6);
}

std::string YachtState::DiceToString(int outcome) const {
  if (outcome > 6) {
    return std::to_string(outcome - 6) + "u";
  } else {
    return std::to_string(outcome);
  }
}

std::vector<Action> YachtState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};
  return {};
}

std::vector<std::pair<Action, double>> YachtState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  if (turns_ == -1) {
    // Doubles not allowed for the initial roll to determine who goes first.
    // Range 0-14: X goes first, range 15-29: O goes first.
    std::vector<std::pair<Action, double>> outcomes;
    outcomes.reserve(30);
    const double uniform_prob = 1.0 / 30.0;
    for (Action action = 0; action < 30; ++action) {
      outcomes.push_back({action, uniform_prob});
    }
    return outcomes;
  } else {
    return kChanceOutcomes;
  }
}

std::string YachtState::ToString() const { return "haha dice: 1 2 3 4 5"; }

bool YachtState::IsTerminal() const { return true; }

std::vector<double> YachtState::Returns() const { return {1, 0}; }

std::unique_ptr<State> YachtState::Clone() const {
  return std::unique_ptr<State>(new YachtState(*this));
}

void YachtState::SetState(int cur_player, const std::vector<int>& dice,
                          const std::vector<int>& scores,
                          const std::vector<ScoringSheet>& scoring_sheets) {
  cur_player_ = cur_player;
  dice_ = dice;
  scores_ = scores;
  scoring_sheets_ = scoring_sheets;
}

YachtGame::YachtGame(const GameParameters& params) : Game(kGameType, params) {}

}  // namespace yacht
}  // namespace open_spiel
