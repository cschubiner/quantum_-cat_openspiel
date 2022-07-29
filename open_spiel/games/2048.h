// Copyright 2022 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_2048_H_
#define OPEN_SPIEL_GAMES_2048_H_

// Implementation of the board game Checkers.
// https://en.wikipedia.org/wiki/Checkers
//
// Some notes about this implementation:
// - Capturing:
//     When capturing an opponent's piece is possible, capturing is mandatory
//     in this implementation.
// - Drawing:
//     Game is drawn if no pieces have been removed in 40 moves
//     http://www.flyordie.com/games/help/checkers/en/games_rules_checkers.html
// - Custom board dimensions:
//     Dimensions of the board can be customised by calling the
//     TwoZeroFourEightState(rows, columns) constructer with the desired
//     number of rows and columns

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace two_zero_four_eight {

constexpr int kNumPlayers = 1;
constexpr int kDefaultRows = 4;
constexpr int kDefaultColumns = 4;
// 2 & 4
constexpr int kNumChanceTiles = 2;
constexpr int kNoCellAvailableAction = kDefaultRows * kDefaultColumns * 2;

struct Coordinate {
  int x, y;
  Coordinate(int _x, int _y)
      : x(_x), y(_y) {}
};

struct ChanceAction {
  int row;
  int column;
  bool is_four;
  ChanceAction(int _row, int _column, bool _is_four)
      : row(_row),
        column(_column),
        is_four(_is_four) {}
};

struct Tile {
  int value;
  bool is_merged;
  Tile(int _value, bool _is_merged)
      : value(_value),
        is_merged(_is_merged) {}
};

// State of an in-play game.
class TwoZeroFourEightState : public State {
 public:
  explicit TwoZeroFourEightState(std::shared_ptr<const Game> game);
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new TwoZeroFourEightState(*this));
  }
  void UndoAction(Player player, Action action) override;
  bool InBounds(int row, int column) const;
  void SetCustomBoard(const std::vector<int> board_seq);
  ChanceAction SpielActionToChanceAction(Action action) const;
  Action ChanceActionToSpielAction(ChanceAction move) const;
  void SetBoard(int row, int column, Tile tile) {
    board_[row * kDefaultColumns + column] = tile;
  }
  Tile BoardAt(int row, int column) const {
    return board_[row * kDefaultColumns + column];
  }
  std::vector<Action> LegalActions() const override;
  ActionsAndProbs ChanceOutcomes() const override;  
  int AvailableCellCount() const;
  std::vector<std::vector<int>> BuildTraversals (int direction) const;
  bool WithinBounds(int x, int y) const;
  bool CellAvailable(int x, int y) const;
  std::vector<int> FindFarthestPosition(int x, int y, int direction) const;
  bool TileMatchesAvailable() const;
  bool Reached2048() const;
  void PrepareTiles();
  int GetCellContent(int x, int y) const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  Player current_player_ = kChancePlayerId;  // Player zero (White, 'o') goes first.
  std::vector<Tile> board_;
};

// Game object.
class TwoZeroFourEightGame : public Game {
 public:
  explicit TwoZeroFourEightGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<TwoZeroFourEightState>(shared_from_this());
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kDefaultRows, kDefaultColumns};
  }
  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 1000; }
  int MaxChanceOutcomes() const override { 
    return kDefaultRows * kDefaultColumns * 2 + 1;
  }
};

}  // namespace two_zero_four_eight
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CHECKERS_H_
