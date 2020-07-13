// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_SPIEL_CONSTANTS_H_
#define OPEN_SPIEL_SPIEL_CONSTANTS_H_

#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// Player ids are 0, 1, 2, ...
// Negative numbers are used for various special values.
enum PlayerId {
  // The fixed player id for chance/nature.
  kChancePlayerId = -1,
  // What is returned as a player id when the game is simultaneous.
  kSimultaneousPlayerId = -2,
  // Invalid player.
  kInvalidPlayer = -3,
  // What is returned as the player id on terminal nodes.
  kTerminalPlayerId = -4
};

// Constant representing an invalid action.
inline constexpr Action kInvalidAction = -1;

enum class StateType {
  kTerminal,  // If the state is terminal.
  kChance,    // If the player to act equals kChanceId.
  kDecision,  // If a player other than kChanceId is acting.
};

// Layouts for 3-D tensors. For 2-D tensors, we assume that the layout is a
// single spatial dimension and a channel dimension. If a 2-D tensor should be
// interpreted as a 2-D space, report it as 3-D with a channel dimension of
// size 1. We have no standard for higher-dimensional tensors.
enum class TensorLayout {
  kHWC,  // indexes are in the order (height, width, channels)
  kCHW,  // indexes are in the order (channels, height, width)
};


}  // namespace open_spiel

#endif  // OPEN_SPIEL_SPIEL_CONSTANTS_H_
