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

#include "open_spiel/algorithms/dqn_torch/dqn.h"

#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/games/efg_game.h"
#include "open_spiel/games/efg_game_data.h"


namespace open_spiel {
namespace algorithms {
namespace torch_dqn {
namespace {

void TestSimpleGame() {
  std::shared_ptr<const Game> game = efg_game::LoadEFGGame(
      efg_game::GetSimpleForkEFGData());
  SPIEL_CHECK_TRUE(game != nullptr);
  DQN dqn(/*use_observation*/game->GetType().provides_observation_tensor,
          /*player_id*/0,
          /*state_representation_size*/game->InformationStateTensorSize(),
          /*num_actions*/game->NumDistinctActions(),
          /*hidden_layers_sizes*/{16},
          /*replay_buffer_capacity*/100,
          /*batch_size*/5,
          /*learning_rate*/0.01,
          /*update_target_network_every*/20,
          /*learn_every*/5,
          /*discount_factor*/1.0,
          /*min_buffer_size_to_learn*/5,
          /*epsilon_start*/0.02,
          /*epsilon_end*/0.01);
  int total_reward = 0;
  std::unique_ptr<State> state;
  for (int i = 0; i < 100; i++) {
    state = game->NewInitialState();
    while (!state->IsTerminal()) {
      open_spiel::Action action = dqn.Step(state);
      state->ApplyAction(action);
      total_reward += state->PlayerReward(0);
    }
    dqn.Step(state);
  }

  SPIEL_CHECK_GE(total_reward, 75);
}

void TestTicTakToe() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  SPIEL_CHECK_TRUE(game != nullptr);
  std::vector<std::unique_ptr<DQN>> agents;
  std::vector<int> hidden_layers = {16};
  for (int i = 0; i < 2; i++) {
    agents.push_back(std::make_unique<DQN>(
        /*use_observation*/game->GetType().provides_observation_tensor,
        /*player_id*/i,
        /*state_representation_size*/game->ObservationTensorSize(),
        /*num_actions*/game->NumDistinctActions(),
        /*hidden_layers_sizes*/hidden_layers,
        /*replay_buffer_capacity*/10,
        /*batch_size*/5,
        /*learning_rate*/0.01,
        /*update_target_network_every*/20,
        /*learn_every*/5,
        /*discount_factor*/1.0,
        /*min_buffer_size_to_learn*/5));
  }
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    Player current_player = state->CurrentPlayer();
    open_spiel::Action action = agents[current_player]->Step(state);
    state->ApplyAction(action);
  }
  for (int i = 0; i < 2; i++) {
    agents[i]->Step(state);
  }
}

void TestHanabi() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tiny_hanabi");
  SPIEL_CHECK_TRUE(game != nullptr);
  std::vector<std::unique_ptr<DQN>> agents;
  std::vector<int> hidden_layers = {16};
  for (int i = 0; i < 2; i++) {
    agents.push_back(std::make_unique<DQN>(
        /*use_observation*/game->GetType().provides_observation_tensor,
        /*player_id*/i,
        /*state_representation_size*/game->InformationStateTensorSize(),
        /*num_actions*/game->NumDistinctActions(),
        /*hidden_layers_sizes*/hidden_layers,
        /*replay_buffer_capacity*/10,
        /*batch_size*/5,
        /*learning_rate*/0.01,
        /*update_target_network_every*/20,
        /*learn_every*/5,
        /*discount_factor*/1.0,
        /*min_buffer_size_to_learn*/5));
  }
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    Player current_player = state->CurrentPlayer();
    open_spiel::Action action;
    for (int i = 0; i < 2; i++) {
      if (i == current_player) {
        action = agents[i]->Step(state);
      } else {
        agents[i]->Step(state);
      }
    }
    state->ApplyAction(action);
  }
  for (int i = 0; i < 2; i++) {
    agents[i]->Step(state);
  }
}

}  // namespace
}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

int main(int args, char** argv) {
  open_spiel::algorithms::torch_dqn::TestSimpleGame();
  open_spiel::algorithms::torch_dqn::TestTicTakToe();
  open_spiel::algorithms::torch_dqn::TestHanabi();
  return 0;
}
