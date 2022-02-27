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

#include "open_spiel/games/phantom_go.h"

#include "open_spiel/games/phantom_go/phantom_go_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace phantom_go {
namespace {

namespace testing = open_spiel::testing;

constexpr int kBoardSize = 9;
constexpr float kKomi = 7.5;

void BasicGoTests() {
    GameParameters params;
    params["board_size"] = GameParameter(9);

    testing::LoadGameTest("phantom_go");
    testing::NoChanceOutcomesTest(*LoadGame("phantom_go"));
    testing::RandomSimTest(*LoadGame("phantom_go", params), 1);
    testing::RandomSimTestWithUndo(*LoadGame("phantom_go", params), 1);
}

void CloneTest() {
    GameParameters params;
    params["board_size"] = GameParameter(kBoardSize);
    std::shared_ptr<const Game> game =
        LoadGame("phantom_go", params);
    PhantomGoState state(game, kBoardSize, kKomi, 0);
    state.ApplyAction(5);

    std::unique_ptr<State> stateClone = state.Clone();

    SPIEL_CHECK_EQ(state.ToString(), stateClone->ToString());
    SPIEL_CHECK_EQ(state.History(), stateClone->History());

    state.ApplyAction(8);

    SPIEL_CHECK_FALSE(state.ToString() == stateClone->ToString());
    SPIEL_CHECK_FALSE(state.History() == stateClone->History());
}

void HandicapTest() {
    std::shared_ptr<const Game> game =
        LoadGame("phantom_go", {{"board_size", open_spiel::GameParameter(kBoardSize)},
                                {"komi", open_spiel::GameParameter(kKomi)},
                                {"handicap", open_spiel::GameParameter(1)}});
    PhantomGoState state(game, kBoardSize, kKomi, 2);
    SPIEL_CHECK_EQ(state.CurrentPlayer(), ColorToPlayer(GoColor::kWhite));
    SPIEL_CHECK_EQ(state.board().PointColor(MakePoint("d4")), GoColor::kBlack);

}

void IllegalMoveTest() {
    GameParameters params;
    params["board_size"] = GameParameter(kBoardSize);
    std::shared_ptr<const Game> game =
        LoadGame("phantom_go", params);
    PhantomGoState state(game, kBoardSize, kKomi, 0);
    SPIEL_CHECK_EQ(state.CurrentPlayer(), ColorToPlayer(GoColor::kBlack));
    state.ApplyAction(5);
    SPIEL_CHECK_EQ(state.CurrentPlayer(), ColorToPlayer(GoColor::kWhite));
    state.ApplyAction(5);
    SPIEL_CHECK_EQ(state.CurrentPlayer(), ColorToPlayer(GoColor::kWhite));
}

void StoneCountTest() {
    GameParameters params;
    params["board_size"] = GameParameter(kBoardSize);
    std::shared_ptr<const Game> game =
        LoadGame("phantom_go", params);
    PhantomGoState state(game, kBoardSize, kKomi, 0);
    SPIEL_CHECK_EQ(state.board().GetStoneCount()[(uint8_t) GoColor::kBlack], 0);
    SPIEL_CHECK_EQ(state.board().GetStoneCount()[(uint8_t) GoColor::kWhite], 0);
    state.ApplyAction(5);
    SPIEL_CHECK_EQ(state.board().GetStoneCount()[(uint8_t) GoColor::kBlack], 1);
    SPIEL_CHECK_EQ(state.board().GetStoneCount()[(uint8_t) GoColor::kWhite], 0);
    state.ApplyAction(6);
    SPIEL_CHECK_EQ(state.board().GetStoneCount()[(uint8_t) GoColor::kBlack], 1);
    SPIEL_CHECK_EQ(state.board().GetStoneCount()[(uint8_t) GoColor::kWhite], 1);

}

void ConcreteActionsAreUsedInTheAPI() {
    std::shared_ptr<const Game> game =
        LoadGame("phantom_go", {{"board_size", open_spiel::GameParameter(kBoardSize)}});
    std::unique_ptr<State> state = game->NewInitialState();

    SPIEL_CHECK_EQ(state->NumDistinctActions(), kBoardSize * kBoardSize + 1);
    SPIEL_CHECK_EQ(state->LegalActions().size(), state->NumDistinctActions());
    for (Action action: state->LegalActions()) {
        SPIEL_CHECK_GE(action, 0);
        SPIEL_CHECK_LE(action, kBoardSize * kBoardSize);
    }
}

//This is a test, that was used to visually analyze resampling
void ResampleFromInfostateVisualTest() {
    std::cout << "Starting ResampleFromMetaposition visual Test\n";
    GameParameters params;
    params["board_size"] = GameParameter(kBoardSize);
    std::shared_ptr<const Game> game =
        LoadGame("phantom_go", params);
    PhantomGoState state(game, kBoardSize, kKomi, 0);

    for (int i = 0; i < 150; i++) {
        std::vector<Action> actions = state.LegalActions();
        std::shuffle(actions.begin(), actions.end(), std::mt19937(std::random_device()()));
        state.ApplyAction(actions[0]);
        if (state.IsTerminal()) {
            break;
        }
    }

    std::unique_ptr<State> resampleState = state.ResampleFromMetaposition(0, nullptr);

    std::cout << "Original state\n" << state.ToString();

    std::cout << "Resampled state\n " << resampleState->ToString();

    /*for(int i = 0; i < state.FullHistory().size(); i++)
    {
        std::cout << state.ActionToString(state.FullHistory()[i].player, state.FullHistory()[i].action) << " " <<
            state.ActionToString(resampleState->FullHistory()[i].player, resampleState->FullHistory()[i].action) << "\n";
    }*/
}

//This test was used to test metaposition resampling on large ammounts of states
//  with different lengths
void ResampleFromInfostateForceTest() {
    std::cout << "Starting ResampleFromMetaposition visual Test\n";
    GameParameters params;
    params["board_size"] = GameParameter(kBoardSize);
    /*std::shared_ptr<const Game> game =
        LoadGame("phantom_go", params);
    PhantomGoState state(game, kBoardSize, kKomi, 0);*/

    for (int n = 10; n < 20; n++) {
        std::cout << "Starting test for n " << n << "\n";
        for (int x = 0; x < 2000; x++) {
            std::shared_ptr<const Game> game =
                LoadGame("phantom_go", params);
            PhantomGoState state(game, kBoardSize, kKomi, 0);

            for (int i = 0; i < n * 10; i++) {
                if (state.IsTerminal()) {
                    state.UndoAction(-1, -1);
                    break;
                }
                std::vector<Action> actions = state.LegalActions();
                std::shuffle(actions.begin(), actions.end(), std::mt19937(std::random_device()()));
                for (long action: actions) {

                    if (action != VirtualActionToAction(kVirtualPass, kBoardSize)) {
                        state.ApplyAction(action);
                        break;
                    }
                }

            }
            std::unique_ptr<State> resampleState = state.ResampleFromMetaposition(state.CurrentPlayer(), nullptr);

        }
    }
}

}  // namespace
}  // namespace phantom_go
}  // namespace open_spiel

int main(int argc, char **argv) {
    open_spiel::phantom_go::CloneTest();
    open_spiel::phantom_go::BasicGoTests();
    open_spiel::phantom_go::HandicapTest();
    open_spiel::phantom_go::ConcreteActionsAreUsedInTheAPI();
    open_spiel::phantom_go::IllegalMoveTest();
    open_spiel::phantom_go::StoneCountTest();
    //open_spiel::phantom_go::ResampleFromInfostateVisualTest();
    //open_spiel::phantom_go::ResampleFromInfostateForceTest();

}
