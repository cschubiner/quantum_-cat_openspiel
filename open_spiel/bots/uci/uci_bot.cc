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

#include "uci_bot.h"

#include <sys/wait.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/ioctl.h>

namespace open_spiel {
namespace uci {

UciBot::UciBot(const std::string &path,
               int move_time,
               bool ponder,
               const Options &options) :
    ponder_(ponder) {

  SPIEL_CHECK_GT(move_time, 0);
  SPIEL_CHECK_GT(path.size(), 0);
  move_time_ = move_time;

  StartProcess(path);
  Uci();
  for (auto const& [name, value] : options)
  {
    SetOption(name, value);
  }
  IsReady();
  UciNewGame();
}

UciBot::~UciBot() {
  Quit();
  int status;
  while (waitpid(pid_, &status, 0) == -1);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "Uci sub-process failed" << std::endl;
  }
}

Action UciBot::Step(const State &state) {
  std::string move_str;
  auto chess_state = down_cast<const chess::ChessState&>(state);
  if (ponder_ && ponder_move_) {
    if (!was_ponder_hit_) {
      Stop();
      Position(chess_state.Board().ToFEN());
      tie(move_str, ponder_move_) = Go();
    } else {
      tie(move_str, ponder_move_) = ReadBestMove();
    }
  } else {
    Position(chess_state.Board().ToFEN());
    tie(move_str, ponder_move_) = Go();
  }
  was_ponder_hit_ = false;
  auto move = chess_state.Board().ParseLANMove(move_str);
  if (!move) {
    SpielFatalError("Uci sub-process returned an illegal or invalid move");
  }

  if (ponder_ && ponder_move_) {
    Position(chess_state.Board().ToFEN(), {move_str, *ponder_move_});
    GoPonder();
  }

  Action action = chess::MoveToAction(*move);
  return action;
}

void UciBot::Restart() {
  ponder_move_ = std::nullopt;
  was_ponder_hit_ = false;
  UciNewGame();
}

void UciBot::RestartAt(const State&state) {
  ponder_move_ = std::nullopt;
  was_ponder_hit_ = false;
  auto chess_state = down_cast<const chess::ChessState&>(state);
  Position(chess_state.Board().ToFEN());
}

void UciBot::InformAction(const State &state,
                          Player player_id,
                          Action action) {
  auto chess_state = down_cast<const chess::ChessState&>(state);
  chess::Move move = chess::ActionToMove(action, chess_state.Board());
  std::string move_str = move.ToLAN();
  if (ponder_ && move_str == ponder_move_) {
    PonderHit();
    was_ponder_hit_ = true;
  }
}

void UciBot::StartProcess(const std::string &path) {

  int output_pipe[2];
  int input_pipe[2];

  if (pipe(output_pipe) || pipe(input_pipe)) {
    SpielFatalError("Creating pipes failed");
  }

  pid_ = fork();
  if (pid_ < 0) {
    SpielFatalError("Forking failed");
  }

  if (pid_) { // parent
    close(output_pipe[0]);
    close(input_pipe[1]);

    output_fd_ = output_pipe[1];
    input_fd_ = input_pipe[0];
    //Read();
  } else { // child
    dup2(output_pipe[0], STDIN_FILENO);
    dup2(input_pipe[1], STDOUT_FILENO);
    dup2(input_pipe[1], STDERR_FILENO);

    close(output_pipe[1]);
    close(input_pipe[0]);

    std::cerr << "neco" << std::endl;
    execlp(path.c_str(), path.c_str(), (char*) nullptr);
    std::cerr << "neco" << std::endl;
    SpielFatalError("Executing uci bot sub-process failed");
  }

}

void UciBot::Uci() {
  Write("uci");
  while (true) {
    auto response = Read(false);
    std::istringstream response_stream(response);
    std::string line;
    while (getline(response_stream, line)) {
      // skip id and option lines
      if (line.rfind("uciok", 0) == 0) {
        return;
      }
    }
  }

}

void UciBot::SetOption(const std::string&name, const std::string&value) {
  std::string msg = "setoption name " + name + " value " + value;
  Write(msg);
}

void UciBot::UciNewGame() {
  Write("ucinewgame");
}

void UciBot::IsReady() {
  Write("isready");

  while (true) {
    auto response = Read(false);
    std::istringstream response_stream(response);
    std::string line;
    while (getline(response_stream, line)) {
      // skip id and option lines
      if (line.rfind("readyok", 0) == 0) {
        return;
      }
    }
  }
}

void UciBot::Position(const std::string &fen,
                      const std::vector<std::string> &moves) {
  std::string msg = "position fen " + fen;

  std::string moves_str = absl::StrJoin(moves, " ");
  if (!moves_str.empty()) {
    msg += " moves " + moves_str;
  }
  Write(msg);
}

std::pair<std::string, std::optional<std::string>> UciBot::Go() {
  Write("go movetime " + std::to_string(move_time_));
  return ReadBestMove();
}

void UciBot::GoPonder() {
  Write("go ponder movetime " + std::to_string(move_time_));
}

void UciBot::PonderHit() {
  Write("ponderhit");
}

std::pair<std::string, std::optional<std::string>> UciBot::Stop() {
  Write("stop");
  return ReadBestMove();
}

void UciBot::Quit() {
  Write("quit");
}

std::pair<std::string, std::optional<std::string>> UciBot::ReadBestMove() {
  while (true) {
    auto response = Read(true);
    std::istringstream response_stream(response);
    std::string line;
    while (getline(response_stream, line)) {
      std::istringstream line_stream(line);
      std::string token;
      std::string move_str;
      std::optional<std::string> ponder_str = std::nullopt;
      line_stream >> std::skipws;
      while (line_stream >> token) {
        if (token == "bestmove") {
          line_stream >> move_str;
        } else if (token == "ponder") {
          line_stream >> token;
          ponder_str = token;
        }
      }
      if (!move_str.empty()) {
        return std::make_pair(move_str, ponder_str);
      }
    }
  }
  SpielFatalError("Wrong response to go");
}

void UciBot::Write(const std::string &msg) const {
  if (write(output_fd_, (msg + "\n").c_str(), msg.size() + 1) != msg.size() + 1) {
    SpielFatalError("Sending a command to uci sub-process failed");
  }
}

std::string UciBot::Read(bool wait) const {
  char *buff;
  int count = 0;
  std::string response;

  struct timeval timeout = {1, 0};

  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(input_fd_, &fds);

  int ready_fd = select(input_fd_ + 1, &fds, nullptr, nullptr,
                        wait ? nullptr : &timeout);

  if (ready_fd == -1) {
    SpielFatalError("Failed to read from uci sub-process");
  }
  if (ready_fd == 0) {
    SpielFatalError("Response from uci sub-process not received on time");
  }

  if (ioctl(input_fd_, FIONREAD, &count) != -1) {

    buff = (char*) malloc(count);
    if (read(input_fd_, buff, count) != count) {
      SpielFatalError("Read wrong number of bytes");
    }
    response = buff;
    free(buff);
  } else {
    SpielFatalError("Failed to read input size.");
  }
  return response;
}

std::unique_ptr<Bot> uci::MakeUciBot(const std::string &path,
                                     int move_time,
                                     bool ponder,
                                     const Options &options) {
  return std::make_unique<UciBot>(path, move_time, ponder, options);
}
}  // namespace uci
}  // namespace open_spiel
