const state = { game: null, busy: false };

const $ = (id) => document.getElementById(id);

function colorValue(code) {
  return {
    R: "#c83f34",
    B: "#2876aa",
    Y: "#b28413",
    G: "#3f9164",
  }[code] || "#8b9287";
}

function validSetupValues() {
  const humans = Number($("humanPlayers").value);
  const bots = Number($("botPlayers").value);
  const total = humans + bots;
  return humans >= 1 && humans <= 2 && bots >= 0 && bots <= 5 && total >= 2 && total <= 5;
}

function validateSetup() {
  const warning = $("warning");
  const submit = $("setup").querySelector("button");
  if (!validSetupValues()) {
    warning.hidden = false;
    warning.textContent = "Quantum Cat supports 2 to 5 total seats in this engine.";
    submit.disabled = true;
    return;
  }
  warning.hidden = true;
  submit.disabled = state.busy;
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.error) {
    throw new Error(data.error || `Request failed: ${res.status}`);
  }
  return data;
}

async function newGame(event) {
  event.preventDefault();
  if (state.busy) return;
  try {
    setBusy(true);
    state.game = await postJson("/api/new", {
      humanPlayers: Number($("humanPlayers").value),
      botPlayers: Number($("botPlayers").value),
      botMode: $("botMode").value,
      seed: Number($("seed").value),
    });
    clearWarning();
    render();
  } catch (error) {
    showWarning(error.message);
  } finally {
    setBusy(false);
  }
}

async function chooseAction(action) {
  if (!state.game || state.busy) return;
  try {
    setBusy(true);
    state.game = await postJson("/api/action", {
      sessionId: state.game.sessionId,
      action,
    });
    clearWarning();
    render();
  } catch (error) {
    showWarning(error.message);
  } finally {
    setBusy(false);
  }
}

function showWarning(message) {
  const warning = $("warning");
  warning.hidden = false;
  warning.textContent = message;
}

function clearWarning() {
  const warning = $("warning");
  warning.hidden = true;
  warning.textContent = "";
}

function setBusy(value) {
  state.busy = value;
  document.querySelectorAll("[data-action]").forEach((node) => {
    node.disabled = value;
  });
  $("setup").querySelector("button").disabled = value || !validSetupValues();
}

function render() {
  const game = state.game;
  if (!game) return;
  document.documentElement.style.setProperty("--rank-count", game.rankCount);
  document.body.dataset.phase = game.phase;
  document.body.dataset.terminal = game.terminal ? "true" : "false";
  $("phase").textContent = game.phase;
  $("turn").textContent = game.terminal
    ? "Finished"
    : `P${game.currentPlayer} ${game.currentRole}`;
  $("trick").textContent = `${game.summary.trickNumber}/${game.summary.numTricks}`;
  $("lead").textContent = game.summary.ledColor || "none";
  renderTask(game);
  renderStepper(game);
  renderBotBadge(game);
  renderSeats(game);
  renderBoard(game);
  renderActions(game);
  renderTrick(game);
  renderLog(game);
}

function renderTask(game) {
  const title = $("taskTitle");
  const hint = $("taskHint");
  if (game.terminal) {
    title.textContent = "Round finished";
    hint.textContent = "Final returns are in the player strip.";
    return;
  }
  if (game.currentRole !== "human") {
    title.textContent = "Bots are thinking";
    hint.textContent = "The table advances automatically after bot decisions.";
    return;
  }
  const copy = {
    discard: ["Discard one rank", "Choose a rank from your hand to discard face down."],
    prediction: ["Choose your bid", "Predict how many tricks you expect to win."],
    play: ["Choose one legal play", "Play a rank as one of the available colors, or paradox if forced."],
    dealing: ["Dealing", "Cards are being dealt automatically."],
    scoring: ["Round finished", "Final returns are in the player strip."],
  }[game.phase] || ["Your turn", "Choose a legal move to continue."];
  title.textContent = copy[0];
  hint.textContent = copy[1];
}

function renderStepper(game) {
  document.querySelectorAll(".step").forEach((step) => {
    step.classList.toggle("active", step.dataset.step === game.phase);
  });
}

function renderBotBadge(game) {
  const statuses = Object.values(game.botStatus || {});
  if (!statuses.length) {
    $("botBadge").textContent = "Human-only table";
    return;
  }
  const first = statuses[0] || {};
  $("botBadge").textContent = formatBotStatus(first);
}

function formatBotStatus(status) {
  const name = status.name || "bot";
  const source = status.source || "";
  if (name === "champion_belief_policy") {
    return "Champion belief policy · best confirmed 3-player bot, rating 1203.77";
  }
  if (source.startsWith("Fallback")) {
    return `${name} · compatible fallback for this player count`;
  }
  return `${name} · ${source || "configured bot"}`;
}

function renderSeats(game) {
  $("seats").innerHTML = game.players.map((player) => {
    const prediction = player.prediction == null ? "-" : player.prediction;
    const score = player.return == null ? "-" : player.return;
    const tokenColor = colorValue(game.colors[player.seat % game.colors.length].code);
    const tokens = player.tokens.map((on, idx) => {
      const code = game.colors[idx].code;
      return `<span class="token ${on ? "" : "off"}" title="${game.colors[idx].name}" style="background:${colorValue(code)}"></span>`;
    }).join("");
    return `
      <article class="seat ${player.isCurrent ? "current" : ""}">
        <div class="seat-player">
          <span class="player-disc" style="--player-color:${tokenColor}">P${player.seat}</span>
          <div>
            <div class="seat-name">${player.seat === 0 ? "You" : `P${player.seat}`}</div>
            <div class="role">${player.role}${player.paradoxed ? " · paradox" : ""}</div>
          </div>
        </div>
        <div class="seat-metrics">
          <div><span>Tricks</span><strong>${player.tricks}</strong></div>
          <div><span>Bid</span><strong>${prediction}</strong></div>
          <div><span>Score</span><strong>${score}</strong></div>
        </div>
        <div class="tokens">${tokens}</div>
      </article>
    `;
  }).join("");
}

function renderBoard(game) {
  const boardHint = document.querySelector(".board-heading p");
  if (game.phase === "play") {
    boardHint.textContent = "Green outlines are legal board plays for the current human.";
  } else if (game.terminal || game.phase === "scoring") {
    boardHint.textContent = "Final public claims for this round.";
  } else {
    boardHint.textContent = "Board claims unlock during the play phase.";
  }
  const rankHeader = `<div class="board-ranks">
    <div></div>
    ${Array.from({ length: game.rankCount }, (_, idx) => `<span>${idx + 1}</span>`).join("")}
  </div>`;
  $("board").innerHTML = rankHeader + game.board.map((row, idx) => {
    const color = game.colors[idx];
    const cells = row.map((cell) => {
      const legalBoardPlay = game.phase === "play" && cell.legalAction != null;
      const classes = [
        "cell",
        legalBoardPlay ? "legal" : "",
        cell.owner != null ? "occupied" : "",
        cell.blocked ? "blocked" : "",
      ].filter(Boolean).join(" ");
      const style = cell.owner != null ? `style="background:${colorValue(cell.color)}"` : "";
      const click = legalBoardPlay ? `data-action="${cell.legalAction}"` : "";
      const owner = cell.blocked ? "blocked" : (cell.owner != null ? `P${cell.owner}` : "");
      return `<button class="${classes}" ${style} ${click} ${legalBoardPlay ? "" : "disabled"}>
        <span class="rank">${cell.rank}</span>
        <span class="owner">${owner}</span>
      </button>`;
    }).join("");
    return `<div class="board-row">
      <div class="row-label" style="background:${colorValue(color.code)}">${color.code}</div>
      ${cells}
    </div>`;
  }).join("");
  $("board").querySelectorAll("[data-action]").forEach((node) => {
    node.addEventListener("click", () => chooseAction(Number(node.dataset.action)));
  });
}

function renderActions(game) {
  const active = game.players.find((player) => player.seat === game.activeHuman);
  if (!active) {
    $("activeHand").innerHTML = "";
    $("actions").innerHTML = game.terminal
      ? `<div class="empty-action">Final returns are shown in the player strip.</div>`
      : `<div class="empty-action">Waiting for bots.</div>`;
    return;
  }
  $("activeHand").innerHTML = active.hand.map((count, idx) => `
    <div class="hand-rank ${count === 0 ? "empty" : ""}">
      <div><strong>${idx + 1}</strong><span>${count} held</span></div>
    </div>
  `).join("");

  const grouped = groupActions(game.actions);
  $("actions").innerHTML = grouped.map((group) => `
    <div class="move-group">
      <div class="move-group-title">${group.title}</div>
      <div class="move-grid">
        ${group.actions.map((action) => {
          const cls = action.kind === "play" ? `play-${action.color}` : action.kind;
          return `<button class="move-button ${cls}" data-action="${action.id}" ${state.busy ? "disabled" : ""}>${actionMarkup(action)}</button>`;
        }).join("")}
      </div>
    </div>
  `).join("");
  $("actions").querySelectorAll("[data-action]").forEach((node) => {
    node.addEventListener("click", () => chooseAction(Number(node.dataset.action)));
  });
}

function groupActions(actions) {
  if (!actions.length) return [];
  if (actions[0].kind !== "play") {
    return [{ title: actions[0].kind === "prediction" ? "Bids" : "Ranks", actions }];
  }
  const labels = { R: "Red plays", B: "Blue plays", Y: "Yellow plays", G: "Green plays" };
  const order = ["R", "B", "Y", "G"];
  const groups = order.map((color) => ({
    title: labels[color],
    actions: actions.filter((action) => action.color === color),
  })).filter((group) => group.actions.length);
  const paradox = actions.filter((action) => action.kind === "paradox");
  if (paradox.length) groups.push({ title: "Forced", actions: paradox });
  return groups;
}

function conciseAction(action) {
  if (action.kind === "discard") return `Discard ${action.rank}`;
  if (action.kind === "prediction") return `Bid ${action.prediction}`;
  if (action.kind === "play") return `${action.color}${action.rank}`;
  if (action.kind === "paradox") return "Paradox";
  return action.label;
}

function actionMarkup(action) {
  if (action.kind === "discard") {
    return `<span>Discard</span><strong>${action.rank}</strong>`;
  }
  if (action.kind === "prediction") {
    return `<span>Bid</span><strong>${action.prediction}</strong>`;
  }
  if (action.kind === "play") {
    return `<span>${action.color}</span><strong>${action.rank}</strong>`;
  }
  if (action.kind === "paradox") {
    return `<strong>Paradox</strong>`;
  }
  return `<strong>${action.label}</strong>`;
}

function renderTrick(game) {
  $("currentTrick").innerHTML = game.currentTrick.map((play) => {
    const card = play.card ? `${play.card.color}${play.card.rank}` : "-";
    return `<div class="trick-card">P${play.seat}: <strong>${card}</strong></div>`;
  }).join("");
}

function renderLog(game) {
  $("log").innerHTML = game.log.slice().reverse().map((entry) => (
    `<div class="log-entry ${entry.kind}">${entry.text}</div>`
  )).join("");
}

$("setup").addEventListener("submit", newGame);
["humanPlayers", "botPlayers"].forEach((id) => {
  $(id).addEventListener("input", validateSetup);
});
validateSetup();
$("setup").requestSubmit();
