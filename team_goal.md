# Cat in the Box ML Goal (auto-refreshed; canonical filename: team_goal.md)

<!-- refreshers: update both team_goal.md and GOAL.md while preserving their existing independent historical entries; do not blindly overwrite either file. -->

Last refreshed: 2026-06-15 (source of truth: EXPERIMENTS.md through Run977)

## Mission

Train ML agents for the FULL base game of Cat in the Box (all cards, real
rules) that match or exceed human performance at **3, 4, AND 5 players**,
judged by Elo - which you must define, anchor, and defend - using only
information-legal play: the bot never reads hidden cards or opponents' hands
at decision time. (Exact-oracle labels computed from the simulator at
TRAINING time are allowed; hidden-information features at INFERENCE time are
not.)

## Binding objective correction - Run1035, 2026-08-19

Official full-match placement is the competitive objective. Promote only when
an information-legal candidate defeats the exact frozen incumbent on
seat-balanced common deals, with independent-deal placement-score confidence
above the neutral 0.5 rate, no significant official-score regression, and no
significant increase in the candidate's personal paradox rate.

A paradox penalizes only the triggering player. Opponent paradoxes may improve
competitive placement, so table-wide any-player paradox frequency is
diagnostic only and must never veto promotion. The historical <40% table-wide
targets and sequencing statements preserved below are superseded wherever they
conflict with this section. Evaluate 3p, 4p, and 5p independently; no 3p
table-survival threshold blocks competitive work at another player count.

The primary inference unit is a distinct chance deal. Seat rotations and
multiple opponent comparisons within one deal are correlated, not independent
evidence. Retain exact led-suit discipline, deterministic human tactical
fixtures, information safety, and recorded-human Elo requirements.

## Binding tactical evaluation correction - Run1036, 2026-08-19

Every normal candidate screen and every formal promotion attempt must execute
the complete locked strategic-move suite for its player count before expensive
matches. Include the exact candidate checkpoint and selector, all cyclic
acting-seat relabelings, and independently verify every fixture identity.

Keep human-reviewed iPhone positions separate from exhaustively proven
final-match official-scoring positions. Count original scenario families as
independent strategic evidence; seat rotations improve coverage but are not
independent human examples. Reject promotion when any locked answer fails.
Explicit exploratory screens may continue only while recording their tactical
failure. The current reviewed/proven suite covers four- and five-player play;
missing three-player locked positions remain a visible coverage gap.

## Latest Run823-969 focus

The current line of work is the user's core rule made operational: preserve the
led-color lane unless the switch is a responsible red/trump rescue. A move that
sloughs off a non-red led color burns future legal-move liquidity; the bot must
learn to do that rarely and deliberately, not as a casual scoring shortcut.

Current status after Runs896-899: the local best-of-N / Strawberry-style
belief-RFT line has its first calibrated offline pass. Run896 scaled actual-red
override rows to `104`, Run897 scaled matching base anchors to `102`, Run898
passed two source-held-out Transformer gates, and Run899 passed a stricter
calibration/test split: at the calibrated threshold, held-out test precision was
`91.67%` on `24` overrides with `0` false base overrides and mean value delta
`+0.3431`.

Run900 resolved the deployment blocker by adding a live
`:action_set_transformer` mode to the full-match evaluator, but the result was
a negative live-transfer test. The raw 5p Run779 policy control was `49.0%`
same-policy paradox rounds at m20. The Run899 selector with calibrated capped
candidates worsened to `58.0%`; all-legal candidates were `62.0%` at m10; and
a policy-gap guarded probe was `61.0%` at m20. Do not promote or install this
selector. The current next step is negative-aware/live-distribution training:
collect the actions the deployed selector would actually override, label those
same-state proposals with continuation/table-liveness outcomes, and train the
next action-set model to reject harmful false positives as well as choose
responsible red rescues.

Run901 started that next step. Online analogs are rejection-sampling
fine-tuning / best-of-N distillation / expert iteration: sample several
candidate continuations, score them with a verifier, and make the verified
better action more likely. We already tried a related whole-trajectory version
in Run604, which improved raw strength but mostly transferred strength rather
than survival. The useful Cat-in-the-Box version must be local and
negative-aware: train on the exact deployed proposal distribution and include
the rejected false positives. Run901 extended
`quantum_cat_live_substitution_replay.py` with
`--proposal-mode=action_set_transformer`, then smoke-mined the Run899 selector
in live 5p play. Smoke result: `8` accepted action-set proposals from `227`
considered states; all `8` had dense proposal outcomes; only `2 / 8` were
positive by dense delta, `5 / 8` were safe by paradox delta, and `3 / 8`
candidate proposals themselves led to any-paradox. That is exactly the bank we
need to scale before training the next selector; do not treat Run901 as a model
promotion.

Run902 scaled that smoke into a first live-distribution negative bank:
`48` rows from `3` shards, `48 / 1805` above-threshold Run899 proposals,
`14 / 48` proposal-positive by dense delta, `26 / 48` proposed candidates with
any-paradox, `28 / 48` bases with any-paradox, and `44 / 48` policy-target
changes. The bank is real, but the first training screen failed source
holdout. Hard-target action-set Transformers trained on two shards and held out
the third had best validation precisions only `28.6%`, `18.2%`, and `25.0%`;
no split passed even a relaxed 4-override gate. A soft `group_relative_soft`
control improved holdout-2 to `60.0%` precision and mean value delta `+0.682`,
but safe rate was only `80.0%`, still below the `90%` minimum. Do not promote,
install, or live-screen Run902 models. Next step: scale this exact
proposal-verifier bank to substantially more source shards before retraining,
and/or add a proposal-specific contrast objective that strongly penalizes the
selector's own false-positive candidate, not only the dense best action.

Run903 added that proposal-specific contrast objective to
`quantum_cat_action_set_transformer_trainer.py`. The trainer now loads proposal
metadata sidecars when `--proposal-contrast-loss-weight > 0`, marks the
selector's above-threshold candidate/base action, and adds a margin loss that
pushes false-positive proposals below the verified target while pushing true
positive proposals above base. The hook worked: on the Run902 bank it found all
`48` above-threshold proposal rows, `34` false-positive candidates, and `14`
positive candidates. But the first contrast-trained soft Transformer still
failed all held-out shards: best precisions were `33.3%`, `25.0%`, and `50.0%`.
Safety improved on holdouts 0 and 2 (`100%` safe rate), but precision is far
below the `75%` exploratory gate. Do not promote or live-screen Run903 models.
The evidence now favors scaling the proposal-verifier bank well past `48` rows
before training another selector; the objective is no longer the only blocker.

Run904 did that scale-up and Run905 falsified the simplest "more live negatives
+ same contrast objective" repair. Run904 collected four more live
Run899-proposal shards under the Run779 5p base distribution: `96` final rows
from `3687` considered proposal states, only `10 / 96 = 10.42%` proposal
positive by dense delta, `77 / 96 = 80.21%` safe by paradox delta, candidate
any-paradox `60 / 96 = 62.50%`, and base any-paradox `50 / 96 = 52.08%`.
Combined with Run902, the bank is now `144` rows with `24` proposal-positive
rows. Run905 trained the Run903 contrast objective on the combined `7`-source
bank with leave-one-source-out validation. It failed all `7 / 7` held-out
screens: no split had an eligible live-screen threshold, best held-out
precisions were `50.0%`, `40.0%`, `100.0%` on only `1` override, `0.0%`,
`66.7%`, `25.0%`, and `33.3%`, and validation value/risk correlations ranged
from negative to near-zero. Do not promote, install, or live-screen Run905
models. The next move should separate concerns: use the negative-heavy live
bank to learn a veto/abstention verifier, and collect or rebalance a deliberately
positive-rich responsible-red-rescue bank before training another selector.

Runs906-907 tested that separated veto idea and closed the simplest neural
version of it. Run906 added `quantum_cat_proposal_veto_trainer.py`, a direct
candidate-vs-base approval MLP over the existing legal-set features plus
proposal scalars. With target `positive_safe`, all `7 / 7` source-held-out
gates failed: the best split had only `2` approvals at `100%` precision, while
several Run904 holdouts inverted positives or selected `0%` precision. Run907
reused the same trainer with target `safe_by_paradox_delta`; this safety-only
veto passed only `1 / 7` held-out sources. It found signal on holdout `4`
(`16` approvals, `93.75%` safe precision, `75%` recall), but failed elsewhere
because unsafe proposal cases are too small/source-specific and false-approval
rates stayed high. Do not promote, install, or live-screen Run906/Run907
models. The next move is data-side: collect a deliberately balanced
proposal-verifier bank with explicit unsafe proposal cases and positive
responsible-red-rescue cases, or first audit simple proposal-scalar rules before
training another neural veto.

Run908 performed that deterministic proposal-rule audit before spending another
training cycle. It added `quantum_cat_proposal_rule_audit.py` and evaluated
simple source-held-out threshold rules over the combined Run902+Run904 bank
(`144` proposals, `24` positive-safe, `117` safe, candidate any-paradox
`86 / 144`, base any-paradox `78 / 144`). Result: positive-safe approval failed
completely (`0 / 7` selected held-out passes and `0 / 7` hindsight train-passing
rules that passed a holdout). Safety-only rules had some hindsight slices
(`4 / 7`) but the rule chosen by training sources still passed `0 / 7` holdouts.
Do not live-screen scalar-threshold proposal rules; the failure is not just
model capacity.

Run909 converted that conclusion into data infrastructure. The live substitution
collector now supports `--require-proposal-outcome` buckets (`positive_safe`,
`unsafe`, `candidate_any_paradox`, etc.) after dense proposal labeling. Tiny
smokes verified both sides of the balanced-bank path: a `positive_safe` smoke
accepted `2 / 6` verified proposals with mean dense value delta `+0.9025` and
mean paradox delta `-0.5`; an `unsafe` smoke accepted `1 / 6` with value delta
`-1.6975` and paradox delta `+1.0`. Next best concrete step: scale these bucketed
collectors into a balanced proposal-verifier bank, then train the verifier on
explicit accepted-positive and rejected-unsafe examples instead of the old
negative-heavy incidental proposal mix.

Run910 added the missing collection control, `--max-accepted-rows`, and used it
to build the first balanced bucketed bank. Two `positive_safe` shards and two
`unsafe` shards each hit a `6`-row accepted cap, for `24` rows total:
`12` positive-safe and `12` unsafe. This is much better class balance than the
old incidental Run902+Run904 proposal stream, but a tiny pair-fold verifier
screen still failed both held-out pairs. Fold `0,2` best validation approval
precision was only `57.14%`, safe rate `57.14%`, false-approval rate `50%`, and
mean paradox delta `+0.1429`; fold `1,3` was worse at `50.0%` precision,
`50.0%` safe, `66.67%` false approvals, and mean paradox delta `+0.25`. Do not
promote or live-screen Run910 verifier models. The right next move is to scale
the same bucketed collector to many more independent positive/unsafe sources
before retraining; this run proves the data shape, not the deployed verifier.

Run911 performed that scale-up to eight bucketed sources: four `positive_safe`
shards and four `unsafe` shards, each capped at `8` accepted rows. The fresh
bank is now `64` rows total, exactly `32` positive-safe and `32` unsafe. It is
balanced and deliberately aligned with the local best-of-N/RFT idea, but the
same high-dimensional verifier still did not transfer. All four balanced
pair-fold screens failed. Best held-out precision/safe rates were `57.14% /
57.14%` on fold `0,4`, `0.00% / 0.00%` on fold `1,5`, `57.14% / 57.14%` on
fold `2,6`, and `50.00% / 50.00%` on fold `3,7`; all approved mean paradox
deltas were non-improving (`+0.2857`, `+1.0`, `+0.2857`, `+0.1667`). The train
splits were nearly perfectly separated while validation probabilities were
unseparated or inverted, so do not live-screen, promote, or install Run911
verifier models. The next best move is not another identical larger MLP pass;
try a lower-dimensional proposal-feature verifier (`scalar_only` or
`scalar_action_flags`) and/or a much larger source-diverse bank before asking
the selector to change live play.

Run912 made that next change concrete. `quantum_cat_proposal_veto_trainer.py`
now supports `--proposal-feature-mode=full|scalar_only|scalar_action_flags`,
where `scalar_action_flags` uses only proposal scalars plus compact public
candidate/base follow-led, red, and token-loss flags. A four-fold Run911 screen
with `scalar_action_flags`, hidden size `24`, and target `positive_safe` still
failed all held-out gates. It no longer sprayed unsafe approvals on most folds:
folds `0,4`, `2,6`, and `3,7` reached `100%` precision/safe rate at their best
thresholds, but with only `1`, `1`, and `2` approvals respectively, below the
`4`-approval live-screen floor. Fold `1,5` broadened to `9` approvals but only
`55.56%` precision and `55.56%` safe rate. Do not live-screen or promote
Run912. The reduced-feature direction is promising only as a conservative
candidate filter; it needs more source-diverse positive-safe rows or a two-stage
selector that first gates with conservative flags and then ranks within approved
actions.

Run913 tested the first of those two options: more source-diverse bucketed
rows. Four more `positive_safe` shards and four more `unsafe` shards each hit
an `8`-row accepted cap, so the combined Run911+Run913 bank is `128` rows:
`64` positive-safe and `64` unsafe. This fixed neither recall nor transfer for
the compact verifier. On eight paired held-out folds with
`--proposal-feature-mode=scalar_action_flags`, every gate failed. Four folds
were high precision but too sparse (`1`, `1`, `1`, and `1` approval), and the
folds that reached the `4`-approval floor had only `50.00-66.67%` precision and
`50.00-66.67%` safe rate. Do not live-screen or promote Run913. More examples
alone are not enough for this verifier shape; the next aligned move is a
two-stage or pairwise objective that explicitly learns "approve only the small
conservative slice, then rank among approved candidates," rather than asking a
single binary verifier to supply both safety and recall.

Run914 turned that next step into data infrastructure. The live substitution
collector now preserves the action-set selector's full `scored_candidates` list
and, after dense labeling, annotates each scored candidate with dense value/risk
deltas, positive/safe flags, paradox flags, and public tactical flags (`color`,
`follows_led`, `is_red_switch`, `switches_off_led`, token-loss, and immediate
win flags). The final smoke `multi_candidate_flags_smoke2` wrote `2` proposal
rows; both rows had `2` scored candidates with outcome labels, and each row had
`1` positive-safe scored candidate. This is not a model promotion. It is the
missing bank shape for the next local best-of-N/RFT experiment: train a
multi-candidate selector or pairwise/listwise objective over the full candidate
set, not a single top-proposal binary verifier.

Runs915-916 used that shape for the first real scored-candidate verifier pass.
Run915 added `--proposal-example-source=scored_candidates` to
`quantum_cat_proposal_veto_trainer.py` and smoked it by duplicating the Run914
metadata as train/validation sources; `2` proposal rows expanded into `4`
candidate examples per source with no skipped metadata. Run916 then collected a
tiny balanced real bank: `16` proposal rows across two `positive_safe` shards
and two `unsafe` shards, expanding to `37` outcome-labeled scored-candidate
examples (`13` positive-safe, `23` safe, `23` candidate-any-paradox). The first
paired source-held-out `scalar_action_flags` screens still failed both folds.
Validation probabilities inverted (`approved` candidates averaged below
`rejected` candidates), and best thresholds had only `42.86%` and `33.33%`
precision with unsafe mean paradox deltas. Do not live-screen, promote, or
install Run916 models. The useful result is diagnostic: candidate-slate
expansion works, but the first binary MLP remains too small/source-specific.
Next best move is a larger Run916-style bank plus a grouped pairwise/listwise
loss that ranks candidates within the same state, not independent binary
approval alone.

Runs917-918 tested that exact diagnosis. Run917 added optional
`--loss-mode=bce|pairwise|bce_pairwise` to the proposal veto trainer and
reported within-state pairwise group counts. On the same Run916 bank,
`bce_pairwise` still failed both paired held-out folds, and the report explained
why: the train folds had only `1-3` mixed positive-vs-negative proposal states.
The loss was wired, but the bank was not shaped for within-state ranking. Run918
therefore added collector buckets `scored_has_positive_safe` and
`scored_mixed_positive_safe`. A tiny mixed-slate smoke found `1` qualifying row
from `99` considered proposal states: `3` scored candidates, all outcome
labeled, exactly `1` positive-safe and `2` rejected alternatives in the same
state. A duplicated-source trainer smoke confirmed that this produces `1`
pairwise group with `2` ranking pairs. Do not promote anything from Runs917-918.
The next high-value move is to scale the `scored_mixed_positive_safe` collector
into a real multi-source bank before rerunning grouped pairwise/listwise
screens.

Run919 did the first tiny scale-up. Three completed mixed-slate shards each hit
a `2`-row cap, yielding `6` proposal rows, `14` outcome-labeled scored
candidates, `7` positive-safe candidates, `14` safe candidates, `7` candidate
any-paradox cases, `6` within-state pairwise groups, and `8` ranking pairs.
`bce_pairwise` still failed all three leave-one-source-out screens. A
pairwise-only control passed one tiny holdout slice (`1` approval, `100%`
precision, `0` false approvals, value delta `+1.7025`, risk delta `-1.0`) but
failed the other two folds. This is not promotable and not live-screenable, but
it is the first evidence that the mixed-slate bucket plus pure ranking can find
a high-quality conservative slice. Next should scale `scored_mixed_positive_safe`
to many more sources and compare pairwise-only, BCE+pairwise, and a true
listwise/top-1 objective before any live deployment attempt.

Run920 added and tested that true listwise/top-1 objective in
`quantum_cat_proposal_veto_trainer.py`. This is the direct local analog of the
modern best-of-N / rejection-sampling / expert-iteration idea: score several
candidate moves in the same state, train the model to put the verified best
candidate on top, then only later consider using that selector live. On the tiny
Run919 bank, both pure `listwise` and `bce_listwise` failed all three
source-held-out threshold gates. The new top-1 diagnostics were also negative:
each mode selected an approved candidate in only `1 / 6` held-out proposal
groups, while choosing candidate-any-paradox actions in `5 / 6` groups. Do not
promote or live-screen Run920 models. The right conclusion is not "best-of-N
learning is wrong"; it is that this bucket needs many more independently mined
mixed candidate slates, and probably a group top-1/ranking gate as the primary
offline metric, before it can support live deployment.

Run921 scaled the mixed-slate bank and gave the first less-toy screen for this
line. Seven new `scored_mixed_positive_safe` sources completed, adding `21`
fresh mixed proposal rows after `3512` proposal states and `106` above-threshold
proposals; one slow shard was interrupted with no artifact and is excluded.
Combined with the three Run919 sources, the training bank is now `10` sources,
`27` proposal groups, `71` scored-candidate examples, and `44` within-state
ranking pairs. Pairwise-only improved to `3 / 5` exploratory paired-holdout
threshold gates, but held-out top-1 selection remained too noisy: pure pairwise
selected approved top-1 candidates in only `12 / 27` groups and
candidate-any-paradox top-1 actions in `18 / 27`. `bce_listwise` had the best
top-1 approval rate (`14 / 27`) but still selected candidate-any-paradox in
`17 / 27` groups. Do not promote or live-screen Run921 models. The evidence is
directionally better than Run920, but still says this verifier/ranker is not
stable enough and is not yet live-integrated. Next useful move is either much
larger source-diverse mixed-slate collection or a stronger group/listwise model
that uses the full slate context instead of compact independent MLP scoring.

Run922 tried the smallest version of that stronger group idea without changing
deployment plumbing: `scalar_action_flags_slate` adds candidate rank plus
score/margin/pred-value/pred-positive/policy-probability statistics relative to
the scored candidate slate. This is still an MLP, but it no longer scores each
candidate in a vacuum. On the same Run921 bank, the best pure-listwise seed
improved held-out top-1 materially (`19 / 27` approved, candidate-any-paradox
`11 / 27`) versus Run921's pure-listwise baseline (`11 / 27` approved,
candidate-any-paradox `19 / 27`). However, two repeat pure-listwise seeds fell
back to `14 / 27` approved and `16 / 27` candidate-any-paradox, matching only
the prior best-ish BCE/listwise range. Do not promote or live-screen Run922.
The right takeaway is that slate context is useful signal, but this MLP is
still seed-fragile; the next model-side move should be a real group/listwise
slate encoder or a larger bank before any runtime integration.

Run923 implemented that real group/listwise slate encoder as
`--proposal-model-kind=deepset`: candidate features are encoded, mean/max pooled
inside each same-state candidate slate, and scored with candidate-minus-slate
context. On the same Run921 bank it did not beat Run922. Hidden-24 DeepSet
listwise reached only `14 / 27` held-out approved top-1 and `16 / 27`
candidate-any-paradox top-1; hidden-48 improved threshold gates to `3 / 5` but
only `15 / 27` approved top-1 with the same `16 / 27` paradox top-1. Do not
promote or live-screen Run923. The current conclusion is sharper: the slate
feature signal is real, but the `27`-group bank is too small/source-specific for
a stronger model to generalize. Next best move is data-side scale: collect many
more independent mixed slates before trying runtime integration or another
higher-capacity architecture.

Run924 performed that data-side scale and directly answers the user's
"Strawberry-style" question for this branch. The approach is possible and is
already the shape of Runs914-924: sample a slate of candidate moves, score each
candidate by continuation/table-liveness outcomes, then train a local ranker to
make the verified better move more likely. Run924 added `30` fresh
`scored_mixed_positive_safe` rows from `8` sources (`7` full shards plus `1`
usable partial) after `7868` proposal states and `197` above-threshold Run899
proposals. Combined with Runs919 and 921, the bank is now `18` sources, `57`
proposal groups, `152` scored-candidate examples, and `95` ranking pairs. The
oracle labels in this bank are useful: label-ranking top-1 would choose approved
candidates in `47 / 57` groups, with candidate-any-paradox top-1 in only
`16 / 57`. The trained slate-feature pure-listwise MLP still failed to learn
that ranking robustly: three repeat six-fold source-held-out screens landed at
`31 / 57`, `28 / 57`, and `30 / 57` approved top-1, while selecting
candidate-any-paradox top-1 actions in `31 / 57`, `34 / 57`, and `33 / 57`.
Reject Run924 models; do not live-screen, promote, install, or distill them into
the base policy. The next aligned move is not "abandon best-of-N/RFT"; it is to
make the verifier/ranker more reliable before using its winners as policy
targets, likely through more diverse slates, harder negative retention, and/or a
trajectory/state-memory selector.

Runs925-926 tested the most direct repair for Run924's bad top-1 behavior:
change the approval target from `positive_safe` to
`positive_safe_no_candidate_any_paradox`, so a candidate is approved only if it
improves dense value, is paradox-safe by delta, and its own continuation avoids
any paradox. This reduced the label/oracle ceiling to a cleaner target:
`65 / 152` approved examples, `50 / 57` groups with an approved candidate, and
label-ranked top-1 candidate-any-paradox only `14 / 57`. But learning still did
not transfer. Run925 pure-listwise repeats reached only `25 / 57`, `26 / 57`,
and `27 / 57` approved top-1, with candidate-any-paradox top-1 still
`32 / 57`, `31 / 57`, and `30 / 57`. Run926 added BCE to push down every
rejected/candidate-any example directly; it was no better (`23-25 / 57`
approved top-1 and `32-34 / 57` candidate-any top-1). Reject both runs. The
lesson is sharper now: the current immediate slate features do not reliably
predict which candidate continuation will avoid paradox out of source. Future
best-of-N/RFT work needs richer state/trajectory features or a different
verifier, not another loss-mode tweak on this same feature bank. The trainer now
also has `--max-validation-candidate-any-paradox-rate-for-live` so future
approval gates can directly cap candidate-any approvals instead of relying only
on value/safe-by-delta metrics.

Runs927-928 made the first actual state-context repair rather than another
target/loss tweak. `quantum_cat_proposal_veto_trainer.py` now supports
`--proposal-feature-mode=scalar_action_flags_slate_state_context`, which adds
compact legal state counters (`trick_number`, `legal_count`, `decision_index`,
led/red/switch legal counts and ratios, avoidable-switch flag) plus the existing
observation/policy-context tail from the pair features. This raised feature size
from `74` to `149`. Run927 showed the added context contains signal: all-bank
top-1 improved to `47 / 57` approved and only `10 / 57` candidate-any-paradox.
But source-held-out repeats still did not transfer stably: main was `27 / 57`
approved and `30 / 57` candidate-any with `2 / 6` gates, while repeats fell to
`23 / 57` approved and `34 / 57` candidate-any. Run928 lowered capacity
(`hidden=8`, dropout `0.18`, weight decay `0.001`) to test overfit; it reduced
all-bank fit and did not improve held-out ranking. Reject both runs. The next
useful move is new data/label shape, not another small MLP tweak: source-diverse
hard negatives and/or fuller trajectory features collected at every prior
decision in the round, so the verifier can see the actual lane-burn history that
precedes a candidate continuation.

Run929 started that fuller-trajectory path. It did not train or promote a
model; it added the missing collector instrumentation for Strawberry-style /
best-of-N adaptation. `quantum_cat_live_substitution_replay.py` now supports
`--record-round-history` and `--round-history-max-events`. Accepted replay rows
can carry JSON-safe prior-decision history for the current round, including
follow-led, red switch, off-led switch, whether the switch was avoidable because
the led color was still legal, avoidable non-red switch, and off-led token-loss
fields. The smoke collected `2` accepted 5p rows with `33` prior history events
and `13` prior play events; the metadata spot-check showed row histories of
`14` and `19` events. This is infrastructure only, but it gives the next
verifier the missing causal substrate: not just "candidate action in this
state," but "what lane-burn sequence led into this candidate state." Next step:
scale a real Run930 trajectory-aware proposal bank and train a history/context
ranker against the Run927/928 source-held-out gates before any live screen.

Runs930-931 turned that instrumentation into the first training-side and
data-side loop. Run930 added
`--proposal-feature-mode=scalar_action_flags_slate_state_round_history_context`
to `quantum_cat_proposal_veto_trainer.py`, appending compact
`round_history_summary` counters to the existing slate/state/context feature
vector (`149 -> 174` features). It then collected `16` mixed 5p proposal rows
across four sources with `306` prior history events and trained four
source-held-out folds. The result was a clean rejection: `0 / 4` gates passed,
held-out top-1 approved was only `5 / 16`, and top-1 candidate-any-paradox was
`11 / 16`. A state-only control produced the same held-out totals. The key
diagnosis is that the Run930 mixed bank had `0` prior avoidable-switch events,
so the history features existed but did not contain the user's actual upstream
lane-burn signal.

Run931 fixed the data generator side of that diagnosis by enabling controlled
led-contrast behavior exploration while still labeling against the frozen policy
base. Four short exploratory shards produced another `16` rows, now with `319`
prior history events, `68` prior avoidable-switch events, `26` prior avoidable
non-red switches, and `68` prior avoidable token-loss events. Those rows were
mostly hard negatives (`31` scored candidates, `25` candidate-any-paradox, only
`3` strict positive-safe-no-any candidates). Mixing them with Run930 and
holding out each original Run930 source still failed `0 / 4` gates with the
same held-out top-1 totals (`5 / 16` approved, `11 / 16` candidate-any). Reject
all Run930/Run931 models. The next useful move is now very specific: scale a
balanced bank that has both responsible-red/no-paradox positives and
avoidable-switch hard negatives, rather than training another small MLP on the
tiny smoke banks or live-screening a selector.

Run932 made that next bank shape admissible directly in the collector.
`quantum_cat_live_substitution_replay.py` now has strict scored-candidate
buckets for `scored_has_positive_safe_no_candidate_any_paradox`,
`scored_mixed_positive_safe_no_candidate_any_paradox`, and
`scored_has_candidate_any_paradox`, plus prior round-history filters for minimum
avoidable switches, avoidable non-red switches, and avoidable token-loss events.
The smoke results prove both sides are collectable: `3` clean positive rows from
`919` proposal states, each with a strict no-candidate-any positive, and `3`
hard-negative rows with `17` prior avoidable-switch events and candidate-any
evidence. A tiny two-source listwise trainer smoke consumed the bank but is not
useful model evidence: held-out top-1 stayed `1 / 3` approved and `2 / 3`
candidate-any-paradox in both directions. Accept the Run932 admission
infrastructure, reject the smoke models, and scale multiple balanced shards
before the next source-held-out listwise/top-1 gate.

Run933 performed that first balanced scale. It added
`quantum_cat_balanced_history_bank_builder.py`, a restartable builder that
collects and combines the Run932 buckets. The scaled bank is now `24` rows from
`8` sources: `12` clean positive/no-any rows, `12` avoidable-switch hard-negative
rows, `58` scored candidates, `19` strict positive-safe-no-candidate-any labels,
`39` candidate-any-paradox candidates, and `63` prior avoidable-switch events.
This accepts the bank shape. The model-side smoke still rejects the current
small MLP listwise verifier: paired source-held-out folds passed only `2 / 4`
relaxed tiny gates, each passing fold approved only `1` high-threshold candidate,
and aggregate validation top-1 selected approved candidates in only `6 / 23`
groups while selecting candidate-any-paradox actions in `17 / 23`. Do not
live-screen, promote, install, or distill Run933 models. The next best move is
not another tiny MLP tweak; either scale this bank much further, add
repeated/common-random-number rollout labels, or implement a direct
group-relative policy update from same-state rollout rewards.

Runs934-936 made that group-relative update real in the proposal verifier.
Before Run934, `--target-mode=group_relative_soft` existed but did not actually
change the proposal-verifier listwise target. Run934 wired a same-state softmax
target over `group_relative_reward = dense_value_delta - candidate_any/actor/
forced_paradox penalties + small strict-approval bonus`, then trained on the
Run933 balanced bank. This improved held-out top-1 from Run933's `6 / 23`
approved and `17 / 23` candidate-any to `8 / 23` approved and `15 / 23`
candidate-any, still with only `2 / 4` relaxed tiny gates. Run935 screened two
obvious variants: sharper MLP soft targets and DeepSet slate context. The sharp
MLP reached `3 / 4` relaxed gates but stayed `8 / 23` top-1 approved and
`15 / 23` candidate-any. DeepSet gave the best top-1 screen on that bank:
`9 / 23` approved and `14 / 23` candidate-any, but only `2 / 4` relaxed gates.
Run936 doubled the balanced history bank to `48` rows from `16` sources and
`108` scored candidates, then ran the same DeepSet/group-relative-soft screen
over `8` paired source holdouts. The larger bank did not transfer: only `2 / 8`
fold gates passed, aggregate top-1 approved was `14 / 40`, and top-1
candidate-any-paradox was `26 / 40`. Reject all Run934-936 models; keep the
group-relative target plumbing and scaled bank as useful infrastructure. Next
needs repeated/common-random-number rollout labels or a more direct policy update
from same-state rewards, not promotion.

Run937 implemented the repeated/common-random-number version for the proposal
slate itself. `quantum_cat_live_substitution_replay.py` now supports
`--repeat-validation-candidate-mode=proposal_scored_candidates`, re-annotates
`proposal.scored_candidates` after repeat-validation action targets are written,
and `quantum_cat_proposal_veto_trainer.py` reads repeat-validation rates when
constructing candidate views. A six-source smoke bank produced `24` rows,
`53` scored candidates, `33` strict approved candidates, and `20`
candidate-any candidates, with `24 / 24` rows carrying exclusive repeat
targets. The held-out DeepSet screen improved the shape but not enough:
top-1 approved `11 / 22`, top-1 candidate-any `11 / 22`, and only `2 / 6`
fold gates passed. Reject Run937 models for deployment, but accept repeated
proposal-slate labeling as a directional improvement over Run936. Next should
scale/diversify this label source, especially red-switch sources; do not live
screen or distill yet.

Run938 made that red-switch diversification concrete. A strict
proposal-slate-only red filter found `0` accepted rows after `40` proposal
states, so the collector added
`--repeat-validation-candidate-mode=proposal_scored_plus_red_follow_value`,
which starts from the live proposal slate and augments it with red-switch,
follow-led, and value candidates before common-seed repeat validation. The
five-source bank has `17` rows, `102` scored candidates (`68` blue/follow-led
and `34` red), `68` approved candidates, `34` candidate-any-paradox candidates,
and every row has both a passed follow-led and passed red-switch candidate.
Five leave-one-source-out DeepSet/listwise folds were the best offline result
yet for this line: `3 / 5` fold gates passed, top-1 approved was
`15 / 17 = 88.24%`, top-1 candidate-any-paradox was `2 / 17 = 11.76%`, and
top-1 safe was `17 / 17`. This is a strong directional win over Run937's
`11 / 22` approved and `11 / 22` candidate-any, but it still narrowly misses
the `<=10%` candidate-any gate, one held-out source fails badly, and the bank is
only `17` rows. Reject Run938 models for live-screen, promotion, installation,
or policy distillation. Next should scale the same red-augmented repeated-label
collector to many more independent sources before rerunning the source-held-out
listwise gates.

Run939 added the first scale-out tranche for that exact path. Two completed
fresh sources added `8` rows; they were deliberately red-heavy (`6 / 8` selected
red) while still preserving both passed follow-led and passed red-switch
candidates in every row. One further shard was interrupted after stalling on an
expensive low-progress state, so the completed Run939 tranche is only those two
sources. Combined with Run938, the seven-source bank is `25` rows and `150`
scored candidates (`98` blue/follow-led, `52` red), with `104` approved
candidates, `46` candidate-any-paradox candidates, `12 / 25 = 48%` selected
red, and `25 / 25` rows containing both passed follow-led and passed red-switch
candidates. The held-out DeepSet/listwise top-1 result is the strongest offline
best-of-N/RFT signal so far: `25 / 25` top-1 approved, `0 / 25`
candidate-any-paradox, and `25 / 25` safe. But calibrated threshold gates still
passed only `3 / 7` folds, mostly because some folds were underconfident or
over-broad at threshold. Do not promote, live-screen, install, or distill yet.
Next should scale to at least `12-16` sources and fix calibration/deployment
semantics for a top-1 selector before any live same-policy survival test.

Run940 added repeatable scale/screen infrastructure and checked whether the two
usable partial sources from the next collection window preserve the Run939
signal. `quantum_cat_red_augmented_repeat_bank_builder.py` now gives us a
restartable red-augmented repeated-slate collector, and
`quantum_cat_red_augmented_repeat_fold_screen.py` runs the exact DeepSet/listwise
leave-one-source-out screen from metadata globs. The new seed windows were
sparse: only `2` positive-row partial sources were useful, while several timed
shards had `0` rows, so the bank is still only `9` sources and `27` rows. But
the 9-source screen was the cleanest offline result yet: `9 / 9` threshold gates
passed, `27 / 27` held-out top-1 choices were approved, `0 / 27` top-1 choices
had candidate-any-paradox, and `27 / 27` were safe. This strengthens the
best-of-N/RFT direction, but it is still offline evidence only. Do not promote,
install, live-screen, or distill Run940 yet. Next: collect `12-16+` sources with
better-yield seed windows, then wire a live top-1 proposal-veto selector so this
ranking metric can be tested by the same-policy paradox gate.

Run823-824 proved that direct same-public-state contrast labels can be clean.
The miner compares the frozen/base action to candidate actions, repeat-validates
continuations, and adds off-led non-red lane burns as hard legal negatives.
Run824 scaled this to `261` rows and `1448` explicit lane-burn negatives; source
holdouts stayed very high precision (`96.2-100%`) with `100%` safe rate. That
validated the label source, but live 5p survival still failed (`56.7%` paradox
round rate at m60), so clean static labels alone were not enough.

Run825-827 tried to deploy those labels as abstaining pairwise/listwise
selectors. The infrastructure worked, including `:pairwise_boundary` deployment
and same-state DeepSets/Transformer selectors, but live/source-held-out transfer
did not. The old boundary mostly proposed same-led-color rank swaps while the
rollout labels often wanted red rescue actions, so the missing behavior became
concrete: red-switch candidates must be first-class, not incidental.

Run828 fixed data collection shape with `--label-base-action-source=frozen_policy`.
Behavior exploration could visit red-switch states while labels stayed relative
to the frozen deployment policy. That produced `192` red-rich 5p states, but
single-rollout dense labels were too noisy; held-out selector precision stayed
low (`8.3-28.6%` DeepSets, `20.0%` best Transformer). Do not promote Run828 or
train larger models on that single-rollout bank.

Run829 added repeat-validation admission and then fixed a subtle but important
target-writing bug: selector trainers consume replay action tensors, so repeated
rollout outcomes must be written back into candidate risk/value targets, not
only metadata/policy targets. The corrected `follow_red_value` queue bank had
`50` rows, all admitted candidates at candidate any-paradox rate `0.0` versus
base any-paradox rate `1.0`, with mean value delta `+1.7230`. But source-held-out
selectors still failed: best DeepSets precisions were `0.0%`, `66.7%` with only
`3` overrides, and `50.0%`; the action-set Transformer reached only `33.3%`,
`25.0%`, and `54.5%`. Do not promote or live-screen Run829 selectors.

Run830 implemented the local version of the modern reasoning/RFT idea more
literally. The trainers now support `--target-mode=group_relative_soft`, where
each legal-action set gets a soft target from repeat-validated
`value_delta - risk_delta - unsafe_penalty`; forced led-lane burns get zero
target mass, and all-bad states fall back to a hard base/abstain target. This is
the "sample candidate moves, score continuations, make the better continuation
move more likely" loop.

Run830 tested both target shapes. On the corrected `50`-row bank, three-source
soft Transformer holdouts failed (`50.0%`, `66.7%` with only `3` overrides, and
`50.0%` precision). Then three more shards expanded the bank to `81` rows with
all target tensors written, but accepted red targets were still only `11 / 81 =
13.6%`. Six-source soft Transformer gates failed (`28.6%`, `46.2%`, `41.7%`
precision), and the hard-target control also failed (`20.0%`, `41.2%`, `40.0%`).
Safe rate was `100%` in these screens; positive precision/source transfer is
the binding problem. Do not live-screen, promote, or install Run830 selectors.

Run831 added the missing data-shaping controls for that next step. The live
substitution miner now annotates repeat-validation candidates with their color,
follow-led status, red-switch status, and dense value; it can require rows with
a passed red switch, require rows with a passed follow-led action, and cap
accepted selected-target colors while collecting balanced banks.

The strict red-passed smoke proved the shape: one admitted same-state row had
both a passed follow-led candidate and a passed red-switch candidate under
common continuations. Four focused 5p shards then produced `18` red-passed rows
from `277` repeat-validation attempts. The selected target colors were much
better balanced for our purpose: `9` red / `4` yellow / `3` green / `2` blue,
and every row contained at least one passed red-switch candidate.

Run831 mixed those `18` focused rows with the prior `81` general rows and
trained source-held-out action-set Transformers. This moved the best red-focused
soft split in the right direction (`66.7%` precision on `9` overrides, `100%`
safe, mean value delta `+0.8716`), but the other red holdout was only `50.0%`
precision on `4` overrides. A hard-target control did not solve it (`71.4%`
precision on `7` overrides for one split, `100%` on only `1` override for the
other). Do not live-screen, promote, or install Run831 selectors.

Run832 added a repeat-bank reporter and scaled the red-passed bank to `58`
focused rows from `888` repeat-validation attempts. Selected target colors are
now properly red-heavy for this curriculum: `30` red / `12` yellow / `10` green
/ `6` blue (`51.7%` red). Every row has a passed red-switch candidate, and
`32 / 58 = 55.2%` also have a passed follow-led candidate. This is the first
bank that really looks like the desired survival-grammar contrast.

Run832 then mixed the `58` focused rows with the prior `81` general rows and
trained source-held-out soft/hard action-set Transformers. Soft targets still
failed (`64.3%` precision on `14` overrides for the initial focused holdout,
`65.4%` on `26` overrides for the new focused holdout). Hard targets were
better on the larger new holdout (`74.1%` precision on `27` overrides, `100%`
safe, mean value delta `+1.2138`) but still below the `85%` gate. Do not
live-screen, promote, or install Run832 selectors.

Run833 scaled the same red-passed bank to `131` focused rows from `2279`
repeat-validation attempts. The data stayed well-shaped on the surface:
selected targets were `64` red / `27` yellow / `20` blue / `20` green, every
row had a passed red-switch candidate, and `59.5%` also had a passed follow-led
candidate. But source-held-out precision still failed badly. Soft target splits
were `64.0%`, `46.2%`, and `63.8%`; hard target splits were `66.7%`, `55.9%`,
and `59.5%`. No Run833 selector should be live-screened or promoted.

Run833 also found the likely reason scaling stalled: repeat validation wrote
common-seed labels for the base action and tested candidates, but the replay
still kept older one-rollout dense masks for non-candidate legal actions. The
selector was therefore training on a mixed label family inside each state, not a
pure common-seed candidate group. The miner now has
`--repeat-validation-exclusive-action-targets`, which clears old per-action
risk/value masks before writing the repeat-validated base+candidates. A strict
red-passed smoke admitted `3` rows and the mask audit passed: each row exposed
exactly `7 / 7` expected labeled actions (`base + 6 candidates`) to the
selector.

Intermediate conclusion after Run835: the Strawberry/RFT-style mechanism was
feasible and wired locally, and red-passed candidate-group mining created the
exact follow-led versus responsible-red-switch contrast we wanted. Run834
verified the corrected exclusive-target path on a small diagnostic replay:
`1` row from `5` evaluated decisions, selected target color red, `6` red-switch
candidates, `3` passed red switches, and a mask audit showing exactly `7 / 7`
labeled actions (`base + 6 candidates`) visible to the trainer. But the old
red-passed bank was mixed-label and should not be scaled further, and the
one-row Run834 artifact was not trainable. Run835 added and smoked the
restartable bounded collector for this exact recipe. The collector reproduced
the same clean exclusive diagnostic row, wrote an aggregate bank report, and a
same-command resume returned immediately with `status=skipped_completed`.

Run836 then added `--repeat-validation-prefilter-candidates`. This keeps the
accepted-row target semantics but tests the small red/follow/policy candidate
queue before paying for all-legal dense labels. In a two-source smoke/tranche,
the prefilter produced `3` clean exclusive rows from `10` evaluated decisions
and avoided dense all-legal labeling for the other `7`. Every accepted row still
audited as exactly `7 / 7` labeled trainer-visible actions (`base + 6
candidates`).

Run837 scaled that exact prefiltered exclusive recipe to a non-trivial clean
bank: `26` rows from `60` evaluated decisions across six restartable 5p shards.
Every row had a passed red-switch candidate, selected target color was red in
`25 / 26` rows, and the full mask audit again passed with exactly `7 / 7`
trainer-visible actions per row (`base + 6 candidates`). This is the first
real evidence that the local Strawberry/RFT-style loop is operational at bank
scale: sample candidate moves, repeat-verify continuations, then train toward
the better candidate group. It is still too small for training. Continue with
the same collector/pre-filter/exclusive recipe until the focused bank reaches
`150-200` rows, then repeat the same soft/hard source-held-out gates.
Source-held-out precision must pass before any live survival screen.

Runs839-842 showed the first selector-training attempt on the larger red bank
does not transfer across sources yet, even after high-margin filtering, a
same-state pairwise boundary, and source-adversarial delta training. Runs843-845
then tested whether later trick-2-through-5 red-rescue mining was the missing
slice; it was not. Late strict labels were very sparse, and relaxed thresholds
or wider candidate queues did not uncover hidden near-misses.

Run846 shifted the same strict repeat-validation recipe earlier, to trick
`0..1`, and found the useful island: `9` clean exclusive rows from `24`
attempts, every row with a passed red-switch candidate and `8 / 9` selected
targets red. Runs847-848 showed that interactive scaling was too sluggish
without partial artifacts. Run849 added `--partial-save-interval-decisions` so
attempt rows, summaries, and partial replays are preserved mid-shard. Run850
then completed a tiny partial-save early tranche: `6` clean rows from `12`
attempts, `6 / 6` selected red, every row with a passed red-switch candidate,
and every final replay row exposing exactly `7 / 7` action target labels and
`7 / 7` action value labels. Combined Run846+Run850 is now `15` rows from `36`
attempts, `14 / 15` selected red, and `15 / 15` rows with a passed red-switch
candidate. This is strong evidence for early/pre-collapse data collection, but
it is still much too small to train or screen. Keep scaling the early bank with
partial saves until it is source-diverse enough for held-out selector gates.

Run851 scaled the same early/pre-collapse recipe to eight four-decision shards.
It completed cleanly with `12` rows from `32` attempts, `11 / 12` selected red,
and `12 / 12` rows with a passed red-switch candidate. The final replay mask
audit again passed at exactly `7 / 7` action target labels and `7 / 7` action
value labels per accepted row. Combined Runs846+850+851 are now `27` rows from
`68` attempts, `25 / 27` selected red, and `27 / 27` rows with passed
red-switch candidates. This independently confirms the early label island, but
the four-decision shard shape is too slow for interactive scaling and the bank
is still too small for serious source-held-out selector training. Next should
prefer many small two-decision shards or an autonomous restartable scheduler
until the early exclusive bank reaches roughly `150-200` rows.

Runs852-853 added and proved that scheduler. `quantum_cat_early_bank_builder.py`
wraps the existing collector into small restartable batches, counts existing
metadata, prefers final shard metadata while falling back to partial metadata,
and refreshes combined bank/attempt reports after each batch. Run852 smoked it
with two one-decision shards and raised the combined bank to `29` rows. Run853
then ran two six-shard two-decision batches, adding `7` more rows. The combined
early bank is now `36` rows from `94` repeat-validation attempts, `34 / 36`
selected red, `36 / 36` rows with a passed red-switch candidate, and every
new accepted row still audits at exactly `7 / 7` action target labels and
`7 / 7` action value labels. This is still too small for selector training,
but bank growth is now operationally smoother.

Runs854-858 continued and then tested the cleaned-up local
Strawberry/RFT-style loop: sample candidate moves, repeat-verify same-state
continuations, then write exclusive base+candidates action targets so the model
is trained only on the validated candidate group. Run854 raised the combined
bank to `59` rows, Run855 to `64`, Run856 to `105`, and Run857 to `153` rows
from `370` repeat-validation attempts. The bank stayed strongly
red-rescue-shaped: `144 / 153` selected red, `153 / 153` rows with a passed
red-switch candidate, and a small follow-led-positive slice (`17 / 153` rows).
Run857 added `48` final accepted rows, and every new row audited at exactly
`7 / 7` action target labels and `7 / 7` action value labels. This crossed the
lower `150-200` row selector gate.

Run858 then ran the required source-held-out action-set Transformer gates on
that `153`-row early/pre-collapse exclusive bank. Mechanically, this is the
modern reasoning/RFT idea applied to Cat in the Box: generate several legal
candidate moves, score multiple continuations with a verifier, and make the
repeat-validated better actions more likely. Empirically it still did not
transfer well enough. Soft target held-out precision was `49.0%`, `56.9%`, and
`59.0%`; hard target held-out precision was `52.5%`, `73.3%`, and `53.6%`.
Safe rate stayed `100%`, but the live-screen gate is `>=85%` positive
precision with enough overrides. Do not live-screen, promote, or install any
Run858 selector. The current blocker is not whether the RFT-style loop is
possible; it is source-stable selector generalization from these features and
labels.

Runs859-861 localized that failure to intra-red card/rank selection, not the
general idea of red rescue. Simple red rules failed (`highest_policy_red_switch`
precision `52.9%`, `48.1%`, `44.0%`), the best one-dimensional red liveness
rule only reached about `51-54%`, a red-switch-only MLP stayed at
`54.4-58.5-56.8%`, and a richer true-state liveness feature ranker was either
coverage-starved or around `50%`. Oracle red/passed-candidate controls stayed
`100%`, so the candidate set contains good red moves; the model cannot infer
which red card is best from the current observable features/labels.

Run862 tested the proposed fix: belief-averaged labels over hidden worlds
consistent with the acting player's observation. The exact table-liveness
variant was too conservative and unstable for this red-rescue curriculum:
on `60` early non-red-led states, belief split agreement was only
`34 / 60 = 56.7%`, stable belief rows were `2 / 60 = 3.3%`, and belief-best red
switches were only `8 / 60 = 13.3%`. A naive belief-rollout smoke is closer to
the intended Strawberry/RFT target but was too slow and unstable (`3` states,
`0 / 3` split agreement). Candidate-capped rollout scoring
(`base + top-red + top-follow`) fixed the immediate cost problem in a tiny
smoke (`3` states in `15.4s`, `2 / 3` split agreement), but it was not
statistical evidence yet.

Runs863-865 made that modern-LLM-style loop concrete and information-legal:
sample candidate moves, score rollout continuations across actor-consistent
hidden worlds, keep stable winners, and make those moves more likely. Run863
scaled the capped belief-rollout probe to `40` states and found `10 / 40 =
25.0%` stable belief overrides of the base follow-led action. Run864 confirmed
the signal at `80` states: stable belief red overrides were `21 / 80 = 26.25%`,
with `35 / 80 = 43.75%` stable belief rows and `48 / 80 = 60.0%` belief-best
match to the hidden actual best. Run865 added opt-in replay export and smoked
it on `20` states, producing `8` strict `stable_red_overrides` replay rows.
The replay loader validated policy targets summing to `1`, exactly `2`
candidate action-value labels per row, and `0` action-paradox labels, so the
export does not fake risk targets. Do not promote from this smoke. The next
step is a source-diverse belief-RFT replay bank, then held-out selector/trainer
gates before any live survival screen.

Run866 added the restartable source-diverse bank builder for that path:
`quantum_cat_belief_red_switch_bank_builder.py`. It ran the strict
`stable_red_overrides` filter across Run865 plus fresh seed shards and produced
a `15`-row combined replay from `100` audited states, `6` sources including one
completed zero-row shard, and `15` distinct row groups. The replay loader
validated the merged bank: policy
targets sum to `1`, every row has exactly `2` action-value candidate labels,
and action-paradox masks stay zero. The main lesson is yield: strict stable-red
positives are clean but sparse (`15 / 100 = 15.00%`), and one completed shard
produced zero rows.

Run867 tested the better curriculum shape: export `all_stable` rows, not just
stable red overrides. On one `20`-state smoke it produced `8` replay rows:
`5` stable red-switch labels and `3` stable base-action anchors. This is a
healthier pretraining target because it teaches both "responsible red rescue"
and "stay with the led-color/base move when the belief rollout says that is
still best."

Run868 scaled that `all_stable` belief-RFT shape into a source-diverse bank:
`57` replay rows from `145` audited states across `6` sources. The label mix is
now useful for curriculum work: `34` red-switch labels and `23` base-action
anchors, with every row exposing exactly `2` value-labeled candidates and no
fabricated risk/paradox labels. This makes the modern LLM-style loop concrete
inside the game: sample legal moves, score belief-rollout continuations across
actor-consistent hidden worlds, and make the stable better move more likely.

Run869 added explicit value-only training support for that replay and ran a
held-out action-set Transformer smoke. It fit training sources well
(`0.996825` train value correlation), but held-out value transfer was weak
(`0.263082` validation value correlation, value MAE `1.047246`) and the old
risk-aware gate selected `0` deployable overrides. Do not live-screen, promote,
or install Run869. The answer to the Strawberry/RFT question is now: yes,
possible and wired locally, but current scale/generalization is not enough.
Next should either scale the source-diverse `all_stable` belief bank by an
order of magnitude or add a value-only preference-accuracy gate before any live
survival screen.

Run870 added that value-only preference gate. Unlike the old risk-aware live
gate, it ignores the intentionally absent risk/safe labels and asks whether a
model can make high-precision value-positive overrides while abstaining on
base-anchor states. On the small Run868 bank it nearly worked but did not pass:
best held-out source-5 setting had `5` overrides, `80.0%` precision,
`66.7%` recall, `85.7%` base-target abstention, and `76.9%` decision accuracy;
the gate failed because it requires at least `8` overrides and `85%`
precision.

Run871 then scaled the `all_stable` belief-RFT bank to `163` rows from `370`
audited states across `10` sources. The final replay has `94` red-switch labels
and `69` base-action anchors; every row has exactly `2` value-labeled
candidates, zero risk/paradox labels, and globally unique row-group IDs after a
builder provenance fix. The data-collection side of the local Strawberry/RFT
loop is now operational at small training scale.

Runs872-875 tested whether that scale is enough for source-held-out action-set
Transformers. It is not. On held-out fresh sources `7,8,9`, the baseline
Transformer reached only `69.2%` precision; base-anchor weighting made it worse
at `60.0%`; a smaller regularized model improved to `77.8%`; and stricter
thresholds topped out at `80.0%` with `10` overrides. All fail the `85%`
value-only preference gate, and none should be live-screened or promoted. The
current bottleneck is no longer bank plumbing; it is source-stable
generalization. Next useful work should add source-invariant calibration,
validation-driven early stopping/snapshot selection, or richer belief/context
features that explain why an apparent red rescue is good in one source and a
false base override in another.

Run876 tested one of those cheap representation fixes by adding the existing
within-round context summary to the Run875 regularized Transformer recipe. It
failed badly: held-out value correlation fell to `0.073540`, the best threshold
overrode all `36` validation states, and precision collapsed to `55.56%`
with `16` false base overrides. Run877 added a proper value-only calibrated
gate: train on sources `0..4`, calibrate on `5,6`, and test the fixed threshold
on `7,8,9`. That also failed. Calibration itself reached only `75.0%`
precision (`8` overrides), and the fixed calibrated threshold transferred to
`47.37%` precision on final sources (`19` overrides, `10` false base
overrides). A new reusable audit script,
`quantum_cat_value_only_gate_audit.py`, records the exact true overrides, false
base overrides, and missed positives; Run875's audit shows the crisp current
mistake pattern: `8` true overrides, `2` false base overrides, `12` missed
positives, and `14` correct base abstains on the `36` held-out states. The two
bad Run875 overrides are early-trick red switches away from a playable led suit,
while many missed positives are also red switches. So the model has learned
"red switch can be rescue" but still lacks the source-stable boundary for when
burning the led lane is actually correct. Next should not promote; it should
target that boundary directly, either by adding same-state contrast features for
the red-switch candidate versus the follow-led base, or by collecting a larger
bank where those false-red and true-red cases are deliberately balanced.

Runs878-879 then made that boundary test explicit with
`quantum_cat_red_switch_boundary_trainer.py`: one example per Run871 state,
where the decision is exactly "take the off-led red switch" versus "stay with
the follow-led base." The direct MLP over the existing same-state pair features
still failed source-held-out transfer. Run878 memorized train rows
(`0.984803` positive probability on train positives vs `0.013807` on train
base) but reached only `57.14%` held-out precision with `14` overrides.
Run879 added inverse-source weighting, stronger regularization, a smaller
network, and 2x base weighting; it got worse (`45.0%` precision, `20`
overrides). A leave-one-source Torch logistic diagnostic was also near chance
on most sources, with held-out AUCs usually around `0.33-0.61` except one tiny
source. This is now stronger evidence that the current observable feature/label
shape is not enough to learn the source-stable true-red versus false-red
boundary. Next should change data, not just model shape: collect repeat-validated
false-red and true-red examples under common continuation sets, increase belief
samples/candidates for the boundary rows, and/or add explicit common-continuation
features that describe why the same red-switch-looking move is good in one deal
and bad in another.

Run880 started that data-change path by exposing stronger belief-bank knobs:
`--belief-source`, `--min-belief-action-coverage`, `--stable-margin`, plus
summary `label_mix` and optional minimum red/base stop targets. A strict 5p
smoke with `0.90` action coverage and `0.50` stable margin produced `9` rows
from `32` audited states: `6` red overrides, `2` base targets, and `1` other
override, with `9 / 9` unique row groups. This proves the stricter belief-RFT
label source is viable, but it is still far too small and not balanced enough
to train. Next should run the same collector longer with explicit minimum red
and base class targets before another source-held-out preference gate.

Runs881-885 tried to make that balanced bank real. Run881 showed the stricter
`0.50` margin shape was too sparse (`0` stable rows in an 8-state partial).
Run882 used a practical `0.25` margin and produced rows, but still no base
anchors: `7` rows from `24` states, label mix `3` red / `0` base / `4` other.
The probe now has explicit `stable_base` and `stable_red_or_base` filters, and
the builder exposes source-behavior knobs. Run883 and Run884 then mined
stable-base rows directly, with and without source exploration; each found only
`1` base target. Run885 combined the diagnostic artifacts into `9` rows:
`3` red overrides, `2` base targets, and `4` other overrides from `80` audited
states. This is not trainable, but it proves the current bottleneck precisely:
base/follow-led-correct anchors are rare under this state source. Next should
change state selection to deliberately mine high-confidence base-correct
follow-led states before another boundary model is trained.

Runs886-889 tested that state-selection change. Switching stable-base mining to
`table_liveness` scoring worked: Run886 found `13` base targets from `96`
audited states, and Run887 scaled to `42` base targets from `264` audited
states. Those anchors were then mixed into the Run871 all-stable belief bank
with source indices offset so they were train-only. Run888 improved the
high-threshold held-out shape (`5` overrides at `100%` precision, mean value
delta `+2.35`) but still failed the gate because recall/decision accuracy were
too low; Run889's positive-heavy rebalance made transfer worse. Do not
live-screen, promote, or install either model. Table-liveness base anchors are
now a reusable data source, but the next step should collect true-red and
false-red/base cases under the same scorer and train an action-level advantage
target, not another loss-weight tweak on the same mixed bank.

Runs890-895 did exactly that and found the first promising same-scorer data
shape. Run890 and Run891 showed that generic `stable_red_or_base`
table-liveness mining is still base-heavy (`94 / 8` then `140 / 19`
base/red), while Run892 showed stronger source exploration alone is too sparse
(`6` red rows from `288` audited states). The key fix was an actual-red
prefilter: `--require-actual-best-red-switch` plus
`--require-actual-best-overrides-base`, which skips generic states before
belief scoring unless the cheap actual table-liveness ranker says a red switch
beats the base/follow-led action. Run893 smoked this at `14` red rows from
`48` audited red-opportunity states; Run894 scaled it to `41` all-red rows from
`144` audited red-opportunity states. Run895 mixed those with Run887's `42`
base anchors and trained two source-held-out balanced Transformers. Both were
precise but coverage-limited: split A had `4` overrides at `100%` precision,
`0` false base overrides, `50%` recall, `80%` decision accuracy; split B had
`6` overrides at `100%` precision, `0` false base overrides, `60%` recall,
`83.3%` decision accuracy. The formal gate still fails because it requires at
least `8` held-out overrides. Do not live-screen or promote Run895. Next should
scale actual-red-prefilter positives and table-base anchors to `100+` each,
then rerun the same held-out gate with enough positives for the coverage
threshold to mean something.

## Where we are (verified baseline as of 2026-06-12 - do not re-derive)

- 3p champion (unchanged since 2026-06-10): `az_runs/best_3p_policy_checkpoint.pt`
  (Run604 step50, SHA-256
  `f15a7dbd9c259b72c5be6b687f6945b52242ca2c506f9f97c3b4426071b2839f`),
  deployed with the aggressive liveness shield
  (`liveness_shield?liveness_shield_base_mode=policy,liveness_shield_phases=play,liveness_shield_min_open_slot_delta=0,liveness_shield_min_public_damage_delta=0,liveness_shield_max_policy_log_gap=-1.0`).
  Shielded survival `38.5%` at m200 (the `<40%` gate PASSES in 3p); raw
  `policy` mode is the strength-priority dial (+0.86/match over the prior
  champion, 600-match paired gate).
- 4p: still no champion. Run726 replaced the noisy m10/m30 leaderboard with a
  real m200 ladder, then Run730 moved the best verified survival baseline:
  - Run730 liveness shield:
    `481 / 800 = 60.12%`, CI `[56.73%, 63.52%]`;
  - Run730 raw policy:
    `515 / 800 = 64.38%`, CI `[61.06%, 67.69%]`;
  - Run689 liveness shield port: `540 / 800 = 67.50%`, 95% CI
    `[64.25%, 70.75%]`;
  - Run720 mixed table-liveness + led-preservation Q/risk selector:
    `551 / 800 = 68.87%`, CI `[65.67%, 72.08%]`;
  - Run689 raw policy: `573 / 800 = 71.63%`, CI `[68.50%, 74.75%]`.
  The best m200-verified 4p config is therefore Run730 liveness shield, still
  nowhere close to the `<40%` gate but a real `7.4pp` improvement over Run689
  shielded. Run733 mined strict trajectory-causal pairs from this Run730
  shielded distribution and trained a guarded 20% causal policy dose, but it
  failed m30 (`76 / 120 = 63.33%`) versus same-seed Run730 liveness
  (`75 / 120 = 62.50%`). Run734 then tested existing exact legal-pressure
  shield knobs: exact forced-only was too late (`28 / 40 = 70.00%` at m10),
  while broader exact-pressure improved a same-seed m30 screen
  (`76 / 120 = 63.33%` public liveness -> `63 / 120 = 52.50%`) but still
  failed the gate and uses hidden hands, so Run730 remains the verified 4p
  baseline. Run735 then built the first 4p oracle-quality table-liveness
  teacher that itself passed m200 (`286 / 800 = 35.75%`, CI
  `[32.43%, 39.07%]`), but the first small policy-stack distillation from its
  all-legal labels did not transfer survival: shielded Run735 m10 was
  `24 / 40 = 60.00%` versus same-seed Run730 shielded `22 / 40 = 55.00%`.
  Therefore Run730 remains the best verified information-legal 4p baseline,
  and Run735 becomes the best 4p oracle label source. Run736 then trained the
  direct action-value/action-risk stacks on the Run735 table-liveness replay:
  offline transfer was excellent (`risk corr 0.9716`, `value corr 0.9756`),
  but the existing `q_policy` selectors still failed live m10 (`65.00%` best
  q-policy screen, and shield-stacked q-policy was worse). So direct aux-head
  transfer is learnable but not yet deployable with the current selector.
  Run737 then added a calibrated `relative_dominance` q-policy selector that
  abstains unless the raw top policy action is clearly riskier than the
  minimum-risk legal action. A same-seed m10 margin sweep still failed
  (`65.00-70.00%`, best `26 / 40`), even while reducing overrides to
  `5.8-12.9%`; so the failure is not merely that `relative` reranks too often.
  Run738 then trained an explicit information-legal `improve_graft`
  teacher-ranker from the Run735 all-legal table-liveness replay. It learned a
  high-precision slice offline, but live coverage was only `3.6-4.4%` of
  considered play decisions and same-seed m10 screens failed (`57.50%` at
  `threshold=0.99`; `52.50%` at `threshold=0.50`, merely tying Run730 public
  liveness on the same seed). Do not repeat small high-margin pairwise grafts
  from the same `2400`-row replay. Run739 then tried the broader/listwise
  version of this transfer by converting the same Run735 replay into soft
  policy targets: it improved offline CE and changed the replay top action on
  `51.7%` of rows, but live same-seed m10 regressed (`60.00%` Run730 public
  liveness control -> `67.50%` Run739 public liveness, raw `72.50%`). Do not
  spend another run on the same static `2400` rows with only policy-target
  weight/temperature tweaks. Run740 then built the larger on-policy and
  late-balanced version of this idea (`6400` soft/listwise rows, best group
  validation CE `0.9093`), but it failed the held-out safe-coverage gate before
  live evaluation: deterministic table-liveness top-1 was worse than Run730 on
  fresh all-play (`37.50%` vs `39.92%`) and late-t4+ (`42.75%` vs `46.58%`)
  rows. The next bottleneck is objective mismatch: probability-mass/CE shaping
  can improve while the deployed argmax action gets worse. Run741 then fixed
  that exact offline objective mismatch with a hard-margin argmax/top-1 loss:
  held-out top-1 improved to `52.92%` all-play and `57.67%` late. But live
  survival still regressed on the same m10 seed: Run741 raw was
  `27 / 40 = 67.50%`, and Run741 public liveness was `25 / 40 = 62.50%`,
  worse than the Run730 public-liveness control (`21 / 40 = 52.50%`). The
  reason is now visible: broad hard cloning of the hidden-info table-liveness
  top action oversteered the information-legal policy toward red/off-led
  token-loss actions. Run741's held-out predicted actions followed led only about
  `52%` of the time and newly lost the led token about `25-30%` of the time,
  versus Run730's `70-78%` follow-led and `4-7%` token-loss rates. Run742 then
  tried the first constrained version, filtering to teacher top actions that
  either followed led or were red switches that won the trick. It improved
  held-out top-1 versus Run730 (`49.08%` all-play and `55.67%` late) but still
  failed the new discipline gate before live evaluation: follows-led fell to
  `64.98%` / `63.80%`, newly-loses-led-token rose to `17.33%` / `13.57%`, and
  off-led token-loss rose to `19.13%` / `16.06%`. Therefore Run730 remains the
  best verified information-legal 4p baseline, and the next bottleneck is not
  just row filtering; the transfer target itself must preserve led-follow
  discipline and allow only narrow, high-confidence public red-rescue
  exceptions. Run743 then implemented that action-level target: default to
  Run730's public anchor action, imitate the teacher only for follow-led
  actions or strict red-rescue exceptions. It fixed the discipline problem and
  nudged held-out top-1 above Run730 (`41.25%` all-play, `49.75%` late) while
  preserving Run730-like follows-led/token-loss rates. But the live effect was
  only noise: Run743 public liveness tied the Run730 control at m30
  (`73 / 120 = 60.83%` vs `74 / 120 = 61.67%`) and raw Run743 was worse
  (`33 / 40 = 82.50%` at m10). Therefore Run730 remains the best verified
  information-legal 4p baseline. Run744 then tried the obvious stronger
  sibling: require nonnegative/positive teacher-anchor table-liveness margin
  for follow-led corrections and oversample those corrections. This preserved
  Run730-like discipline, but did not earn a live screen: `m001_rep3` scored
  only `40.08%` all / `47.75%` late held-out top-1, and the stronger
  `m000_rep3` scored `41.42%` all / `48.58%` late, not clearly better than
  Run743 (`41.25%` / `49.75%`). Run745 then changed mechanism by training an
  `improve_graft` selector against the Run730 public-liveness base action
  rather than against raw policy. The mechanism now works mechanically
  (`improve_base_mode=liveness_shield`, trainer
  `--base-action-source metadata_policy_best`), but the cheap `48`-row shard
  was too small: the best regularized candidate found only `1 / 16` held-out
  override with `100%` precision/safe rate, while the unregularized model
  overfit (`2 / 16` held-out overrides, `0%` precision). Run746 then did that
  scaled test properly: added metadata-grouped validation plus an explicit
  live-screen gate, generated `320`-row and `640`-row liveness-base corpora,
  and trained summary/raw-summary `ImproveMLP` variants. None passed. D320
  found at most `3 / 80` held-out overrides at `66.7%` precision or `2 / 80`
  at perfect precision; D640 found at most `1 / 160` high-precision held-out
  override, while broad thresholds produced many false positives
  (`27 / 160` overrides at only `14.8%` precision). Run747 then audited those
  D320/D640 false positives and missed positives. The obvious public lane/exit
  features are present in the selector snapshots, but the current pairwise
  objective cannot separate hard-negative rescue lookalikes from rare real
  rescue positives: D640 broad threshold `0.50` selected `25 / 160` overrides
  at only `20.0%` precision, while threshold `0.97` selected only `1 / 160`
  override. Run748 then implemented same-state CE and pairwise-margin variants
  of this selector. They changed the failure mode but still failed grouped
  safe coverage: the best D640 high-coverage candidate found `8 / 160`
  overrides at only `25.0%` precision, while the high-precision candidate again
  found only `1 / 160`. Therefore Run730 remains the best verified
  information-legal 4p baseline. Run749 then tested the first architecture
  change: a DeepSets-style legal-set selector that scores all labeled legal
  actions jointly and includes the shielded base action as an explicit abstain
  target. It still failed grouped safe coverage on the same D320/D640
  liveness-base corpus: D640 `balanced` found only `3 / 160` overrides at
  `66.7%` precision, and D640 `base2` found `3 / 160` at `33.3%`. The next
  transfer should change the data source, not keep architecture/weight
  knob-sweeping on the same corpus. Run750 made that data-source change by
  mining failed-selector shadow proposals against the Run730-style liveness
  base. The miner now supports `--improve-base-mode liveness_shield` and caches
  the configured base action. A first `32`-row shadow corpus produced a useful
  contrast mix (`11` POSITIVE, `21` NEGATIVE), but the diagnostic BCE selector
  overfit train (`7 / 24` overrides at `100%` precision) and abstained on every
  grouped validation row (`0 / 8`). Run751 then fixed the next data-shape issue:
  top-k same-state failed-selector candidates are now audited and replay-labeled
  with the base action. The tiny late-trick smoke corpus had only `8` states, but
  each carried `4` audited actions (top-3 proposal candidates + base), produced
  a balanced target mix (`4` POSITIVE, `4` NEGATIVE), and yielded `6`
  same-state pairwise training pairs. Run752 then added explicit trick quotas
  and produced the first non-toy quota-balanced bank: `128` rows, exactly
  `32` each from tricks 4/5/6/7, with `48` POSITIVE target rows and `80`
  NEGATIVE target rows. That fixed collection shape, but not selector
  calibration. The best BCE+stateCE variant found `15 / 32` grouped validation
  overrides with `66.7%` precision and `93.3%` safe rate, below the `>=80%`
  precision / `>=8` override live-screen gate. Run753 added exact trick-gated
  trainer reporting and `improve_trick_numbers` deployment support, then tested
  the promising trick-5 slice as an official filtered gate. It still failed:
  best held-out trick-5 was `9 / 15` overrides, `77.8%` precision, `100.0%`
  safe, below the `>=80%` precision bar; high thresholds overfit train and did
  not transfer. Run754 then scaled the trick-5-only shadow bank to `128` rows
  (`56` POSITIVE / `72` NEGATIVE) and retrained three summary-feature selector
  variants. All failed the official filtered gate: best was BCE+pairwise,
  `9 / 32` held-out overrides, `55.6%` precision, `100%` safe. Therefore
  Run750-754 are infrastructure/data evidence, not promotion candidates. Run755
  then tried the existing DeepSets legal-set selector on the same trick-5 bank;
  it improved only slightly to `10 / 32`, `60.0%` precision, `100%` safe, still
  failing the gate. Next needs better labels (`audit_rollouts > 1`,
  repeated-seed/exact continuation labels) or a deterministic led-token/red-
  rescue criterion, not another scale/threshold/legal-set rerun. Run756 then
  tested the most conservative deterministic red-rescue hypothesis: do not let
  the liveness shield override a base action that spends the led-color token to
  win a still-needed trick. The m30 same-seed diagnostic was exactly identical
  to the Run730 shield control (`80 / 120 = 66.67%`, CI `[58.23%, 75.10%]`),
  with `43` base responsible-red-win opportunities and `0` candidate attempts
  to leave such a base action. So this specific overcorrection is not the
  current 4p failure mode. Run757 then made the led-switch audit
  liveness-shield-aware and measured the actual Run730 shielded config on a
  4p m60 behavior audit. The important finding: when the non-red led color was
  legally available, both raw and shielded Run730 followed it `100%` of the
  time (`0` switches with led color legal). Switches happened only when the
  led-color lane was already unavailable. Therefore the next label target is
  upstream lane depletion: earlier moves that cause future led-color exits to
  disappear, not a one-step "follow led if available" classifier. Run758 then
  split those no-led-color switches by token status. In the shielded Run730
  audit, `82.26%` of switches newly burned a still-available led token because
  no led-color rank/cell was playable. Red was legal on `86.27%` of those new
  token burns and the shield chose red `94.89%` of the time, so local
  red-choice is mostly handled. Run759 then traced those future
  token-available/no-led-action bottlenecks back to their direct earlier
  zeroing cause. In a 4p m40 Run730 liveness-shield audit, every event had a
  direct prior action that took the target player/color potential count from
  `>0` to `0` while the target player's token remained available; `160 / 183`
  were caused by another player, `180 / 183` were caused by an action in the
  target color, and `183 / 183` happened while the target player had that rank
  in hand. The next target is therefore table-wide, opponent-aware
  rank/cell/exit protection: avoid claiming a color/rank cell that removes
  another player's last potential exit in that color when their token is still
  available. Run760 turned that into the first dense training artifact:
  `1304` anchored replay rows from a 4p m40 Run730 liveness-shield behavior
  bank. `647 / 1304 = 49.62%` rows changed the policy target, exactly matching
  rows where the deployed action zeroed another player's future exit, and
  `1304 / 1304 = 100%` label-best actions avoided other-player exit damage.
  The labels are anchored to avoid the old over-red failure: policy targets
  only move when the deployed action damages another player's exit, and safe
  follow-led actions beat red among ties. Run761/762 then tested the two
  obvious transfer paths and rejected both before live evaluation. Run761's
  policy-stack dose could drive replay other-exit damage down to `38.50%`, but
  only by dropping follows-led to `76.99%` and raising newly-loses-led-token to
  `16.56%`, recreating the known over-red/off-led failure. Run762's frozen
  policy + aux-stack route learned correlations by step80 (`action-value corr
  0.5456`, risk corr `0.5320`), but threshold/relative q-policy selectors
  barely overrode (`0.15-0.23%`) and additive selectors only nudged replay
  damage from `49.16%` to `49.00%` while adding damage on rows the anchor had
  handled. Therefore Run730 remains the best verified information-legal 4p
  baseline, and the next cell-exit attempt needs a different target shape, not
  a longer replay fit. Run763 then tested that different deployment shape with
  a trick-3-only `improve_graft` trained from Run760. Offline filtered gates
  looked strong (`42 / 78` held-out trick-3 overrides at `92.86%` precision /
  `95.24%` safe; stricter threshold `37 / 78` at `100%` / `100%`), but
  unguarded live play failed badly (`70.00-72.50%` at m10). Override samples
  showed why: the graft was replacing led-color follows, especially led-yellow,
  with ultra-low-policy-probability red plays. Run763 added an opt-in
  improve-graft lane guard (`improve_guard_led_follow=true` plus urgent-red
  gating). This made the selector safe but not useful: guarded m30 tied the
  Run730 control exactly (`76 / 120 = 63.33%` vs `76 / 120 = 63.33%`) while
  firing only `17 / 480` trick-3 overrides. Therefore Run730 remains the best
  verified 4p baseline; future learned grafts should train/deploy with
  led-lane preservation and red-urgency constraints from the start. Run764
  baked those constraints directly into the Run760 improve-graft labels:
  `6965` candidates were forced led-lane negatives, including `3623` raw
  positives, and the offline trick-3 gate passed (`23 / 95` held-out
  overrides at `100%` precision / `100%` safe). Live survival still did not
  move: same-seed m30 tied Run730 exactly (`84 / 120 = 70.00%` for both) while
  firing only `28 / 480` trick-3 overrides. The all-play deployment dial also
  failed on m10 (`27 / 40 = 67.50%` versus same-seed control
  `26 / 40 = 65.00%`) despite firing `50 / 1130` considered play decisions.
  Therefore Run730 remains the best verified 4p baseline, and the next
  cell-exit experiment needs broader
  trajectory credit assignment or a deployment mechanism that acts earlier/more
  often without losing led-lane discipline. Run767 then scaled the led-rank
  continuation slice to `182` rows and showed the signal is real but brittle:
  label-best changed `127 / 182 = 69.8%`, but independent and led-shaped
  `ImproveMLP` selectors failed grouped validation. Run768 tried a DeepSets
  legal-set selector on that corpus; it fit all rows but still found only
  `2 / 48` high-precision held-out overrides in the led-shaped model. Run769
  changed the data source by adding broader trick `2-5`, legal-count `<=10`
  continuation shards, including an exploration shard with selected switches.
  That produced append `286` rows and balanced `264` rows, but both legal-set
  models failed the live-screen gate: balanced topped out at `1 / 7 = 14.3%`
  held-out precision, append at `2 / 6 = 33.3%` and only `5 / 17 = 29.4%`
  among thresholds with at least `8` overrides. Therefore Run730 remains the
  best verified 4p baseline; do not keep scaling the same led-continuation
  legal-set objective without stronger cross-source labels or trajectory-level
  survival/liquidity supervision. Run770 then tried the trajectory-level route
  directly by training current-Run730 action-value/action-risk auxiliary heads
  on sparse terminal-liveness labels. The risk head learned a modest independent
  signal (best validation AUC `0.7219`), but live selector transfer failed:
  strict q-risk gates tied the same-seed Run730 control at `26 / 40 = 65.00%`
  because they barely overrode, while looser gates regressed to
  `30-31 / 40 = 75.00-77.50%`. Therefore Run730 remains the best verified
  information-legal 4p baseline; do not promote or m200-screen Run770.
  Run771 then audited whether a survival-first repeated-rollout teacher could
  produce stable, recall-consensus action corrections before training. It did,
  but only sparsely: `13 / 22` evaluated rows were high-margin, yet only `4`
  stable rows survived top-1 stability plus all-legal recall consensus. Those
  `4` rows were real policy corrections, with forced-audit margins of
  `+1.0`, `+3.0`, `+5.0`, and `+5.0`, and paradox deltas `0`, `-1`, `0`, and
  `-1`, but the replay is far too small to train. Run772 tried scaling that
  exact conservative criterion to a balanced `12`-row bank (`4` each from
  `near_paradox`, `hard`, and `mid_rank_follows_led`), collected the `72`
  target states, then was manually stopped after `12` processed states because
  it had `6` high-margin rows but `0` stable rows after `923.319s`. Therefore
  broad repeated-rollout stable-label mining is currently diagnostic, not the
  next main training substrate. Run773 then checked whether existing
  deterministic public liveness keys could replace that sparse teacher source:
  the public key repaired plain `heuristic_safe14` on the same m30 seed
  (`91 / 120 = 75.83%` -> `72 / 120 = 60.00%`) but only tied/slightly nudged
  the Run730 neural liveness control (`74 / 120 = 61.67%`), while late-only
  min-trick variants regressed to `70.00-75.00%`. Run774 closed the remaining
  public exit/liquidity shields: `exit_liquidity_shield_safe14` was
  `93 / 120 = 77.50%`, and `liquidity_feasibility_shield_safe14` was
  `91 / 120 = 75.83%`. Therefore Run730 remains the best verified
  information-legal 4p baseline, and existing one-step deterministic-public
  heuristics are not the next main substrate either.
- 5p: Run731 replaced the fuzzy `~44-64%` note with a real m200 ladder, and
  Run732 rejected blind survived-round q-policy behavior distillation. Run775
  then confirmed a much stronger hidden-info exact late-table liveness teacher:
  `202 / 1000 = 20.20%` at m200, but a small public policy-stack behavior
  clone from its survived rounds failed to transfer (`48.67-60.67%` at m30
  across raw/q-policy student variants). Run813 then confirmed a narrower
  exact late-exit liveness teacher on the Run779 step40 checkpoint: tricks
  `5..6`, no policy gap, exact legal pressure, `321 / 1000 = 32.10%` at m200
  with CI `[29.21%, 34.99%]`. This is an oracle teacher / training-label
  source because exact legal pressure reads the full simulator state, not an
  information-legal deployed bot. No 5p champion. Best verified
  deployable 5p config is still effectively Run600 q-policy/raw:
  - Run600 5p q-policy:
    `538 / 1000 = 53.80%`, CI `[50.71%, 56.89%]`;
  - Run600 5p raw policy:
    `539 / 1000 = 53.90%`, CI `[50.81%, 56.99%]`;
  - `heuristic_safe15`: `633 / 1000 = 63.30%`;
  - `oracle_safe`: `717 / 1000 = 71.70%`.
  Run732 q-policy behavior-distilled student failed confirmation at
  `563 / 1000 = 56.30%`, CI `[53.23%, 59.37%]`, worse than the Run600
  q-policy baseline despite `0.9096` behavior-validation top-1.
  The old q-policy selector does not materially beat raw policy, but both
  beat own-hand-safe built-ins. Run775's teacher is an oracle/label source,
  not an information-legal deployable policy. Run776-784 then showed that
  broad exact-table aux labels, red-switch-only labels, small late-exit labels,
  dense low-exit warning labels, hard selected-vs-best contrasts, binary
  all-legal warning labels, direct policy-veto training, and unlimited q-policy
  overrides all fail. Run785 changed source to 5p upstream cell-exit
  protection labels. The source is feasible (`160` focused damage rows with
  `159 / 160` policy-target changes and `158 / 160` no-other-exit-damage
  label-best actions), but transfer still failed: frozen aux-head q-policy
  regressed to `86 / 150 = 57.33%` at m30, and a legal-set/base-abstain
  diagnostic found `15 / 79` held-out overrides at only `60.0%` precision
  (`0` live-screen-eligible thresholds). Run779 reset-head step40 remains the
  best non-promoted 5p public m30 survival artifact at `70 / 150 = 46.67%`.
  Run786 tried the obvious abstention calibration fix (`target_base_weight=4`)
  and failed too: `8 / 66` held-out overrides, `50.0%` precision, `87.5%`
  safe, `0` eligible thresholds. Do not keep weight-sweeping the current
  legal-set objective on the same Run785 corpus. Run787 then audited the
  actual one-step labels with reconstructed same-policy continuations:
  the Run785 legal-set selected overrides had `0 / 15` survival-improving
  actions, and the first `40` one-step target changes had only `5 / 40`
  survival-improving targets. This explains the transfer failure: unfiltered
  5p cell-exit labels are mostly too local/noisy for training. Run788 then
  built the continuation-filtered replay exporter and confirmed the mechanical
  path works (`40` rows, `10 / 40` continuation-supported overrides,
  `30 / 40` hard-negative rejected overrides), but the first tiny filtered slice
  is not a training substrate: strict led-lane shaping left only `3 / 40`
  deployable positives and the legal-set diagnostic failed; the no-lane-force
  control also failed (`3 / 12` held-out overrides, `0.0%` precision/safe).
  Run789 scaled that path to all `172` Run785 one-step target changes:
  `46 / 172 = 26.74%` were supported by `risk_or_score`, only
  `26 / 172 = 15.12%` actually reduced any-player paradox risk, and
  `126-146 / 172` became hard negatives depending on support mode. The larger
  corpus still failed every grouped legal-set gate. Strict led-lane shaping
  left only `13` deployable positives under `risk_or_score` and `7` under
  `risk_only`; removing the lane guard produced many false/riskier overrides.
  Run790 repeated-validated the `26` any-risk-safer candidates at common8 and
  all `26 / 26` stayed continuation-supported, but strict led-lane shaping left
  only `7 / 26` deployable positives. Led-compatible mining added two more
  stable strict-compatible positives (`9` total), and the high-confidence mixed
  replay still failed the grouped strict legal-set diagnostic (`3 / 61`
  overrides, `33.33%` precision/safe, mean risk delta `+0.6000`). Next: scale
  led-compatible proposal mining and repeated-validate any-risk-safer
  candidates until there are tens of held-out strict positives, not another
  threshold/weight sweep. Run791 did that scale-up on a fresh 480-row 5p
  policy-damage shard: `198` led-compatible best-per-state candidates,
  `32 / 198` any-player-safer at common1, and all `32 / 32` stayed stable at
  common8. This lifts the strict-compatible repeated-positive bank from Run790's
  `9` to roughly `41`, but the same strict legal-set/base-abstain diagnostic
  still failed (`7 / 55` overrides, `42.86%` precision, `71.43%` safe). The
  next blocker is selector discrimination/objective shape, not proof that
  strict-compatible repeated positives exist. Run792/793 then showed the
  binary abstainer family can look clean offline and still over-fire live, even
  after guard-aligned DAgger hard negatives. Run794/795 moved those rows into a
  multi-source legal-set/base-abstain selector and added a positive head; this
  improved calibration but still produced only `3` held-out overrides at
  `66.67%` precision, below the live-screen gate. Run796 tried stable
  q-teacher labeling from balanced states; collection worked, but scoring was
  too slow for the interactive 5p loop (`4 / 16` candidate-consensus states in
  `338s`, `0` stable exports). Run797 switched to deterministic exact
  table-liveness labels. Its action-value best mostly followed led
  (`635 / 640`) and rarely burned the led token (`5 / 640`), but the direct
  table-liveness policy target still burned tokens too often (`107 / 640`), so
  the replay is useful as value/risk data, not as a policy-imitation target.
  Run798's smoke legal-set was clean but coverage-starved (`3` held-out
  overrides, `100%` precision/safe); Run799's scaled legal-set failed
  calibration (`6` held-out overrides, `50%` precision/safe). No 5p live screen
  was warranted after Run799. Run800 added the stricter test we wanted:
  explicit source-holdout validation and a fresh independent deterministic
  table-liveness shard. The new shard reproduced the label property
  (`1636 / 1642` value-best actions followed led; only `6 / 1642` burned led),
  but both cross-shard selectors failed: train Run797 -> validate Run800 found
  `55` overrides at `29.09%` precision / `49.09%` safe, and train Run800 ->
  validate Run797 found `12` overrides at `41.67%` precision / `58.33%` safe.
  This rejects the current legal-set selector family under a real source-holdout
  gate; more deterministic positives alone did not fix transfer. Run801 tested
  the alternative target decomposition suggested by Run800: learn smooth
  `value_delta`/`risk_delta` heads and use a deterministic abstention rule.
  The smooth quantities transfer extremely well across shards
  (`value_corr=0.985-0.987`, `risk_corr=0.988-0.991`), but the override
  boundary still fails: train Run797 -> validate Run800 gives `25` overrides
  at `24.00%` precision / `28.00%` safe, and train Run800 -> validate Run797
	  gives `11` overrides at `45.45%` precision / `45.45%` safe. This means the
	  table-liveness branch needs hard-negative boundary data, not more smooth
	  value/risk regression or threshold sweeps. Run802 tested the obvious
	  synthetic version: pairwise rank true positive overrides above the base
	  action and led-lane-burn lookalikes. It still failed source-holdout:
	  train Run797 -> validate Run800 collapsed to `6 / 1642` strict-margin
	  overrides at `33.33%` precision / `66.67%` safe, and train Run800 ->
	  validate Run797 reached only `10 / 640` overrides at `60.00%` precision /
	  `80.00%` safe. Run803 then did mine actual selected false positives and
	  validated on a fresh third shard. The cross-fit version overfit the mined
	  shards (`96.15%` precision/safe on the mine slice) but reached only
	  `25 / 1657` validation overrides at `56.00%` precision / `64.00%` safe.
	  Run804 then tested the next obvious representation fixes: multi-source
	  delta training, raw+summary observation context, and source-balanced
	  adversarial training. None passed. Best was the weak source adversary:
	  `26 / 1657` overrides at `46.15%` precision / `46.15%` safe. Therefore
	  selected-error hard negatives, raw observation appends, and simple
	  source-invariance are not enough. Run805 then tried the first true
	  legal-action-set transformer. The full model overfit (`86` train overrides
	  at `100%` / `100%`) and validated at only `52 / 1657` overrides,
	  `32.69%` precision / `40.38%` safe. The regularized model found a tiny
	  clean validation slice (`3 / 1657` at `100%` / `100%`) but missed the
	  `8`-override floor; coverage-qualified best was only `9 / 1657` at
	  `55.56%` / `66.67%`. Run806 then tested the tiny-clean-slice hypothesis
	  with a fourth independent shard and calibrated thresholds from Run803 to
	  Run806. It failed: the calibration side still had only `3` clean overrides,
	  no eligible threshold met the `8`-override floor, and applying the best
	  calibration threshold to Run806 produced `2 / 1660` overrides at `0.00%`
	  precision / `50.00%` safe. The next attempt needs explicit sequence/trick
	  context or a true multi-shard leave-one-source-out model-selection loop,
	  not more threshold sweeps on the same current-state transformer. Run807
	  added cheap within-round aggregate context to the action-set transformer
	  and trained Run797+Run800+Run803 -> validate Run806. It still failed:
	  best validation slice was `50 / 1660` at `30.00%` precision / `30.00%`
	  safe; a stricter diagnostic found only `22 / 1660` at `31.82%` /
	  `31.82%`. The audit showed all selected actions followed led and none
	  lost a token, so the current 5p failure is led-follow rank/exit allocation
	  inside the lane, not off-led lane burn. Run808 targeted that directly with
	  a focused follow-led/no-token-loss intra-lane rank filter. It fit the train
	  shards (`194` train overrides at `100%` / `100%`) but failed Run806:
	  best deployable-coverage slice was only `9` overrides at `33.33%` /
	  `33.33%`, with validation value/risk correlations near zero. The missing
	  signal is likely explicit board-cell/exit scarcity or repeated exact
	  continuation labels for same-state follow-led rank choices, not more
	  capacity or weights on the current feature bank. Run809 added explicit
	  same-state intralane scarcity percentiles/z-scores over follow-led/no-token
	  alternatives. It improved the tiny clean Run806 island from `1` to `3`
	  overrides at `100%` / `100%`, with strong value/risk margins, but still
	  failed deployable coverage: best coverage-qualified slice was `70`
	  overrides at `31.43%` precision / `34.29%` safe. Scarcity features help a
	  little, but the next missing piece is stronger/repeated exact continuation
	  labels for those same intralane alternatives.
- Checkpoints are table-size-specific (observation widths 76/97/113), so 4p
  and 5p need their own promoted checkpoints.
- Run727/728 finished the 4p floor check:
  - `heuristic_safe15`: `625 / 800 = 78.12%`;
  - `oracle_safe`: `631 / 800 = 78.87%`;
  - initial deal-floor audit: `0 / 10000` raw negative deals and `0 / 10000`
    best-discard negative deals.
  Conclusion: the 4p wall is not caused by unavoidable dead deals, and
  own-hand safety is not enough.
- Web human-match logging now exists (Run729), but no real human match pool has
  been collected yet. iOS logging/export remains undone, so "human-level" is
  still unmeasurable.

## Current diagnosis - where we are going wrong (as of Run809)

1. **The m10/m30 4p leaderboard was fiction.** Run720 looked like `52.5%` at
   m10 and `58.33%` at m30, but Run726 measured it at `68.87%` over m200. The
   true best verified pre-E3 4p config was Run689 liveness shield port at
   `67.50%`. Run730 now improves that to `60.12%`, still far above the gate.
   Survival decisions still require m200; small screens are only triage.
2. **Behavior cloning is not enough, even from good hidden-info teachers.**
   Every distillation that ever worked (Run553, Run604) had a teacher at or
   below the target rate. Run730 showed one shield-behavior cycle can transfer
   live 4p survival even from a weak teacher, but the new best teacher is still
   `60.12%`, not an oracle-grade source. Run775 finally confirmed a strong 5p
   hidden-info teacher at `20.20%`, but a small public behavior clone still
   failed the m30 gate. Run776 confirmed dense exact-table action labels are
   trainable, but broad policy-moving transfer regressed live survival; the
   useful signal was aux-only selector training, whose best 5p screen was
   still only `72 / 150 = 48.00%` at m30. Run777 then sharpened the replay to
   red-switch contrast rows with nearly zero non-red switch target mass; it
   still only tied `48.00%` at step20 and got worse at step10/40. Run778 then
   targeted late trick-4-to-6 table-exit allocation directly from the Run776
   step20 distribution, but the aux-only m30 screens regressed to
   `82 / 150 = 54.67%` and `88 / 150 = 58.67%`. Do not assume a blind cycle 2
   will compound; Run605 gave gains back in 3p, and Run775-778 say
   exact-teacher transfer needs causal upstream trajectory labels or veto-style
   teacher data, not just survived action imitation, broad all-action policy
   movement, a standalone red-switch curriculum, or a small late-exit aux
   replay from an already-tight distribution. Run779 is the first 5p public
   transfer to improve the best m30 screen after Run776: sparse causal
   low-exit warning labels plus reset-head training reached
   `70 / 150 = 46.67%`, still failing the gate but directionally better.
   Run780 then scaled/densified that warning family to `371` rows and `1689`
   all-legal action labels, but held-out top-1 collapsed to only `18-23%` and
   live m30 regressed to `51.33-54.67%`. Run781 showed compacting those dense
   rows to the `4.55` labeled candidates per state does not fix the learning
   problem (`16-18%` held-out top-1, no live screen). Run782 extracted hard
   selected-vs-label-best contrasts from the dense outcomes and learned them
   offline (`86.36%` held-out top-1), but live m30 failed badly
   (`56.67-60.67%`) because two-action contrasts left too many deployed legal
   actions unconstrained. Run783 relabeled every dense legal action as blunt
   safe/paradox binary targets, but offline ranking collapsed (`5-7%` top-1)
   and live m30 still failed (`55.33-63.33%`). Do not repeat dense soft
   labels, selected-vs-best hard labels, or binary all-legal labels with the
   current q-policy selector as the fixed deployment mechanism. Next 5p work
   tested that deployment/training-shape hypothesis directly: Run784 converted
   anchor-risky/safe-labeled warning states into disciplined direct policy
   veto targets and also tried removing the q-policy policy-gap cap. Both
   catastrophically oversteered to `150 / 150 = 100%` m30 paradox rounds,
   often collapsing earlier at trick 3-5. Therefore the dense warning outcomes
   are not safe as direct policy targets or unbounded selector targets. The
   next useful 5p work needs stronger hard-negative coverage for low-policy
   safe-looking alternatives, a legal-set selector with explicit abstain and
   grouped held-out gates, or a new data source that labels upstream
   lane-preserving choices before the late warning state. Run785 tested that
   new upstream cell-exit source. Data collection worked, but both transfer
   mechanisms failed: aux-head q-policy regressed live to `57.33%`, and
   legal-set/base-abstain grouped validation failed precision (`60.0%` best,
   `0` eligible thresholds). The next 5p bottleneck is label
   validity/calibration for hard-negative lookalikes, not another q-policy knob
   or naive legal-set retrain. Run786 confirmed that the easy abstention
   calibration knob does not fix it: stronger base weighting lowered coverage
   but validation precision fell to `50.0%`. Stop base-weight/safe-loss
   sweeping on this corpus. Run787 added the missing continuation audit and
   found the label-level failure directly: `13 / 15` failed-selector overrides
   were rejected by continuation, and the first `40` one-step target changes
   were rejected `30 / 40`. The next 5p step is no longer "train from
   cell-exit labels"; it is "continuation-filter cell-exit labels, keep only
   candidate-vs-base survival improvements, and export rejected positives as
   hard negatives." Run788 implemented that exporter and proved the artifact
   shape, but the first `40`-row slice is still too sparse/noisy for training:
   strict led-lane shaping leaves only `3` deployable positives, and both the
   strict and no-lane-force legal-set diagnostics select only false/riskier
   held-out overrides. The next step is larger/source-balanced continuation
   filtering with enough held-out deployable positives, not deployment. Run789
   did the all-172 `common1` version and confirmed the deeper problem: most
   one-step targets are still rejected (`126 / 172` under `risk_or_score`,
   `146 / 172` under `risk_only`), strict led-lane shaping leaves too few
   deployable positives (`13` or `7`), and no-lane controls select false/riskier
   overrides. The next useful 5p label source is higher-confidence
   repeated-rollout continuation data for the any-risk-safer candidates plus
   more led-lane-compatible positives. Run790 then confirmed the
   higher-confidence part: all `26 / 26` common1 any-risk-safer candidates
   survived common8 validation, so the core survival signal is real. But only
   `9` strict-compatible stable positives remained after led-lane shaping and
   led-compatible mining, so the grouped strict selector still failed. The
   current blocker is positive yield under the deployable objective, not
   common1 label noise. Run791 then improved that positive yield materially:
   a fresh 480-row source produced `32` common8-stable led-compatible positives,
   giving about `41` strict-compatible repeated positives including Run790. But
   the current legal-set selector still failed grouped validation, so the
   bottleneck has moved from "can we find stable deployable positives?" to "can
   the selector separate those positives from hard-negative lookalikes?"
3. **The 4p wall is trajectory collapse, not initial deal doom.** Run728 found
   `0 / 10000` negative initial deals under the exact own-hand liveness oracle,
   while Run727 showed exact own-hand-safe play still paradoxes around `79%`.
   The missing skill is table-wide late exit allocation after avoidable lane
   burns, especially trick-6/7 forced legal-count-1 states.
4. **The human-Elo workstream now has web logging, but no data.** Run729 added
   durable web JSONL logging for completed human-vs-bot games. The next gap is
   actual human games and/or matching iOS export, not more schema design.
5. **5p has a real oracle teacher, but no public transfer yet.** Run731's best
   deployable m200 rate is `53.80%`, which is `13.8pp` above the gate. Run732
   showed that survived-round cloning from this still-failing teacher can
   imitate well but degrade live survival to `56.30%`. Run775 confirmed exact
   late-table liveness can drive 5p to `20.20%` at m200, but that teacher uses
   hidden/table information and a small public policy-stack clone still failed
   at `48.67-60.67%` on m30. Run776 converted the teacher into dense
   action-value/risk labels; broad policy-moving training failed
   (`56.67-60.00%` m30), while aux-only selector transfer improved to
   `72 / 150 = 48.00%` m30 but still missed the gate and was not promoted.
   Run777 confirmed that the red-switch exception is not the main missing
   piece: the best checkpoint chose red on `98.40%` of switch opportunities
   where red was legal, and red switches won `97.53%` of resolved tricks, yet
   paradox rate stayed high. The first-paradox shape is still forced
   legal-count-1 collapse, mostly trick 6 in 5p. Run778 confirmed that simply
   mining late table-exit rankings from trick 4-6 and fitting frozen aux heads
   is not enough: step20/step40 q-policy screens regressed to `54.67%` and
   `58.67%`. The next 5p attempt needs explicit earlier causal labels for
   moves that reduce future exits, or a runtime shield/teacher whose vetoes can
   be distilled. Run779 tested that causal warning direction and moved the best
   public m30 screen to `46.67%`; it is not enough, but it is the current best
   5p transfer direction. Run780 tried the dense all-legal version and
   regressed (`51.33-54.67%` m30); Run781 compacted the same dense labels and
   still failed offline. The dense outcomes are useful analysis material but
   should be converted into harder contrast/veto targets before another
   transfer run.
6. **4p causal pairs are real, but pairwise policy nudges are still too weak.**
   Run733 mined `142` strict survived causal pairs from the current best
   Run730 liveness-shield distribution. The data was exactly the upstream
   shape we wanted (`136 / 142` selected actions followed led; safe actions
   split `49` follow-led and `93` red-switch rescues, mostly trick 3). A
   guarded behavior+causal policy dose preserved lane discipline, but did not
   improve live survival at m30. The remaining problem is not simple avoidable
   non-red lane burning; it is table-wide future exit allocation.
7. **Immediate forced-exit intervention is too late.** Run734 tested the
   existing exact legal-pressure liveness shield. Exact forced-only barely
   fired (`5 / 1139` decisions at m10) and got worse; broader exact-pressure
   helped at m30 (`63.33%` -> `52.50%`) but still failed, used hidden hands,
   and left first paradoxes as trick-7 forced legal-count-1 collapses. The
   useful target is broader all-legal continuation pressure, not a narrow
   "only stop an already-forced player" guard.
8. **We finally have a sub-40 4p teacher, but not a student.** Run735's
   hidden-info table-liveness teacher passed m200 at `35.75%`. The label
   source is real. The current transfer path is not: a `2400`-row guarded
   policy-stack clone improved offline teacher top-1 (`43.75%` anchor ->
   `54.37%`) but failed live m10 (`60.00%`). The next bottleneck is
   distillation/representation/data coverage, not whether table-wide future
   exit allocation is the right signal.
9. **The risk/Q surface is learnable, but q-policy is not calibrated.** Run736
   froze the policy/body and trained only the separate action-value and
   action-risk stacks on Run735 labels. Held-out replay metrics were excellent
   (`risk corr 0.9716`, `risk Brier 0.0061`, `value corr 0.9756`), but live
   q-policy screens still failed (`65.00%` best m10) and q-policy-under-
   liveness was worse than public liveness. The selector either over-applies
   relative risk, fires too late in threshold mode, or creates a harmful base
   distribution before the liveness shield can help. Run737 checked the
   obvious abstention variant and it still failed (`65.00-70.00%` m10), so do
   not spend more cycles on scalar q-policy knob sweeps for this checkpoint.
10. **Sparse teacher-ranking is not enough either.** Run738 did the next
    obvious thing after Run737: a deployable information-legal pairwise
    teacher-ranker/abstainer trained from Run735's table-liveness action
    labels. Offline precision was good, but the model only found a tiny
    high-confidence live slice (`3.6-4.4%` overrides), and same-seed m10 did not
    beat Run730 public liveness. The next useful transfer needs broader/listwise
    supervision or a much larger on-policy corpus that changes enough upstream
    lane-allocation decisions to affect trick-6/7 collapse.
11. **Static soft/listwise replay transfer is not enough either.** Run739
    broadened the Run735 signal into soft policy targets and changed the replay
    top action on `51.7%` of rows. That solved Run738's replay-coverage
    problem but not live survival: same-seed m10 regressed from Run730 public
    liveness `60.00%` to Run739 public liveness `67.50%`, with raw Run739 at
    `72.50%`. The bottleneck is now distribution/target quality: the small
    static Run735 replay does not cover enough of the live states where
    Run730/Run739 create trick-5/6/7 table-exit collapse.
12. **Bigger on-policy soft targets improve mass, not the deployed action.**
    Run740 fixed the Run739 distribution objection by labeling fresh
    Run730-liveness on-policy states and oversampling trick-4+ exit-allocation
    positions. Offline soft CE improved strongly, but held-out deterministic
    table-liveness top-1 got worse than the Run730 anchor. This is now an
    objective mismatch, not only a data-size mismatch: CE/probability mass can
    move toward the teacher while the argmax action still fails the
    table-liveness gate used by deployment.
13. **Unconstrained hard top-action cloning learns hidden-info red burns.**
    Run741 did exactly what Run740 asked for - optimize deterministic
    table-liveness top-1 directly - and passed that offline gate decisively.
    But live m10 got worse, and the held-out feature audit explains why:
    the hidden-info teacher often prefers red/off-led token-loss moves that are
    table-good with oracle knowledge but unsafe as an information-legal
    instinct. The next transfer objective must be stratified: follow led when
    the lane should be preserved, and only learn red-switch exceptions when
    they are high-confidence, responsible, and preferably trick-winning.
14. **Simple row-filtered constrained cloning is still too loose.**
    Run742 filtered the table-liveness top-action replay to teacher actions
    that followed led or were red switches that won the trick. That improved
    held-out top-1 over Run730, but still degraded the public action shape:
    follows-led dropped to `64.98%` all / `63.80%` late and newly-loses-led
    rose to `17.33%` all / `13.57%` late. The next target must constrain the
    action distribution itself, not merely the rows: preserve Run730-like
    led-follow behavior unless a public, high-margin red-rescue exception is
    available.
15. **Action-level anchoring preserves discipline but the first dose was too weak.**
    Run743 defaulted each target to the Run730 anchor action and only imitated
    the table-liveness teacher for follow-led actions or strict red-rescue
    exceptions. This fixed the Run741/742 over-red drift: held-out follows-led
    and token-loss rates stayed Run730-like, while top-1 improved slightly.
    But live m30 only tied Run730 public liveness (`60.83%` vs `61.67%`) and
    raw Run743 was worse. The relabeler is useful, but the weak all-row
    anchor-follow dose is retired.
16. **Positive-follow oversampling still does not move argmax enough.**
    Run744 repeated only nonnegative/positive-margin follow-led corrections.
    This kept the held-out action shape disciplined (`70-78%` follows-led,
    `4-7%` newly-loses-led-token), but it did not beat Run743's held-out
    coverage enough to justify a live screen. `m001_rep3` was too weak, and
    `m000_rep3` only nudged all-play top-1 while losing late rows. The current
    policy-stack-only KL-anchored recipe mostly moves probability mass/CE under
    anchored targets; it is not changing the deployed argmax enough.
17. **Liveness-base grafting is promising infrastructure, but tiny validation
    is not evidence.**
    Run745 changed the mechanism: train candidate overrides relative to the
    Run730 public-liveness base action and deploy them after the liveness
    shield. The code path works, but the `48`-row cheap shard left only `16`
    validation states; the best regularized model had exactly one held-out
    safe positive override, while the larger-looking unregularized model failed
    precision. Future liveness-base graft screens need a larger grouped corpus
    and a hard minimum held-out override count before live evaluation.
18. **Scaled liveness-base grafting still does not generalize safe coverage.**
    Run746 added the missing grouped gate and scaled the corpus to `320` and
    `640` rows. This removed the Run745 underpowered-validation excuse, but the
    selector still failed: broad thresholds produced many false positives, and
    strict thresholds abstained almost everywhere. The next bottleneck is
    missing public selector features/objective calibration, not simply row
    count.
19. **The audit points to hard-negative calibration, not absent obvious features.**
    Run747 inspected D320/D640 validation false positives and missed positives.
    Public candidate/base/delta features already include follows-led, red,
    would-win, off-led token loss, public/own slot damage, min open slots, min
    lane surplus, future token-loss pressure, and post-hit forced pressure.
    The failure is that same-looking rescue candidates split into real positives
    and false positives: D320 threshold `0.50` found `17 / 80` overrides at
    `29.4%` precision, D640 threshold `0.50` found `25 / 160` at `20.0%`, and
    D640 threshold `0.97` found only `1 / 160` override. This fed the same-state
    hard-negative / contrastive tests in Run748.
20. **Same-state contrastive losses do not rescue the small `ImproveMLP` selector.**
    Run748 added `state_ce`, `bce_state_ce`, `pairwise_margin`, and
    `bce_pairwise_margin` objectives to the liveness-base trainer. D320 showed
    the tradeoff immediately: pure ranking found coverage but poor precision,
    while hard-negative BCE stayed precise only by abstaining. D640 confirmed
    it: weighted BCE found `1 / 160` high-precision override, BCE+pairwise found
    `8 / 160` overrides but only `25.0%` precision, and BCE+stateCE found
    `4 / 160` at `25.0%`. Do not keep knob-sweeping this small independent
    candidate/base MLP on the same D320/D640 corpora. Next change architecture
    to a true legal-set/listwise selector, collect DAgger hard negatives from
    failed selector distributions, or return to direct policy-stack transfer
    with strict led-discipline gates.
21. **The first legal-set architecture also fails on the same corpus.**
    Run749 added a DeepSets-style selector over all labeled legal actions,
    including base-as-abstain. This was the intended architecture change after
    Run748. It did not pass: D320 balanced found `13 / 80` overrides at `46.2%`
    precision, D320 base-weighted found `3 / 80` at `66.7%`, D640 balanced
    collapsed to `3 / 160` at `66.7%`, and D640 base-weighted found `3 / 160`
    at `33.3%`. The same D320/D640 liveness-base label bank is now exhausted
    for selector work. Next collect DAgger/shadow hard negatives from failed
    selector proposal distributions, or switch back to policy-stack transfer
    with explicit discipline gates.
22. **Shadow proposal mining is now the right data source, but the first batch is too small.**
    Run750 fixed `quantum_cat_improve_graft_miner.py` so failed-selector
    proposals can be mined against the current liveness-shield base, not raw
    policy. The first `32`-row corpus is exactly the missing contrast shape:
    `11` true rescues and `21` hard negatives from actions the failed selector
    actually wanted to play. But a one-candidate-per-state BCE classifier
    overfit and produced `0 / 8` grouped validation overrides. This says "scale
    and enrich this data source," not "live-screen a new selector."
23. **Top-k shadow mining fixes the row shape, but not the scale yet.**
    Run751 added `--proposal-audit-top-k` and records/audits multiple
    same-state failed-selector candidates plus the shielded base action. That
    finally gives pairwise/listwise trainers real same-state hard negatives
    (`6` pairwise pairs in an `8`-state smoke). The smoke also showed candidate
    supply exists across late tricks (`113` trick-4, `109` trick-5, `101`
    trick-6, `16` trick-7 candidates), but collection grabbed all `8` rows from
    trick 4 because it took the first eligible state per match. The next corpus
    needs scale and explicit trick quotas, especially tricks 5-7.
24. **Quota-balanced late mining fixed collection, not selector precision.**
    Run752 added `--proposal-trick-quotas` and collected `32` rows each from
    tricks 4/5/6/7. The label bank is useful (`48` POSITIVE, `80` NEGATIVE), but
    existing summary-feature selector variants still failed grouped validation.
    Pairwise BCE overrode `11 / 32` validation states at only `36.4%` precision;
    BCE+stateCE improved safe rate to `93.3%` and coverage to `15 / 32`, but
    precision was still only `66.7%`. Strict variants recovered precision only
    by collapsing to `1-2` held-out overrides. Per-trick diagnostics show the
    model does not fire on validation trick 7; the false positives are earlier
    (notably trick 4 and the tiny trick-6 slice), while trick 5 is close but
    still below bar (`7 / 9 = 77.8%`). Next work should add trick-aware
    calibration/diagnostics, more quota-balanced data, or richer public features
    for early false-positive rescue lookalikes.
25. **Trick gating is implemented, but the trick-5 slice is not validated.**
    Run753 added `--live-screen-trick-numbers` to the trainer and
    `improve_trick_numbers` to `AZImproveGraftBot`, so trick-specific selector
    deployment is now possible. The first official trick-5 gate still failed:
    `9 / 15` held-out overrides, `77.8%` precision, `100%` safe, with
    `eligible_thresholds=0`. The clean-looking high-threshold all-row result was
    train+validation leakage, not evidence. Do not live-screen trick-gated
    improve-graft models until a trick-filtered held-out gate passes.
26. **Scale-only trick-5 summary-feature screens are falsified.**
    Run754 collected a healthy trick-5-only bank (`128` rows, `56` POSITIVE /
    `72` NEGATIVE, candidate-set recall `1.0`) and gave the official filtered
    gate a real `32`-state validation split. The selector still failed badly:
    BCE+stateCE found `7 / 32` overrides at `28.6%` precision; strict BCE found
    only `1 / 32` at `0%`; BCE+pairwise was best at `9 / 32`, `55.6%`
    precision, `100%` safe. This is no longer just a small-validation artifact.
    Next work needs richer public/action-difference features or a more explicit
    pairwise architecture, not another threshold/scale-only rerun.
27. **The current legal-set architecture also does not solve trick-5 rescues.**
    Run755 trained the existing DeepSets legal-set selector on the same Run754
    trick-5 bank with base-as-abstain. It had real signal but still failed:
    best held-out threshold selected `10 / 32` overrides at `60.0%` precision,
    `100%` safe, `eligible_thresholds=0`. This is only a small lift over the
    best Run754 pairwise result (`55.6%`). The next bottleneck is likely label
    quality or deterministic criteria for led-token preservation/red-rescue, not
    simply "score all actions jointly."
28. **The immediate switch is too late; the causal bug is upstream cell
    occupation.** Runs757-758 showed Run730 does follow led whenever the led
    color is actually legal, and that most switches occur only after the led
    color has become unplayable despite the token still being available. Run759
    then found a direct earlier zeroing cause for `183 / 183` shielded
    token-available/no-led-action burn events at m40: usually another player
    claimed the exact color/rank cell the future target player needed. The next
    label family should be table-wide exit protection, not a local "follow led"
    classifier or a red-switch chooser.
29. **The first cell-exit label bank exists, but must be trained carefully.**
    Run760 mined `1304` dense action-conditioned rows from 4p Run730 liveness
    behavior. The signal is dense (`28.70%` of considered play decisions), and
    final label-best actions avoid other-player exit damage on `100%` of rows.
    The important guard is anchoring: a naive least-damage target over-selected
    red, so the final bank changes policy targets only when the deployed action
    itself zeroes another player's exit, and prefers safe follow-led actions
    before red.
30. **Naive cell-exit transfer is also not enough.** Run761 showed the policy
    stack can learn the Run760 target, but the useful-looking `38.50%`
    replay other-exit damage dose brought back the known discipline failure:
    follows-led fell to `76.99%` and newly-loses-led-token rose to `16.56%`.
    Run762 showed the safer aux-head route can learn correlations (`0.5456`
    action-value corr, `0.5320` risk corr) while preserving the raw policy, but
    the deployment selectors had no usable margins: threshold/relative modes
    almost never overrode, and additive mode traded tiny changed-row gains for
    new damage on previously-safe rows. Next: change the label/selector target
    shape, likely direct shield-veto contrast labels or larger repeated-seed
    cell-exit contrasts against the anchor action with a held-out precision
    gate.
31. **Offline-passing cell-exit grafts need lane guards, but guards alone do
    not create strength.** Run763's trick-3 cell-exit `improve_graft` passed
    filtered held-out gates, yet unguarded live m10 regressed to `70-72.5%`
    because it often replaced a led-color follow with a red play that the base
    policy considered nearly impossible. The new opt-in improve guard blocks
    non-urgent led-lane burns and filtered `3482` candidates in the m30, but
    the guarded selector tied Run730 (`76 / 120` each). Next: bake
    led-lane/red-urgency constraints into the labels/training data, not only
    into a post-hoc deployment guard.
32. **Baking lane discipline into the labels is necessary but still not
    sufficient.** Run764 did exactly the Run763 follow-up: forced
    base-follows-led/off-led-token-burn candidates negative, allowed only
    urgent red-win exceptions, and passed the offline trick-3 gate at
    `23 / 95` held-out overrides with `100%` precision and safe rate. But live
    m30 tied the Run730 control exactly (`84 / 120 = 70.00%` each), with only
    `28 / 480` trick-3 overrides, and all-play deployment regressed at m10
    (`27 / 40` versus same-seed control `26 / 40`). The lesson is that the safe
    slice is too sparse and local even when broadened; the next cell-exit work
    needs broader trajectory credit assignment or an earlier/more frequent
    deployment mechanism that keeps the same led-lane discipline.
33. **Broader led-continuation data still memorizes instead of generalizing.**
    Run767/768/769 escalated the direct led-lane hypothesis in the right order:
    sidecar-preserving replay mix, larger led-rank continuation corpus,
    legal-set architecture, led-lane label shaping, then broader trick/legal
    count shards with explicit selected-switch exploration. The label signal
    stayed real: Run769 append had `205 / 286 = 71.7%` target changes and
    label-best continuation paradox `0 / 286`; balanced had `188 / 264 =
    71.2%` target changes and label-best paradox `0 / 264`. But the learned
    selector did not transfer across held-out metadata groups. Balanced
    validation collapsed to `14.3%` best precision; append improved only to
    `33.3%` at tiny coverage and `29.4%` at the minimum `8` overrides. The
    next attempt should validate labels across source/seed before training,
    train explicit trajectory survival/liquidity heads, or add deterministic
    public table-wide exit-pressure features. Do not repeat the same legal-set
    objective on a merely larger version of this corpus.
34. **Repeated-rollout labels are stable only in a tiny slice.** Run771 showed
    the stability audit can find genuine policy corrections: `4` stable rows
    all agreed with all-legal recall and had non-worse paradox deltas. But the
    hit rate is too low for broad mining: Run772's balanced scale-up processed
    `12` states with `6` high-margin cases and found `0` stable rows before
    being stopped at `923.319s`. Treat this teacher as a diagnostic and
    cross-check tool, not a bulk pretraining source.
35. **Existing public liveness keys help but do not solve 4p.** Run773 showed
    the deployable public liveness key is real enough to repair
    `heuristic_safe14` (`75.83%` -> `60.00%` on the same m30 seed), but it does
    not meaningfully beat Run730 liveness (`61.67%` -> `60.00%`) and remains
    far above the `<40%` gate. Delaying it to min trick `3` or `5` got worse,
    so the useful corrections are early/mid play-phase, not final-trick
    cleanup.
36. **The remaining one-step public shields are worse.** Run774 tested
    `exit_liquidity_shield_safe14` and `liquidity_feasibility_shield_safe14`
    on the same m30 seed; they landed at `77.50%` and `75.83%` respectively.
    This closes the current deterministic-public shield lane. The next 4p
    attempt should change the data/target: trajectory-level survival labels,
    table-liveness teacher approximation, or targeted mining of upstream
    color/rank cell claims that later remove another player's final led-color
    exit.

## Next experiments - prioritized, concrete (retire each as it lands in EXPERIMENTS.md)

- **E1: DONE in Run726.** Trustworthy 4p m200 ladder established. Best verified
  config is Run689 liveness shield port at `67.50%`, not Run720's noisy m10
  result.
- **E2: DONE in Runs727-728.** Built-in own-hand-safe baselines are worse than
  learned/shielded configs, and initial 4p deals are not intrinsically doomed
  under the exact own-hand liveness oracle.
- **E3: DONE in Run730.** One shield-behavior distillation cycle from Run689
  liveness shield transferred live m200 survival (`67.50%` -> `60.12%`) but
  still failed the `<40%` gate. Keep the artifact as the current best 4p
  teacher/candidate, but do not promote it.
- **E4: PARTIAL.** Web human-match JSONL logging exists. Next: collect real
  human-vs-bot games and add iOS logging/export so pass-and-play sessions feed
  the same Elo pool.
- **E5: DONE in Run731.** 5p m200 ladder established. Best verified config is
  Run600 q-policy at `53.80%`, essentially tied with raw at `53.90%`; built-in
  own-hand-safe baselines are worse.
- **E6: DONE in Run732.** One survived-round q-policy behavior-distillation
  cycle from the best verified 5p teacher failed m200 confirmation
  (`56.30%`), so do not repeat blind survived-behavior cloning from a teacher
  still failing around `54%`.
- **E7: DONE in Run733.** Strict 4p trajectory-causal pairs from the current
  best Run730 shielded distribution are diagnostic and dense enough to train,
  but a guarded 20% causal policy dose failed m30 (`63.33%` vs same-seed
  Run730 `62.50%`). Do not repeat this exact behavior+strict-causal policy
  mixture.
- **E8: DONE in Run734.** Existing exact legal-pressure shield knobs were
  tested as oracle diagnostics. Forced-only was too narrow/late
  (`70.00%` at m10); broader exact-pressure improved same-seed m30 from
  `63.33%` to `52.50%` but still failed and is not information-legal. Use it
  as a label target, not as a deployed selector.
- **E9: DONE in Run735.** A hidden-info all-legal table-liveness teacher passed
  4p m200 (`35.75%`), proving an oracle-quality label source exists. The first
  small guarded policy-stack distillation failed live screens (`60.00%` m10),
  so do not repeat the same `2400`-row clone. Next: scale/balance teacher
  labels, train a direct risk/Q head first, or approximate the table-liveness
  teacher as an information-legal selector.
- **E10: DONE in Run736.** Direct risk/Q-head transfer from Run735 labels is
  learnable offline but not usable via current `q_policy` dials. The best
  q-policy m10 was `65.00%`, and q-policy as the liveness-shield base made the
  screen worse. Do not repeat aux-stack pretraining plus off-the-shelf
  q-policy selection. Next: selector calibration/abstention on live on-policy
  states, or a learned information-legal approximation of the table-liveness
  teacher's action ranking.
- **E11: DONE in Run737.** The obvious calibrated-abstention q-policy variant
  (`relative_dominance`) failed the same-seed m10 sweep: margins
  `0.01/0.03/0.06/0.10` scored `65.00%/70.00%/67.50%/67.50%`, respectively.
  This throttled overrides to `5.8-12.9%` but did not improve survival. Retire
  scalar q-policy knob sweeps for this checkpoint; move to explicit
  teacher-ranking / teacher-approximation transfer.
- **E12: DONE in Run738.** A deployable `improve_graft` teacher-ranker trained
  from Run735's table-liveness all-legal replay learned a precise but too-small
  action slice. Same-seed m10 screens failed: `57.50%` at `threshold=0.99` and
  `52.50%` at `threshold=0.50`, with only `3.6-4.4%` live override coverage.
  Do not repeat small high-margin pairwise rankers on the same replay. Next:
  broader/listwise teacher transfer, larger balanced on-policy table-liveness
  labels, or a deployment objective that explicitly optimizes safe coverage.
- **E13: DONE in Run739.** Broader/listwise teacher transfer from the same
  Run735 `2400`-row replay also failed. The soft policy target was learnable
  offline (`validation CE 1.1501` vs anchor `1.8284`, frozen body exact) and
  changed the replay top action on `51.7%` of rows, but same-seed m10 live
  screens regressed (`60.00%` Run730 public liveness -> `67.50%` Run739 public
  liveness; Run739 raw `72.50%`). Do not repeat weight/temperature sweeps on
  this static replay. Next: larger balanced on-policy table-liveness labels
  from states actually reached by Run730/Run739, especially trick-5/6/7
  forced-exit precursors, with held-out on-policy safe-coverage validation
  before another live screen.
- **E14: DONE in Run740.** Larger on-policy/late-balanced table-liveness labels
  were built and trained (`6400` soft rows; best group validation CE `0.9093`),
  but the held-out safe-coverage gate failed before live evaluation: Run740's
  argmax policy hit deterministic table-liveness top actions less often than
  Run730 on fresh all-play (`37.50%` vs `39.92%`) and late-t4+ (`42.75%` vs
  `46.58%`) rows. Do not repeat soft/listwise policy-only CE optimization as
  the main 4p transfer route. Next: optimize deployed top-action safe coverage
  directly with a margin/top-1 objective, DAgger-style teacher-corrected live
  substitution, or a learned information-legal selector whose offline gate is
  held-out table-liveness top-1/false-positive coverage, not CE.
- **E15: DONE in Run741.** Direct hard-margin top-action transfer fixed the
  Run740 offline gate but failed live survival. Held-out deterministic
  table-liveness top-1 improved to `52.92%` all-play and `57.67%` late, but
  same-seed m10 regressed (`52.50%` Run730 public-liveness control ->
  `62.50%` Run741 public liveness; Run741 raw `67.50%`). The hard clone cut
  follow-led discipline to about `52%` and raised newly-loses-led-token
  decisions to `25-30%`. Do not repeat all-row hard-margin table-liveness
  top-action training. Next: constrained/stratified top-action training with
  explicit held-out gates for follows-led, token-loss, off-led loss, and
  controlled red-switch rates.
- **E16: DONE in Run742.** First constrained top-action transfer
  (`teacher follows led OR red-switch wins trick`) improved held-out top-1
  versus Run730 but still failed the held-out discipline gate before live
  evaluation. Run742 scored `49.08%` all / `55.67%` late top-1, but follows-led
  fell to `64.98%` all / `63.80%` late and newly-loses-led-token rose to
  `17.33%` all / `13.57%` late. Do not repeat row filtering alone. Next:
  action-level constrained targets, gated imitation, or penalties that keep
  Run730-like led-follow discipline unless a high-margin public red-rescue
  exception is available.
- **E17: DONE in Run743.** Action-level constrained relabeling preserved
  Run730-like discipline and improved held-out top-1 slightly, but live m30
  only tied Run730. Run743 public liveness scored `73 / 120 = 60.83%` versus
  same-seed Run730 public liveness `74 / 120 = 61.67%`; raw Run743 failed badly
  at `33 / 40 = 82.50%`. Do not repeat the same all-row anchor-follow dose.
  This fed the positive-margin/oversampled follow-correction test in Run744.
- **E18: DONE in Run744.** Positive-margin follow-led oversampling preserved
  Run730-like discipline but failed offline. `m001_rep3` scored `40.08%` all /
  `47.75%` late held-out top-1; `m000_rep3` scored `41.42%` all / `48.58%`
  late. Since neither clearly beat Run743 (`41.25%` all / `49.75%` late), no
  live m10/m30 was warranted. Do not spend another run on anchor-follow
  oversampling/threshold tweaks at this scale. Next: change the transfer
  mechanism - larger/unfrozen but tightly gated policy updates, DAgger-style
  student-state teacher corrections, or a deployable safe-coverage selector.
- **E19: DONE in Run745.** Liveness-base `improve_graft` is now implemented:
  the trainer can use `metadata_policy_best` as the base action, and deployment
  can run `improve_base_mode=liveness_shield`. The cheap `48`-row probe was
  underpowered: the best regularized model found only `1 / 16` held-out
  positive safe override, while the unregularized model overfit. No live screen
  was warranted. Next: generate a larger grouped liveness-base live-substitution
  corpus and require a minimum held-out override count before live screening.
- **E20: DONE in Run746.** The larger grouped liveness-base graft test failed.
  D320 and D640 corpora were dense (`243 / 320` and `496 / 640` target changes),
  but no summary/raw-summary `ImproveMLP` candidate passed the grouped
  live-screen gate. Do not keep scaling this exact selector setup. This fed the
  D320/D640 false-positive/missed-positive audit in Run747.
- **E21: DONE in Run747.** D320/D640 graft false-positive/missed-positive audit
  completed. The missing piece is not the obvious lane/exit feature set: those
  features are already available in the selector snapshots. The current pairwise
  threshold objective is not calibrated against hard negatives. This fed the
  same-state contrastive/hard-negative selector tests in Run748.
- **E22: DONE in Run748.** Same-state CE and pairwise-margin selector objectives
  were implemented and tested on D320/D640. None passed the grouped gate. The
  small independent `ImproveMLP` liveness-base graft family should be retired
  unless the architecture or data source changes.
- **E23: DONE in Run749.** A true legal-set selector with base-as-abstain was
  implemented and tested on D320/D640. It also failed grouped safe coverage on
  the same corpus. Retire selector architecture/weight sweeps on this bank; next
  change data source with DAgger/shadow hard negatives or move to policy-stack
  transfer with discipline gates.
- **E24: DONE in Run750.** First failed-selector shadow hard-negative corpus
  built against the Run730-style liveness base. The data source works (`11 / 32`
  positives, `21 / 32` hard negatives), but the first `32`-row BCE selector did
  not generalize (`0 / 8` grouped validation overrides). Next: scale to
  `128-256` rows, stratify beyond trick `0`, and record top-k same-state
  failed-selector candidates so pairwise/legal-set objectives have true
  same-state hard negatives.
- **E25: DONE in Run751.** Top-k same-state late shadow mining is implemented.
  The miner audits top-3 failed-selector candidates plus the liveness base,
  stores candidate evaluations, writes q/risk labels for all audited actions,
  and reports proposal supply by trick. The `8`-state late smoke proved the
  shape (`q_label_count=4`, `risk_label_count=4`, `6` pairwise pairs), but the
  validation selector abstained (`0 / 2` overrides), and collection overfocused
  on trick 4. Next: scale to `128-256+` rows with explicit trick-5/6/7 quotas
  before any live screen.
- **E26: DONE in Run752.** Explicit trick quotas and per-trick trainer
  diagnostics are implemented. The `128`-row quota bank hit exactly
  `32/32/32/32` rows for tricks 4/5/6/7, but all selector variants failed the
  offline live-screen gate. Best broad variant: BCE+stateCE, `15 / 32`
  validation overrides, `66.7%` precision, `93.3%` safe. Best strict variant:
  `1` perfect validation override, too sparse. Next: do not live-screen Run752;
  use the per-trick diagnostics to build a trick-aware or richer selector and/or
  expand to `256+` rows.
- **E27: DONE in Run753.** Trick-gated deployment/eval support is implemented.
  `--live-screen-trick-numbers` now emits official trick-filtered threshold
  diagnostics, and `improve_trick_numbers` can restrict an improve-graft bot to
  exact trick numbers. The first trick-5-only check failed the live-screen gate:
  best filtered validation was `9 / 15` overrides, `77.8%` precision, `100%`
  safe. Next: do not live-screen Run753; collect more trick-5 quota data and/or
  add richer public features before trying the gate again.
- **E28: DONE in Run754.** Scaled the trick-5-only shadow corpus to `128` rows
  and reran the filtered gate with BCE+stateCE, strict BCE, and BCE+pairwise
  variants. All failed. Best held-out result was BCE+pairwise at `9 / 32`
  overrides, `55.6%` precision, `100%` safe. Next: stop scale-only
  summary-feature trick-5 reruns; add richer proposed-vs-base public features
  or a selector architecture that explicitly compares candidate/base action
  consequences.
- **E29: DONE in Run755.** Tried the existing legal-set/DeepSets selector on
  the scaled Run754 trick-5 bank. It still failed the gate: `10 / 32` held-out
  overrides, `60.0%` precision, `100%` safe. Next: improve labels or derive
  deterministic led-token/red-rescue training targets before more selector
  sweeps.
- **E30: DONE in Run756.** Added and screened
  `liveness_shield_keep_responsible_red_win`, plus counters for responsible
  red-win base/candidate opportunities. The knob made no live difference on a
  4p m30 same-seed screen: both control and variant were `80 / 120 = 66.67%`.
  The counters explain why: `43` base responsible-red-win opportunities, `45`
  candidate responsible-red-win selections, but `0` candidate attempts to move
  away from a responsible base red win. Next: do not pursue this guard as a
  survival lever; use richer/repeated continuation labels or a direct training
  target for when to follow led versus spend the lane to red.
- **E31: DONE in Run757.** Made `quantum_cat_led_switch_audit.py` support full
  neural selector payloads, including the deployed-style liveness shield, and
  added led-available/led-unavailable rates. On a 4p m60 audit of Run730,
  raw policy and liveness shield both followed led `100%` when led color was
  legal, and both had `0` switches with led color legal. The shield improved
  red choice when red was legal on switch (`96.91%` vs raw `83.23%`), but the
  paradox context stayed high (`62.08%` shielded). Next: mine/label earlier
  actions that make later led-color legal exits vanish; do not train an
  immediate "follow led if available" target as though that were the observed
  4p bug.
- **E32: DONE in Run758.** Added token-status counters to the led-switch audit.
  On a 4p m40 Run730 shielded audit, `204 / 248 = 82.26%` switches newly lost
  a still-available led token; red was legal on `176 / 204 = 86.27%` of those
  new-token-loss switches; the shield chose red in `167 / 176 = 94.89%`; but
  only `67 / 204 = 32.84%` were responsible red-win switches, and the win rate
  was `58.82%` counting unresolved as loss. Next: mine future
  token-available/no-led-action bottlenecks and label earlier rank/cell/exit
  decisions that avoid creating them.
- **E33: DONE in Run759.** Added
  `quantum_cat_future_led_bottleneck_audit.py` and traced future
  token-available/no-led-action token burns to direct earlier zeroing causes.
  On a 4p m40 Run730 shielded audit, `183 / 183 = 100%` of such events had a
  direct prior action that changed the target player's potential count in that
  led color from `>0` to `0`; `160 / 183` were caused by another player,
  `180 / 183` by an action in the target color, and `183 / 183` while the
  target player had that rank in hand. Next: train/evaluate table-wide
  color/rank cell-protection labels that avoid removing another player's last
  future exit in a still-tokened color.
- **E34: DONE in Run760.** Added
  `quantum_cat_cell_exit_protection_miner.py` and mined the first anchored
  cell-exit protection replay:
  `az_runs/run760_4p_cell_exit_protection_labels_20260613/run730_liveness_cell_exit_anchored_m40_seed20260616.npz`.
  It contains `1304` rows from `160` hand-rounds, `28.70%` collection rate,
  `49.62%` policy-target changes, and `100%` label-best actions with no
  other-player exit damage. Next: train/evaluate a guarded aux-head or
  policy-stack dose from this bank, and keep the anchoring guard to avoid the
  old red-overselection failure.
- **E35: DONE in Run761.** Trained the guarded policy-action-stack dose from
  Run760. Conservative step20 failed to beat the anchor offline; stronger
  step40/step60 doses reduced replay other-exit damage but failed the
  discipline gate (`16.56%` newly-loses-led-token at step60/final). Reject; no
  live screen.
- **E36: DONE in Run762.** Trained only the separate action-value/risk aux
  stacks from Run760 and swept deployment-shaped q-policy selectors. Aux
  correlations improved by step80, but selector margins were not usable:
  threshold/relative barely fired and additive selectors only reduced replay
  damage by `0.16pp` while increasing unchanged-row damage. Reject; no live
  screen.
- **E37: DONE in Run785.** Tested the 5p version of upstream cell-exit
  protection after dense warning targets failed. The miner produced useful
  data (`160` focused damage rows, `159 / 160` target changes, `158 / 160`
  no-other-exit-damage label-best actions), but the conservative aux-head
  route failed live (`86 / 150 = 57.33%` m30) and the legal-set/base-abstain
  diagnostic failed grouped held-out precision (`15 / 79` overrides at
  `60.0%`, `0` eligible thresholds). Reject; no m200, no promotion, no runtime
  legal-set support. Next: improve label validity with same-state hard
  negatives or repeated-seed/exact continuation labels around cell-exit
  candidates before another training/live-screen attempt.
- **E38: DONE in Run786.** Tested the narrow abstention-calibration hypothesis
  from the Run785 false-positive audit: increase legal-set base/abstain target
  weight to `4.0` and reduce safe-head pressure. It failed: best grouped
  validation was `8 / 66` overrides at `50.0%` precision and `87.5%` safe,
  with `0` live-screen-eligible thresholds. Reject; do not keep sweeping
  base weights or safe-loss weights on this same cell-exit corpus.
- **E39: DONE in Run787.** Added
  `quantum_cat_cell_exit_continuation_audit.py` and audited the exact
  candidate-vs-base continuations for Run785/786. The best Run785 legal-set
  selected overrides were mostly invalid (`13 / 15` rejected; `0 / 15`
  reduced any-player paradox rate). A 40-row slice of one-step cell-exit target
  changes was also mostly invalid (`30 / 40` rejected; only `5 / 40` reduced
  any-player paradox rate). Reject unfiltered one-step 5p cell-exit labels as
  a training substrate. Next: build continuation-filtered replay and hard
  negatives from this audit path.
- **E40: DONE in Run788.** Added
  `quantum_cat_continuation_filtered_replay.py` and exported the first
  continuation-filtered 5p cell-exit replay from the Run787 first-40 audit.
  Export worked (`40` rows, `10` continuation-supported overrides, `30`
  rejected one-step positives as hard negatives), but the tiny legal-set
  diagnostics failed before live screening. Strict led-lane shaping reduced the
  replay to only `3` deployable positives and produced `0` held-out positives;
  the no-lane-force control had `3` held-out positives but still selected
  `3 / 12` false/riskier overrides (`0.0%` precision/safe). Reject as a model
  update; next: scale and balance continuation-filtered replay slices before
  another trainer screen.
- **E41: DONE in Run789.** Scaled the continuation-filtered audit to all
  `172` Run785 one-step target changes. Aggregate `common1` continuation:
  `46 / 172 = 26.74%` supported by `risk_or_score`, `126 / 172` rejected;
  only `26 / 172 = 15.12%` reduced any-player paradox risk. Exported both
  `risk_or_score` and `risk_only` all-172 replays and ran strict/no-lane
  legal-set diagnostics. All failed grouped gates. Best strict
  `risk_or_score` found only `1 / 50` held-out override despite perfect
  precision/safe; no-lane `risk_or_score` selected `11 / 51` overrides at only
  `18.18%` precision/safe. `risk_only` also failed (`2` held-out overrides at
  `50%` precision/safe, with or without lane force). Reject; next:
  repeated-rollout validation for the `26` any-risk-safer candidates and more
  led-lane-compatible positives.
- **E42: DONE in Run790.** Repeated-rollout validated the Run789
  any-risk-safer subset. All `26 / 26` candidates stayed
  continuation-supported at common8, with `26 / 26` selected
  any-player-safer and `0` rejected. Strict led-lane shaping kept only
  `7 / 26` as deployable positives. Led-compatible proposal mining found
  `67` best-per-state candidates, but only `8 / 67` were any-player-safer at
  common1; common8 validation kept all `8 / 8`, adding two new
  strict-compatible positives (`9` total). The high-confidence mixed replay
  (`193` rows: `28` repeated positives, `165` hard negatives) still failed the
  strict legal-set grouped gate: only `9` strict deployable positives overall,
  `5` held-out positives, and the best threshold selected `3 / 61` overrides
  at `33.33%` precision/safe. Reject as a model update; next: scale
  led-compatible/repeated-positive mining before training.
- **E43: DONE in Run791.** Scaled led-compatible mining on a fresh 5p
  policy-damage shard. The source shard collected `480` rows from `327` rounds
  with `474 / 480` policy-target changes. The reusable
  `quantum_cat_led_compatible_selector_builder.py` found `426` led-compatible
  one-step-positive candidate actions across `198` states. Common1 continuation
  kept `74 / 198 = 37.37%` supported overrides, but only `32 / 198 = 16.16%`
  reduced any-player paradox risk; `124 / 198` were rejected and `15 / 198`
  were riskier. Common8 repeated validation of the `32` any-risk-safer rows was
  perfect: `32 / 32` supported, `32 / 32` any-player-safer, `0` rejected. A
  high-confidence replay with those `32` positives plus `124` hard negatives
  still failed the strict legal-set grouped gate: validation had `14` positives
  in `55` states, but best threshold selected only `7 / 55` overrides at
  `42.86%` precision and `71.43%` safe. Reject as a model update; next selector
  work needs better target/features/loss or more merged positives, not a live
  screen.
- **E44: DONE in Run796.** Tested stable q-teacher labeling from balanced 5p
  tactical states. Collection worked (`24 / 24` states; balanced
  `near_paradox`, `mid_rank_follows_led`, `red_win_switch`, and `token_loss`;
  average legal count `13.0`), but full-match forced-action scoring was too
  slow for this loop. The candidate-consensus run processed only `4 / 16`
  states in `338.057s` and exported `0` stable rows before interruption.
  Reject as an interactive data source unless the scorer is batched/cached.
- **E45: DONE in Run797.** Built deterministic exact table-liveness replay as
  a cheaper 5p label source. The scaled shard has `640` rows from `50` rounds
  and confirms the important split: the dense action-value best follows led in
  `635 / 640` rows and burns the led token in only `5 / 640`, while the direct
  deterministic policy target burns the led token in `107 / 640` rows. Keep
  this replay for value/risk/legal-set learning; do not directly imitate the
  table-liveness policy target.
- **E46: DONE in Run798.** Trained the positive-head legal-set selector on the
  Run797 smoke replay. Held-out behavior was clean (`3 / 51` overrides,
  `100%` precision, `100%` safe, mean value delta `+0.065163`, mean risk delta
  `-0.029839`) but below the `8`-override coverage floor. Reject as deployable;
  scale before any live screen.
- **E47: DONE in Run799.** Trained the same selector on the scaled Run797
  replay. Scaling gave enough positives (`74` positive states in `640`), but
  held-out calibration failed: best threshold selected only `6 / 222` overrides
  at `50%` precision and `50%` safe. Reject; no runtime hook and no live screen.
  Next attempt should use cross-shard validation and stronger false-positive
  calibration around table-liveness/token-loss conflicts, not another threshold
  sweep.
- **E48: DONE in Run800.** Added source-holdout validation, embedded replay
  row-group/source support, and padded batched training to
  `quantum_cat_legal_set_selector_trainer.py`. Generated an independent
  deterministic table-liveness holdout shard (`1642` rows from `120` rounds;
  source paradox rate `68 / 120 = 56.67%`). The label source replicated:
  value-best followed led in `1636 / 1642` rows and burned the led token in
  only `6 / 1642`, while the direct policy target burned led in `360 / 1642`.
  But true source-holdout selector transfer failed both directions. Train
  Run797 -> validate Run800: `55` overrides, `29.09%` precision, `49.09%`
  safe. Train Run800 -> validate Run797: `12` overrides, `41.67%` precision,
  `58.33%` safe. Reject; no live screen. Next: do not run another
  threshold/weight sweep of this selector. Use the source-holdout gate, but
  change the target decomposition or mine explicit cross-shard false-positive
  hard negatives first.
- **E49: DONE in Run801.** Tested the target-decomposition hypothesis with
  `quantum_cat_delta_calibration_trainer.py`: train per-action value/risk delta
  heads plus positive/safe heads, then select with a deterministic abstention
  rule rather than a legal-set softmax. The smooth labels transfer across
  source shards (`value_corr=0.985-0.987`, `risk_corr=0.988-0.991` on held-out
  actions), but the high-precision override boundary still fails. Train Run797
  -> validate Run800: best `25` overrides at `24.00%` precision and `28.00%`
  safe. Train Run800 -> validate Run797: best `11` overrides at `45.45%`
  precision and `45.45%` safe. Reject; no live screen. Next: mine the
  source-holdout selected false positives as explicit hard negatives or train a
  pairwise within-state lookalike discriminator; do not keep doing smooth
  value/risk regression or threshold sweeps.
- **E50: DONE in Run802.** Tested the pairwise within-state lookalike
  discriminator directly with `quantum_cat_pairwise_boundary_trainer.py`.
  The model scored each candidate relative to the base action using
  `[candidate_features, base_features, candidate_minus_base]`, trained true
  positive overrides above the base action and synthetic led-lane hard
  negatives, and kept the same source-holdout gate. It failed. Train Run797 ->
  validate Run800: the wide strict-margin rerun found only `6 / 1642` overrides
  at `33.33%` precision and `66.67%` safe. Train Run800 -> validate Run797:
  best was `10 / 640` overrides at `60.00%` precision and `80.00%` safe, still
  below the `85%` precision / `90%` safe live-screen gate. Reject; no live
  screen. This was superseded by E51, which mined the actual source-holdout
  selected false positives and validated on a third independent shard.
- **E51: DONE in Run803.** Tested the actual selected-false-positive idea with
  `quantum_cat_false_positive_boundary_trainer.py`. Generated a third
  deterministic table-liveness shard (`1657` rows, source paradox `67 / 120 =
  55.83%`, value-best follows led `1647 / 1657`). Direction A trained Run797,
  mined Run800, validated Run803: best after approval was `10 / 1657` overrides
  at `40.00%` precision / `40.00%` safe. Direction B trained Run800, mined
  Run797, validated Run803: best after approval was `57 / 1657` at `43.86%`
  precision / `49.12%` safe. The focused selected-only variant overfit the mine
  split but still got only `60 / 1657` at `43.33%` / `46.67%`. The strongest
  cross-fit variant mined both directions (`294` selected false positives),
  trained final stage-1 on Run797+Run800, and validated on Run803: stage-1
  improved to `33 / 1657` at `51.52%` / `57.58%`; after approval the best was
  `25 / 1657` at `56.00%` / `64.00%`. Reject; no live screen. Next: stop
  weight/threshold tuning this local approval boundary. Move to richer
  source-invariant state/action-set representation or leave-one-source-out
  multi-shard training with explicit source validation.
- **E52: DONE in Run804.** Tested multi-source and richer/source-invariant delta
  training on Run797+Run800 with Run803 held out. Plain summary DeltaCalibrator
  got `48 / 1657` overrides at `31.25%` precision / `31.25%` safe; seed-matched
  summary got `48 / 1657` at `43.75%` / `47.92%`; raw+summary observation got
  `39 / 1657` at `41.03%` / `43.59%`. Added
  `quantum_cat_source_invariant_delta_trainer.py`; weak adversary `0.1` was the
  best Run804 slice at `26 / 1657` overrides, `46.15%` precision / `46.15%`
  safe, while stronger adversary `1.0` regressed to `28.57%` / `28.57%`.
  Reject; no live screen. Next: do not keep appending raw features or tuning
  source-adversary weights. Build an action-set/listwise or sequence-context
  model that compares legal actions inside the state with richer structured
  public board/hand/trick context.
- **E53: DONE in Run805.** Added `quantum_cat_action_set_transformer_trainer.py`,
  a padded legal-action-set Transformer encoder with set-aware selection,
  value/risk, positive, and safe heads. Full model on Run797+Run800 validated
  on Run803: train top slice `86` overrides at `100%` precision / `100%` safe,
  but validation best only `52 / 1657` at `32.69%` precision / `40.38%` safe.
  Regularized/early-stop variant: validation best by precision was a tiny clean
  `3 / 1657` at `100%` / `100%`, but it failed the `8`-override floor; best
  coverage-qualified slice was `9 / 1657` at `55.56%` / `66.67%`. Reject; no
  live screen. Next: do not scale this exact transformer up. Either build a
  multi-shard early-stopping/calibration setup to test whether the tiny clean
  slice can be made reliable, or add explicit sequence/trick-history context so
  the model can distinguish future-lane-collapse situations across sources.
- **E54: DONE in Run806.** Tested that tiny-clean-slice hypothesis directly
  with `quantum_cat_action_set_calibrated_gate.py`. Generated a fourth
  deterministic 5p table-liveness shard (`1660` states, source paradox
  `82 / 120 = 68.33%`, value-best follows led `1645 / 1660`). Calibrating the
  regularized Run805 Transformer on Run803 still found only `3 / 1657` clean
  overrides, so no threshold met the `8`-override floor. Applying the best
  calibration threshold to Run806 produced only `2 / 1660` overrides at
  `0.00%` precision and `50.00%` safe; even the oracle-best Run806 threshold
  found just `1 / 1660` clean override. Reject; no live screen. Next: stop
  threshold-sweeping the existing current-state transformer. Add sequence/trick
  history or a true multi-shard leave-one-source-out selection loop.
- **E55: DONE in Run807.** Added `--round-context-mode=summary` to
  `quantum_cat_action_set_transformer_trainer.py` and taught the calibrated
  gate to rebuild that feature schema. The context adds `29` earlier-in-round
  features/interactions, including prior follow-led/token-loss rates, own prior
  token-loss rates, previous base token loss, acting-player/position scalars,
  and candidate token-loss interactions. Main run trained on Run797+Run800+
  Run803 and held out Run806: best validation threshold was `50 / 1660`
  overrides at `30.00%` precision / `30.00%` safe, with no eligible threshold.
  A stricter threshold diagnostic still found only `22 / 1660` at `31.82%` /
  `31.82%`. Selection audit: selected actions followed led `100%` of the time
  and lost tokens `0%`; false positives were intra-lane rank/allocation errors
  with true mean value delta `-0.007606` and true mean risk delta `+0.009820`
  despite predicted `+0.129710` value and `-0.101484` risk. Reject; no live
  screen. Next: target follow-led/no-token-loss rank allocation directly with
  source-balanced hard negatives, stronger exact continuation labels, or a
  richer board/trick-prefix representation.
- **E56: DONE in Run808.** Added
  `quantum_cat_intralane_rank_filter_trainer.py`, a focused abstaining MLP over
  only follow-led/no-token-loss candidate overrides. Train bank:
  `7651` examples (`1069` positives, `5087` hard negatives). Run806 validation
  bank: `3270` examples (`445` positives, `2182` hard negatives). The model
  fit train strongly (`194` train overrides at `100%` precision / `100%` safe),
  but transfer collapsed: validation value/risk corr `0.073164` / `0.055193`;
  best validation threshold had only `1` clean override, and best
  coverage-qualified slice was `9` overrides at `33.33%` / `33.33%`. Reject;
  no live screen. Next: do not increase MLP capacity or reweight this same bank.
  Build explicit same-state follow-led rank/cell scarcity features or
  repeated-seed exact continuation labels for intra-lane candidates.
- **E57: DONE in Run809.** Extended
  `quantum_cat_intralane_rank_filter_trainer.py` with
  `--intralane-context-mode=rank_scarcity`, appending `62` same-state
  candidate-filtered rank/exit-scarcity percentile/z-score features. Main
  Run809 kept the Run808 split and recipe, adding only this context. It moved
  the best clean held-out slice from `1` to `3` Run806 overrides at `100%`
  precision / `100%` safe, with mean value delta `+0.202592` and mean risk
  delta `-0.074807`, but still failed coverage. Best coverage-qualified slice:
  `70` overrides at `31.43%` precision / `34.29%` safe; validation value/risk
  corr stayed weak (`0.103677` / `0.102197`). Reject; no live screen. Next:
  generate stronger repeated-seed/exact continuation or pairwise oracle labels
  for the same follow-led/no-token-loss candidate family.

## Core competitive metric and personal safety guardrail

```text
competitive_placement_score =
  mean((opponents_beaten + 0.5 * opponents_tied) / opponents_present)

candidate_personal_paradox_rate =
  candidate_paradoxes / candidate_hand_round_opportunities
```

Evaluation: official full matches against the frozen player-count-specific
incumbent, common chance deals, and every candidate seat exactly balanced.
The candidate's personal paradox rate is compared directly with incumbent
opponents. Homogeneous all-player table survival remains useful for diagnosing
cooperative board behavior, but is not competitive evidence or a veto.

Preserve led-color flexibility unless a switch improves the candidate's own
official expected placement. Protecting an opponent from a paradox is
undesirable when that opponent's paradox would improve the candidate's rank.

## Targets, in priority order

1. **4p competitive gate**: beat the exact Run730 public-liveness incumbent on
   seat-balanced independent deals without personal-paradox or official-score
   regression. Historical homogeneous-survival findings are diagnostic.
   Current working
   diagnosis (Run726-728): the 4p bottleneck is a table-wide endgame
   allocation problem created during play, not a local label shortage or
   unavoidable initial deal doom. Next levers: one Run604-style behavior
   distillation cycle from the best verified shielded teacher already moved
   the ceiling to `60.12%`; Run733 proved current-distribution causal pairs are
   learnable and can be trained without over-switching, but the guarded policy
   dose still failed; Run734 showed exact-pressure contains useful table-wide
   signal but a one-step oracle shield is still not enough. Run735 built the
   missing oracle-quality 4p teacher (`35.75%` m200), but the first
   `2400`-row policy-stack clone did not transfer. Next focus on transfer:
   Run736 shows direct risk/Q-head pretraining learns the labels but current
   q-policy selection still fails live, Run737 shows simple abstention does not
   fix it, and Run738 shows a small high-precision pairwise teacher-ranker is
   too sparse to move the game. Run739 shows broad soft/listwise targets from
   the same small static replay can move many replay decisions but still harm
   live survival. Run740 shows that larger/balanced on-policy soft/listwise
   labels still do not help if the deployed argmax action gets worse on
   held-out table-liveness top-1. Run741 shows that optimizing top-action
   table-liveness directly is still harmful if it broadly clones hidden-info
   red/off-led token-loss decisions. Run742 shows that row-filtering to
   follow-led or red-winning teacher actions is still too loose. Run743/744
   show that action-level anchoring can preserve discipline but still be too
   weak to move deployed argmax behavior. Run745 showed a better deployment
   mechanism - learned safe-coverage overrides after the liveness shield - but
   the first `48`-row probe was too underpowered. Run746 removed that excuse:
   grouped D320/D640 selectors still failed held-out safe coverage. Run747
   then showed the obvious public lane/exit features are already present, and
   Run748 showed same-state CE / pairwise-margin losses still do not make this
   small independent MLP selector pass grouped safe coverage. Run749 showed the
   first true legal-set/base-abstain architecture also fails on the same corpus.
   Run750-752 show the right next data shape: failed-selector shadow proposals
   against the liveness base, with top-k same-state candidates audited and
   explicit late-trick quotas. The next useful transfer work must improve
   selector calibration on that source, while preserving led-follow discipline
   first and adding only narrow high-confidence public rescue exceptions.
2. **5p competitive gate**: beat the exact Run600 q-policy incumbent on
   seat-balanced independent deals with the same personal safety guardrail;
   Run731 confirms the historical deployable 5p survival baseline is
   still `53.80%`, Run732 showed that one Run604/Run730-style survived
   behavior-distillation cycle from the best verified 5p teacher worsens to
   `56.30%` at m200, and Run775 shows a hidden-info exact late-table teacher
   can pass strongly at `20.20%` while a small public behavior clone still
   fails at m30. Run813 adds a more directly targeted exact late-exit teacher:
   tricks `5..6`, no policy gap, exact legal pressure, `32.10%` at m200 with
   CI upper `34.99%`; its learned late-exit filter failed live, so the useful
   output is the teacher/contrast source, not the selector. Next 5p levers
   should use exact teachers for deterministic policy targets, veto contrasts,
   or trajectory-level causal continuation labels that teach late-exit
   allocation without hidden inference features. Run814 proved a feasible
   capped replay shape (`top_policy_actions=4`, `max_actions_per_state=8`,
   `120` deterministic-teacher rows from `5` matches); scale that before any
   all-legal exact replay. Run776-778 show that broad exact-table aux labels,
   red-switch-only labels, and small late-exit aux labels are not sufficient.
   Run779 shows sparse low-exit causal warning labels can improve the m30 best
   to `46.67%`, but still miss the gate; Run780 shows naive dense all-legal
   soft labels regress to `51.33-54.67%` and are poorly fit offline, and Run781
   shows compacting those labels does not repair them. Run782 then confirmed
   hard selected-vs-best contrasts are learnable offline but too narrow live
   (`56.67-60.67%` m30), and Run783 confirmed blunt binary all-legal relabeling
   also fails live (`55.33-63.33%` m30). The next attempt should not be another
   aux-head q-policy replay from the same dense warning outcomes. Run784 then
   showed that direct policy-veto consumption and unlimited q-policy overrides
   are worse, both hitting `100%` m30 paradox rounds. So the next attempt must
   improve label validity and abstention, not just consume the same labels more
   aggressively. Run785 then moved to the better upstream cell-exit source, but
   aux-head q-policy still regressed to `57.33%` and a legal-set/base-abstain
   grouped gate failed at `60.0%` precision. The next 5p attempt must add
   stronger hard-negative/repeated-continuation calibration around cell-exit
   candidates before any live screen. Run786 confirms that simply increasing
   legal-set abstain/base weight is not enough (`50.0%` precision, `87.5%`
   safe); retire that knob on the current corpus.
   Run787 proves the underlying labels need a continuation filter: in the
   selected-overrides audit, no selected action reduced paradox risk; in the
   first 40 one-step target changes, only `12.5%` reduced any-player paradox
   risk. The next 5p experiment should build a continuation-filtered
   cell-exit replay, not another unfiltered selector/training pass.
   Do not rerun blind survived-behavior cloning from the same `5333` survived
   rows, and do not repeat another small Run776-step20 late-exit aux replay
   without a new causal or veto label source.
3. **3p competitive raw policy**: improve official match placement and personal
   safety against the frozen promoted incumbent; no table-wide threshold is a
   prerequisite for 4p or 5p work. Behavior-cloning the shield transfers strength, not
   survival (Run604/605); veto-contrast labels FAILED in the 4p campaign
   (Runs 607-609, 715, 722) and are unproven in 3p - treat as low-prior.
   Table-wide liveness aux targets from `quantum_cat_liveness_oracle.py`
   remain the live option.
4. **Strength**: keep growing independent-deal official placement advantage at
   every promotion; never trade away wins to protect opponents from paradox.
5. **Human Elo**: use the web logger, add iOS export, and collect
   human-vs-bot matches to disk. Define Elo anchored on a frozen reference pool
   (e.g. `heuristic` = 1000, `heuristic_safe14`, `oracle_safe`, and the frozen
   2026-06-10 champion as fixed anchors). A claim of "matches/exceeds human
   performance" requires >= 100 recorded human matches and the bot's Elo at or
   above the human cohort's, with CIs. Until human games exist, report progress
   against the anchor pool only.

## Hard methodology rules (paid for in runs 596-752; do not relitigate without changing the binding constraint - compute scale or label quality)

- Competitive decisions require independent common-deal blocks with all
  candidate seats represented; small live screens are exploratory only.
- Formal promotion requires replicated 600-match seat-rotated competitive
  gates, placement improvement, official-score nonregression, and personal
  paradox nonregression, with candidate/incumbent checkpoint SHA-256,
  exact selector modes, backup, and an EXPERIMENTS.md entry.
- Keep safe bots in every eval and league pool (`heuristic_safe14/15`,
  `oracle_safe`, shielded champion). Score-only promotion inside a
  paradox-rich pool is exactly how the project once produced a 97.6%-paradox
  "champion".
- Falsified at current scale:
  - q-policy / value-shaped continuation self-play (Runs 596-602);
  - low-sim (16) belief-MCTS self-play (Run 603);
  - one-step offline distillation of late bottleneck rows (Runs 487-552);
  - selfish own-hand-margin objectives at inference (4 confirmations);
  - built-in own-hand-safe 4p/5p policies as survival teachers/floors
    (`heuristic_safe15`, `oracle_safe`, Runs727/731);
  - self-distillation beyond one cycle (Run 605);
  - blind 5p survived-round behavior distillation from a still-failing q-policy
    teacher (Run 732);
  - small 5p public policy-stack behavior cloning from the hidden-info exact
    late-table teacher (Run 775): the teacher passed m200 at `20.20%`, but the
    student failed m30 (`48.67-60.67%`), so use contrast/risk/trajectory labels
    next rather than pure survived action imitation;
  - 4p shield-veto contrast labels and small veto doses (607-609, 715,
    722-723);
  - 4p single-label Q selectors - table-liveness alone, led-preservation
    alone, bounded/unbounded veto-pair variants (711-719, 716-717);
  - 4p late forced-exit oversampling (721), deterministic red-exit aux
    curriculum (724), hard shields incl. oracle_safe at 4p (709), cheap
    online survival-teacher rollouts (710);
  - 4p guarded behavior+strict-causal policy dose from the current Run730
    shielded distribution (733): it preserved led discipline but failed m30;
  - 4p immediate exact forced-only shield (734): too late/narrow and worse at
    m10.
  - 4p small guarded policy-stack clone from the Run735 table-liveness teacher:
    offline labels learned, live m10 still failed (`60.00%`) and shielded mode
    was worse than same-seed Run730 (`55.00%`).
  - 4p direct action-value/action-risk aux-stack transfer from Run735 labels
    plus current `q_policy` selectors: excellent replay validation, failed live
    m10 (`65.00%` best q-policy; shield-stacked q-policy worse).
  - 4p small high-margin `improve_graft` pairwise teacher-ranker from Run735
    labels (Run738): good offline precision, but only `3.6-4.4%` live override
    coverage and no same-seed m10 improvement over public liveness.
  - 4p soft/listwise policy replay from the same Run735 `2400` labels
    (Run739): offline CE improved and top actions changed broadly, but live
    same-seed m10 regressed (`60.00%` control -> `67.50%` shielded Run739).
  - 4p larger on-policy/late-balanced soft/listwise policy replay (Run740):
    offline soft CE improved, but held-out deterministic table-liveness
    top-action safe coverage was worse than Run730, so no live screen was
    warranted.
  - 4p all-row hard-margin table-liveness top-action transfer (Run741):
    held-out top-1 improved sharply, but live same-seed m10 survival regressed
    because predicted actions overselected red/off-led token-loss moves.
  - 4p row-filtered constrained top-action transfer (Run742):
    `follow-led OR red-win-switch` improved held-out top-1 versus Run730 but
    still failed the held-out discipline gate before live evaluation.
  - 4p action-level anchor-follow transfer (Run743): discipline preserved, but
    live m30 only tied Run730 and raw survival regressed.
  - 4p positive-margin/oversampled follow-led corrections (Run744):
    discipline preserved, but held-out top-action coverage did not clearly beat
    Run743, so no live screen was warranted.
  - 4p tiny liveness-base `improve_graft` probes (Run745): mechanism works, but
    `48` rows / `16` validation states are too small to justify live screening
    when the best model finds only one held-out override.
  - 4p scaled liveness-base `improve_graft` with the current `ImproveMLP`
    feature/objective setup (Run746): D320/D640 grouped validation still failed
    safe-coverage gates; broad thresholds were false-positive-heavy and strict
    thresholds abstained.
  - 4p tiny/small shadow-selector diagnostics (Runs750-752): the data source,
    top-k row shape, and trick quotas are useful, but none are live-screen
    evidence. Run750/751 abstained; Run752 found coverage but failed precision
    (`66.7%` best broad validation precision, strict variants only `1-2`
    overrides).
  - 4p trick-gated improve-graft selector (Run753): trick-specific deployment
    is implemented, but the trick-5 held-out gate still failed (`9 / 15`,
    `77.8%` precision, `100%` safe), so no live screen.
  - 4p scaled trick-5-only summary-feature selectors (Run754): collection was
    healthy, but all filtered gates failed; best was `9 / 32`, `55.6%`
    precision, `100%` safe.
  - 4p legal-set selector on scaled trick-5 shadow data (Run755): best
    threshold was `10 / 32`, `60.0%` precision, `100%` safe; no live screen.
  - 5p cell-exit transfer from Run785: the label source is feasible, but
    frozen aux-head q-policy failed live (`57.33%` m30) and the legal-set
    abstainer failed grouped validation (`15 / 79` overrides at `60.0%`
    precision, `0` eligible thresholds). Do not repeat this corpus with only
    q-policy knobs, simple aux-head training, or the current legal-set
    objective.
  - 5p legal-set abstention calibration from Run786: stronger base weighting
    failed (`8 / 66` overrides, `50.0%` precision, `87.5%` safe). Do not keep
    sweeping base/positive/safe-loss weights on the same Run785 cell-exit mix.
  - unfiltered 5p one-step cell-exit labels after Run787: selected overrides
    had `0 / 15` survival-improving actions, and the first 40 policy-target
    changes had only `5 / 40` survival-improving actions. Do not train from
    unfiltered one-step cell-exit targets.
  - tiny first-slice continuation-filtered 5p cell-exit replay after Run788:
    the exporter works, but `40` rows with only `3` strict deployable positives
    is not enough; both strict and no-lane-force legal-set diagnostics failed
    offline. Do not live-screen or promote this tiny selector.
  - all-172 `common1` continuation-filtered 5p cell-exit replay after Run789:
    larger coverage confirms the label problem. Strict gates have too few
    deployable positives, and no-lane controls select false/riskier overrides.
    Do not live-screen or promote these selectors; do not keep sweeping
    thresholds/weights on the same common1 corpus.
  - high-confidence common8-positive/common1-negative 5p strict selector after
    Run790: repeated positives are real, but the strict objective has only `9`
    deployable positives and only `5` held-out positives. The diagnostic
    selected `3 / 61` overrides at `33.33%` precision/safe with worse mean
    risk. Do not train/live-screen from this sparse selector; mine more
    strict-compatible positives first.
  - high-confidence Run791 strict selector: label mining is now much healthier
    (`32` common8-stable strict-compatible positives), but the same legal-set
    objective still failed grouped validation (`7 / 55` overrides,
    `42.86%` precision, `71.43%` safe). Do not live-screen this selector; the
    next change must improve separation from hard negatives.
- Proven and reusable:
  - survival-grammar teacher distillation (Run 553);
  - ONE cycle of shield-behavior distillation (Run 604, +0.86 raw strength);
  - the deterministic table-protective liveness shield at inference;
  - the exact own-hand liveness oracle (`quantum_cat_liveness_oracle.py`,
    brute-force-validated) for training labels and action filters;
  - `liveness_shield_oracle_min_margin` as a strength-vs-survival dial
    (favor strength for human-facing play);
  - MIXING dense label families beats any single family in 4p screens, but
    Run726 showed the m200 effect is modest (Run720 true rate `68.87%`);
  - dense label generators (`--dense-label-kind table_liveness` /
    `led_preservation`), aux-stacks-only training
    (`--train-action-aux-stacks-only`), weighted replay mixing, causal-pair
    mining (707/723/733) as diagnostics, grouped validation via preserved
    `row_group_ids` (721).
  - hidden-information exact legal-pressure diagnostics (734) as training-time
    label targets only: they improved same-seed m30 but are not deployable and
    are not sufficient as one-step selectors.
  - hidden-information 4p table-liveness teacher (735) as the strongest current
    oracle label source: `35.75%` m200, CI upper `39.07%`, still training-time
    only.
  - action-conditioned aux stacks can learn the table-liveness risk/value
    labels very well (Run736: `risk corr 0.9716`, `value corr 0.9756`); the
    missing piece is calibrated deployment/selector use, not raw capacity.
  - the existing deployable `ImproveMLP`/`improve_graft` path is mechanically
    usable for information-legal table-liveness transfer (Run738); future use
    must target safe coverage, not just high offline precision.
  - `improve_graft` can now be trained and deployed against a shielded base
    action (`--base-action-source metadata_policy_best`,
    `improve_base_mode=liveness_shield`, Run745). Use it only with grouped
    validation and a minimum held-out override-count gate; Run746 shows the
    current feature/objective setup is not enough.
  - failed-selector shadow mining against the liveness base is now available
    (Run750), and top-k same-state candidate auditing/replay labeling is
    available for pairwise/listwise objectives (Run751). Explicit trick quotas
    and per-trick threshold diagnostics are available (Run752). The next useful
    corpus/model should improve calibration on this source, not change promotion
    rules.
  - 5p cell-exit protection mining is feasible as a diagnostic/data source
    (Run785), but it needs stronger hard-negative or repeated-continuation
    labels before it should drive another model update or live screen. Run786
    confirms abstention-weight tuning alone is not that stronger label. Run787
    provides the continuation-audit path needed to create that stronger label.
    Run788 provides the continuation-filtered replay exporter, but the first
    40-row slice is too sparse for deployment evidence. Run789 provides the
    all-172 common1 continuation corpus and shows the next useful source is
    repeated-rollout confirmation and more led-lane-compatible positives, not
    another selector-threshold sweep. Run790 proves the repeated-rollout
    validation path can identify stable survival positives (`26 / 26` common8
    confirmed) and that led-compatible mining can add strict-compatible
    positives, but the yield (`9` strict-compatible positives total) must be
    scaled before another trainer screen. Run791 adds the reusable
    `quantum_cat_led_compatible_selector_builder.py` and proves the scaled
    source works: `32 / 32` led-compatible any-risk-safer candidates survived
    common8. The open problem is now selector discrimination, not label
    existence. Run792 shows a merged binary abstainer can pass grouped and
    cross-source offline gates on the curated high-confidence pairs, but live
    `improve_graft` still regresses at m30 (`80 / 150 = 53.33%` versus
    shadow/control `77 / 150 = 51.33%`). Keep its runtime filters
    (`improve_led_colors`, candidate color, base-off-led, responsible-red-win)
    as infrastructure, but do not promote or repeat threshold sweeps on this
    scorer family. Run793 fixes the DAgger miner so it mirrors live guarded
    `improve_graft` and the q-policy base, then adds `32` live-distribution
    shadow rows (`13` positive, `19` negative) to the high-confidence bank.
    The augmented abstainer passes offline and improves one m10 screen
    (`26 / 50` versus `33 / 50` control), but fails m30 (`82 / 150` versus
    `77 / 150` control), including at a stricter threshold. The next selector
    must compare the legal set jointly with base-as-abstain, not another
    standalone binary `ImproveMLP`. Run794/795 tested that legal-set direction
    on the Run790 + Run791 + Run793 bank; source-grouped validation exposed the
    offline illusion again, and the positive-head version was safer but too
    sparse (`3` overrides at `66.67%` precision). Run796 shows stable
    q-teacher labels are too expensive without batching/caching. Run797 shows
    deterministic table-liveness action-value labels are promising and
    led-disciplined, but the direct table-liveness policy target itself
    over-burns led tokens. Run798/799 show the current positive-head legal-set
    selector can be clean on a smoke shard but does not generalize when scaled.
    Run800 adds that stronger cross-shard gate and the selector fails both
    directions, even with a larger independent holdout shard. Use Run797/800
    value/risk labels only after changing the target decomposition or mining
    cross-shard hard negatives; do not direct-clone the deterministic
    table-liveness policy target or live-screen the current legal-set selector.
- Information hygiene: the `mcts` eval mode searches the true hidden state -
  diagnostics only, never deployment or promotion evidence. Search at
  inference must use belief modes that sample from the player's own
  infostate.
- Bidding stays on the shared top-rank heuristic
  (`quantum_cat_ai.prediction_top_rank_score`) for ALL bots and training
  data unless a learned bidder beats it in a 600-match paired gate.

## Training phases (kept, refined)

- **Phase 1 - survival / legality-liquidity pretraining** (proven in 3p):
  teacher distillation into the policy stack with KL anchor; for 4p/5p the
  phase-1 transfer path itself is the open problem - no m200-verified
  deployable 4p source below `60.12%` or deployable 5p source below `53.80%`
  exists yet, even though Run775 now provides a hidden-info 5p teacher at
  `20.20%`. Run730 proves weak-teacher distillation can move live 4p survival,
  but not enough; Run732 shows blind 5p survived-behavior cloning can move the
  wrong way; Run775 shows pure behavior cloning can fail even from a strong
  hidden-info 5p teacher; Run776 shows dense exact-table labels help only when
  kept as aux selector heads so far, while broad policy movement from the same
  labels regresses live play; Run777 shows a clean red-switch contrast
  curriculum teaches the exception but does not improve survival, so red-switch
  should remain an audit constraint rather than the main curriculum; Run778
  shows a small late-exit aux-only replay from the already-tight Run776 step20
  distribution also regresses, so the next 5p survival pretraining needs
  upstream causal/veto labels rather than another late-only ranking replay;
  Run779 confirms sparse upstream low-exit causal labels are the best 5p
   transfer direction so far (`46.67%` m30 best), while Run780/781 show naive
   dense soft all-legal warning labels regress despite more labels and compacting
   the candidate tensor does not help; Run782/783/784 show hard two-action
   contrasts, binary all-legal labels, direct policy-veto training, and
   unlimited q-policy overrides are worse; Run785 shows 5p cell-exit labels are
   feasible but still not transferable without stronger hard-negative or
   repeated-continuation calibration; Run786 shows base-weighted abstention
   calibration does not repair that transfer; Run787 shows why: unfiltered
   one-step cell-exit target changes are mostly not continuation-survival
   improvements; Run788 converts those audits into filtered replay rows, but
   the first tiny slice fails legal-set validation and must be scaled/balanced
   before another screen; Run789 scales to all `172` target changes and still
   fails grouped validation; Run790 confirms the repeated-continuation survival
   signal is real (`26 / 26` common8 positives), but only `9`
  strict-compatible positives remain after deployable led-lane shaping, so the
  next 5p pretraining data needs scaled led-compatible/repeated-positive
  mining before another trainer screen; Run791 does that and finds `32`
  fresh strict-compatible common8 positives, but the current legal-set
  selector still fails grouped precision/safety, so the next 5p pretraining
  work needs better selector targets/features/loss or merged larger positive
  banks rather than a live screen. Run792 merges those high-confidence banks
  into a binary abstainer and finally gets clean offline separation, but live
  same-policy deployment still fails because the scorer fires on broader
  follow-rank/candidate-distribution cases than the curated pair labels; the
  m30 confirm regresses and red-switch-only guarded deployment is too sparse to
  affect the gate. The next 5p attempt should train a stronger legal-set or
  trajectory-level selector over broader repeated-continuation labels, not
  another `ImproveMLP` threshold/guard sweep. Run793 confirms that even after
  collecting live-distribution DAgger hard negatives, the independent binary
  scorer over-fires on same-color rank rewrites at m30; use the Run793
  top-k/sidecar data as input to a legal-set/base-abstain objective, not as a
  reason to continue binary threshold tuning. Run733 shows 4p strict causal-pair
   data is real but one safe-vs-selected policy dose
  still fails; Run734 shows immediate
  forced-only guards are too late while broader exact-pressure is useful but
  insufficient. The next improvement
  likely needs calibrated use of the Run735 all-legal table-liveness teacher:
  a live-distribution teacher approximation, DAgger-style teacher-corrected
  substitution, or a selector/top-action objective that explicitly trades
  precision for safe coverage. Run740 shows that a larger balanced on-policy
  corpus plus soft/listwise CE is still not enough if the deployed argmax
  action does not improve. Run741 shows the mirror failure: argmax/top-1 can
  improve offline while live survival worsens if the objective does not
  preserve led-follow discipline and constrain red-switch/token-loss moves.
  Run742 shows that filtering rows to follow-led or red-winning teacher
  actions is not sufficient; the target/action distribution itself needs a
  discipline-preserving gate or penalty. Run743 and Run744 show that
  action-level anchoring and oversampled follow-led corrections can preserve
  discipline, but the current policy-stack-only KL-anchored recipe does not
  move deployed argmax enough. Run745 shows liveness-base `improve_graft`
  composition is mechanically available, but tiny held-out validation is not
  enough evidence. Run746 then shows that simply scaling this exact
  `ImproveMLP` selector to D320/D640 still fails grouped safe-coverage gates.
  Run747 shows the obvious public pressure features are already present, but
  the current objective cannot separate hard-negative rescue lookalikes from
  rare large-margin positives. Run748 shows that adding same-state CE and
  pairwise-margin losses to the same small independent `ImproveMLP` still fails
  grouped safe coverage. Run749 shows a true legal-set selector with
  base-as-abstain also fails on the same D320/D640 corpus. The next Phase-1 work
  should change data source before another liveness-base graft screen. Run750
  shows that failed-selector shadow proposal mining is the right data-source
  shape, Run751 shows that top-k same-state auditing fixes the pairwise/listwise
  row shape, Run752 shows that explicit trick quotas can build a
  quota-balanced `128`-row bank, Run753 makes trick-gated deployment/eval
  possible, Run754 shows that simply scaling trick-5 rows under the same
  summary-feature selector does not fix precision, Run755 shows that the
  existing DeepSets legal-set selector does not fix it either, and Run756 shows
  that preserving responsible red-win base actions is already handled by the
  current liveness candidate ordering on the screened seed. Run757 shows that
  the deployed-style liveness shield also does not switch away when the led
  color is legally available; Run758 shows most no-led-color switches newly burn
  an available led token, and red choice on those burns is already mostly
  handled by the shield. Run759 shows the direct upstream cause is usually
  another player claiming the exact future color/rank exit cell. Run760 now
  provides dense anchored labels for that failure mode. The failure is earlier
  table-wide rank/cell/exit exhaustion. The current selector family still fails
  the grouped and trick-filtered offline gates.
  Do not spend more cycles on one-step cloning, uncalibrated q-policy dials,
  small high-margin pairwise grafts from the same replay, or soft/listwise
  policy-target sweeps that only improve CE/probability mass. Also do not
  repeat all-row hard table-liveness top-action cloning or row-filter-only
  constrained cloning without action-level discipline gates; do not repeat
  anchor-follow oversampling/threshold tweaks at current scale; do not
  live-screen liveness-base grafts from tiny validation sets with fewer than
  about `8-10` held-out safe positive overrides; do not keep scaling or
  knob-sweeping selector architectures on the current D320/D640 liveness-base
  bank without changing the data source; do not promote or live-screen the
  Run750 `32`-row diagnostic selector, the Run751 `8`-row top-k smoke, any
  Run752 selector variant, the Run753 trick-5-gated near miss, Run754
  scale-only trick-5 summary-feature variants, the Run755 legal-set rerun, or
  the Run756 responsible-red-win guard.
  Use `improve_trick_numbers` only after a trick-filtered held-out gate passes.
  Do keep the Run750-755 shadow mining/eval machinery, but the next selector
  needs better labels (`audit_rollouts > 1`, repeated-seed/exact continuation
  labels) or deterministic led-token/red-rescue targets for early
  false-positive lookalikes. After Run757, prefer labels that blame upstream
  moves for reducing future led-color legal availability over immediate
  follow-led labels. After Run758, make that even sharper: target states where
  the future led token is still available but no led-color action is playable.
  After Run759, target the prior color/rank cell claims that take another
  player's future potential led-color action count from positive to zero while
  their token is still available. After Run760, train from the anchored
  cell-exit replay rather than a naive least-damage target; the naive variant
  over-selected red. After Run763, do not live-screen unguarded cell-exit
  improve-grafts: use led-lane preservation and urgent-red constraints in the
  labels or enable `improve_guard_led_follow` from the first screen. After
  Run764, do not repeat Run760 trick-3 improve-grafting with only stricter
  led-lane label shaping as the survival intervention; trick-gated deployment
  ties m30 control and all-play deployment regresses at m10, with both still
  around `65-70%`. After Run765, also do not repeat dense handoff-pressure
  exact-exit improve-grafting as a standalone selector on top of Run730: even
  with `improve_guard_led_follow` and the new `improve_max_policy_log_gap`
  guard, the m30 confirm regressed to `84 / 120 = 70.0%` versus same-seed
  Run730 control `75 / 120 = 62.5%`. Keep the policy-gap guard as
  infrastructure, but move the next survival work toward trajectory-level
  survival targets or counterfactual continuation labels for both base and
  candidate actions. After Run766, the most promising focused slice is
  non-red-led/follow-legal continuation labeling from the Run730 liveness
  distribution, but do not confuse it with a red-switch curriculum: in the
  scaled diagnostic, the base followed led in every evaluated row, label-best
  actions still followed led in `44 / 59` rows, and red switches were safe only
  `47 / 220 = 21.4%` of labeled action outcomes. The missing skill appears to
  be led-color rank/exit allocation inside the lane, not "switch to red more."
  The small Run766 graft did not pass grouped held-out coverage (`1 / 15`
  validation override; gate required `4`), so no live screen was warranted.
  After Run767, this focused led-rank slice is confirmed as real but not yet
  transferable with the small independent `ImproveMLP`: the larger d182 corpus
  kept the base action follow-led in `182 / 182` rows, label-best still
  followed led in `131 / 182` rows, and red-switch candidates were only `20.6%`
  survival-safe despite winning the trick whenever `would_win_now` was
  observable. The v1 graft overfit grouped validation (`4 / 47` overrides at
  `50%` safe/precision), and the precision-shaped v2 made `0 / 47` held-out
  overrides. Do not repeat threshold/regularization sweeps on this selector
  family; the next attempt should use a stronger action-conditioned/legal-set
  model or trajectory-level table-wide survival targets over the exact
  continuation data, still gated by grouped held-out safe override coverage
  before any live screen.
  After Run768, the first legal-set attempt on that exact d182 corpus also
  fails grouped held-out transfer. A DeepSets-style `LegalSetSelector` can fit
  the corpus, and legal-set led-lane shaping correctly forced `735` off-led
  lane-burn candidates negative while preserving `62` urgent red-win
  exceptions, but validation still collapses: v1 only finds `2 / 48` high-safe
  held-out overrides, v2 finds `4 / 48` at `50%`, and v3 shaped returns to
  `2 / 48`. Do not add runtime/deployer support for `LegalSetSelector` until a
  grouped gate passes offline. The next useful step is broader/source-balanced
  continuation data or trajectory-level table-wide survival targets, not more
  small-model thresholding on d182.
  After Run794/Run795, the same lesson holds on the newer Run790 + Run791 +
  Run793 combined corpus. Multi-source legal-set training is now supported with
  source-prefixed metadata groups, and the positive-head legal-set variant
  removed most false positives, but the best held-out setting still found only
  `3` overrides at `66.67%` precision. Do not add runtime support or live-screen
  legal-set selectors until grouped offline coverage and precision both pass.
  The next useful step is cleaner/source-balanced contrast generation or
  trajectory-level table-wide survival labels, not more threshold sweeps on the
  current `381`-state replay mix. After Run796, stable q-teacher full-match
  labels are also not the next cheap answer: balanced state collection works,
  but scoring is too slow at current cost. After Run797, deterministic exact
  table-liveness replay is the best fresh label source, but only through its
  action-value/action-risk labels: the value-best action follows led in
  `635 / 640` scaled rows, while direct table-liveness policy top actions burn
  the led token in `107 / 640`. After Run798/Run799, the current legal-set
  selector remains below the offline gate (`3` clean smoke overrides, then
  `6` scaled overrides at `50%` precision/safe). Next should be cross-shard
  validation plus stronger false-positive calibration around token-loss
  conflicts, ideally with batching before any larger trainer loop. After
  Run800, cross-shard validation exists and is the right gate, but the current
  selector fails it in both directions. Do not keep training this selector on
  more table-liveness shards without changing the learning target or adding
  explicit false-positive hard negatives from source-holdout failures. After
  Run801, smooth value/risk target decomposition is also insufficient by
  itself: the deltas regress well across shards, but selected overrides remain
  false-positive heavy. After Run802, the synthetic pairwise/lookalike boundary
  is also insufficient: it improves the best reverse split only to `10 / 640`
  overrides at `60.00%` precision / `80.00%` safe and collapses in the other
  direction. After Run803, actual selected false positives also fail to transfer:
  the cross-fit variant overfits the mined shards (`96.15%` precision/safe on
  the mine slice) but reaches only `56.00%` precision / `64.00%` safe on a third
  shard. The next table-liveness attempt should change representation and
  validation structure: multi-shard leave-one-source-out training, source-
  invariant regularization, richer state/hand/action-set context, or sequence
  context. After Run804, simple multi-source training, raw observation appends,
  and source-adversarial regularization are also insufficient. Do not run
  another local approval-boundary, raw-feature append, or source-adversary
  threshold/weight sweep. After Run805, a basic action-set Transformer also
  overfits the source shards, though a regularized version found a tiny clean
  third-shard slice. After Run806, that tiny clean slice failed calibrated
  transfer to a fourth shard (`2 / 1660` overrides at `0.00%` precision /
  `50.00%` safe). After Run807, cheap within-round aggregate context is also
  insufficient: best held-out slice was only `30.00%` precision/safe, and the
  selected false positives all followed led and lost no token. Next should be
  explicit intra-lane rank/exit allocation training or richer board/trick-
  prefix state, not generic round-context aggregates, current-transformer
  threshold sweeps, or a bigger version of the same model. After Run808, the
  narrow follow-led/no-token-loss rank filter also overfits source shards and
  fails transfer, so the next attempt needs new information: explicit
  board-cell/exit scarcity features or repeated exact continuation labels for
  same-state follow-led rank choices. After Run809, explicit intralane scarcity
  features modestly expand the tiny clean held-out island but still fail
  deployable coverage. After Run810, exact table-liveness intralane labels are
  finally source-stable enough to pass an offline held-out gate, and
  `:intralane_filter` runtime support exists, but live 5p m10 only moves
  `29 / 50 -> 28 / 50` any-paradox rounds. Treat intralane rank allocation as
  a useful micro-skill, not the survival fix. After Run811, raw-policy source
  generation still produced `0` base-lane-burn rows, so the next miner must
  explicitly select non-red-led states where the base action is off-led and a
  legal follow-led action exists before training another rescue selector. After
  Run812, that explicit census falsified the base-lane-burn hypothesis for the
  current 5p checkpoint/window: across `1996` trick-2-to-5 play decisions,
  `led_follow_available=1386` but `base_lost_led_follow_available=0`. Forced
  switch quality is weak (`28 / 211 = 13.27%` red switches and `16 / 211 =
  7.58%` switch wins), but a gated red-switch selector was too sparse to solve
  survival (`85 / 150 = 56.67%` raw m30 -> `84 / 150 = 56.00%` gated m30).
  First-paradox traces are now the sharper diagnosis: raw m30 had `85 / 85`
  forced first paradoxes, `82 / 85` on trick `6`; gated m30 had `84 / 84`
  forced, `81 / 84` on trick `6`. Next should focus on exact late-exit /
  final-trick allocation, not more midgame followable-lane-burn mining.
  After Run813, that diagnosis is confirmed in a useful way: the learned
  late-exit selector overfit the offline labels and failed live (`70.00%`
  no-gap, `54.67%` with a policy-gap cap), but the exact late liveness teacher
  itself passed m200 at `321 / 1000 = 32.10%`, CI `[29.21%, 34.99%]`. Treat it
  as the current best 5p oracle teacher for distillation/contrast labels, not
  as a legal inference-time config. Run814 then proved the next replay path:
  capped exact-teacher candidate rows (`top_policy_actions=4`,
  `max_actions_per_state=8`) produced `120` deterministic-teacher policy-target
  rows from `5` matches at a feasible cost, while all-legal exact rows were too
  slow for interactive iteration. The scaled Run814 continuation then produced
  `260` train rows and `130` independent validation rows from the same exact
  teacher. Run815 showed direct policy-stack distillation can learn those
  labels offline (`0.8154` validation top-1) but does not transfer to live 5p
  survival (`33 / 50 = 66.00%` versus same-seed Run779 raw
  `31 / 50 = 62.00%`). Run816 showed the inherited Run779 action-value head
  already ranks the exact-late labels well offline (`0.9818` validation top-1),
  but q-policy deployment still fails: unguarded value reranking catastrophically
  over-steers (`50 / 50 = 100%`), gap-guarded lower-pressure q-policy improves
  only to `28 / 50 = 56.00%`, and late-only min-trick-5 q-policy ties or
  regresses the raw control. Keep the new
  `action_value_rerank_min_trick_number` / `max_trick_number` guard as
  infrastructure, but do not promote Run815/Run816. Run817 mined explicit
  exact-shield-veto contrast labels and added all-legal-negative replay support,
  but the result was diagnostic rather than deployable: aggregate follow-led
  and token-loss counts were identical between raw/base and shield, so the
  exact shield is mostly correcting subtle late allocation choices inside a
  similar legal/action-feature family. Raw-only veto labels were already ranked
  almost perfectly by the inherited Run779 action-value head (`0.9818`
  validation top-1), while reset/trained raw-only and all-legal-negative
  Q-stacks were worse (`0.8197` and `0.5246` validation top-1 respectively).
  Keep `--negative-scope=legal` as useful replay infrastructure, but do not run
  more small reset-Q fits on this shard. Run818 then tried the intended
  preserve-the-head route: a frozen-Run779 residual selector trained on all
  exact-shield behavior rows, including no-veto abstention rows. It also fails:
  the full-feature selector memorizes train but reaches only `0.3889`
  validation override precision at its best threshold, while the scalar-only
  selector is more conservative but covers only `2` overrides at `0.5000`
  precision. Frozen Q margins are catastrophically uncalibrated for abstention
  on the same rows (`0.0094` best precision, `0.0000` no-veto abstain), which
  explains Run816 oversteer. Treat exact hidden-shield labels as oracle
  diagnostics, not a directly cloneable deployment teacher. Run819 built the
  first belief-consistent late-exit audit: sample information-state
  determinizations, score late exits across those worlds, and compare belief
  labels to true-state table-liveness. Uniform infostate sampling is fair and
  cheap, but too noisy to train from directly: on the scaled `300`-state audit,
  belief overrides occur in `141 / 300 = 47.00%` of late states but match the
  true-state oracle only `43 / 141 = 30.50%`; even high-margin overrides at
  margin `0.50` improve true-state table-liveness only `7 / 20 = 35.00%` of
  the time and are non-worse only `9 / 20 = 45.00%`. Policy-weighted belief
  sampling may be the right fair teacher, but the current implementation is too
  slow: a `40`-state smoke with `8` samples and `32` candidates was interrupted
  after roughly two minutes inside repeated public-history likelihood scoring.
  Next should optimize/cap/cache policy-weighted belief scoring, then rerun the
  same late-exit audit before generating replay or training.
  Runs820-822 did optimize and confirm that path: full-history posterior
  scoring got faster, but the larger fixed-RNG audits did not beat uniform
  belief on base-action override precision, so posterior labels are not the
  current training source. Run823 switched to direct information-legal
  contrasts: same public state, base action versus a follow-led/red-switch
  replacement, accepted only when `3` common-seed continuation rollouts agree
  the base reaches forced paradox while the replacement survives. The first
  hard-negative bank is small but clean: `54` 5p rows, `284` extra off-led
  non-red lane-burn negatives, source-held-out selector probes at `23 / 31` and
  `13 / 20` overrides with `100%` positive precision/safe rate and `0` forced
  lane-burn selections. Do not promote a checkpoint from this yet; next should
  scale this hard-negative contrast generator across more seeds, then train an
  action-conditioned/legal-set selector only if larger source-held-out gates
  stay clean. Runs824-827 scaled and stress-tested that branch: Run824's
  scaled hard-negative labels stayed clean offline but failed live 5p survival,
  Run825's direct pairwise-boundary deployment also failed live, Run826's live
  proposal mining exposed that the boundary misses many red-rescue label-best
  actions, and Run827 showed that DeepSets/Transformer setwise models on the
  current `160`-state bank still fail held-out-source precision. Next should
  change the data/candidate source, not reweight this same bank. Run828 then
  fixed the collection shape by exploring red-rich live states while labeling
  against the frozen policy base (`129 / 192` behavior-red rows, `42 / 192`
  label-best-red rows), but single-rollout dense labels still failed source
  holdouts. Runs829-838 turned the current "Strawberry/RFT-style" idea into a
  real data path: candidate actions are sampled from red/follow/policy queues,
  repeat-validated under common continuations, and written as exclusive
  base-plus-candidates labels. Run838 reached `153` clean rows across `36`
  sources, all with base plus six candidate labels and `93.7%` selected-red
  targets in the scaled tranche. Run839 then falsified the first training hope:
  both soft and hard action-set selectors memorized train labels but failed
  source-held-out precision (`44-62%`, gate requires `>=90%`). Run840 filtered
  to the decisive high-margin island (`46 / 153` rows, all red targets), but
  held-out precision still failed badly (`8-54%`). Conclusion: RFT-style
  collection is mechanically working, but the current action-set selector lacks
  source-stable information for ranking red rescue choices. A corrected
  non-leaky one-feature diagnostic on the high-margin bank also failed
  (`27-44%` held-out precision for the best simple rule), so the answer is not
  just a quick threshold over the existing exit features. Runs841-842 tested
  two obvious model-side repairs on the same high-margin island: a same-state
  pairwise boundary (`7-44%` precision) and source-adversarial delta training
  (`17-50%` precision). Both failed. A feature-distinguishability audit showed
  the target red actions are not exactly feature-aliased with the other labeled
  actions (`0 / 46` identical), but the current labels/features still do not
  yield a source-stable ranking. Do not live-screen or promote from
  Runs839-842. Run843 then tested a later-trick version of the same
  RFT-style collector (`trick_number 2..5`) with stricter four-rollout repeat
  validation. The plumbing worked, including collector pass-through for
  `--min-trick-number`, `--max-trick-number`, and `--max-led-bucket-rows`,
  but yield was only `1` admitted row from `48` attempts across six shards.
  The single row had a passed red-switch candidate, but the selected target was
  yellow, so late-trick strict labels are diagnostic and too sparse for
  training. Runs844-845 added diagnostic attempt recording and proved the next
  important thing: the sparsity is not mostly from thresholds being barely too
  strict, nor from testing only six candidate actions. On Run844, relaxed
  threshold grids found the same `2 / 16` qualifying attempts as the strict
  verifier; on Run845, widening to `12` candidates found only `1 / 9`
  qualifying attempts and `0` qualifying red switches. Run846 then tested that
  exact shift earlier: trick `0..1`, same strict verifier, same six-candidate
  queue. It produced `9` clean exclusive rows from `24` attempts (`37.5%`),
  every row had a passed red-switch candidate, and `8 / 9` selected targets
  were red. This strongly supports early/pre-collapse data as the next source.
  Run847 tried scaling immediately with larger shards and `4` workers, but was
  interrupted after the first worker batch stopped updating for several
  minutes; no completed data artifacts came out of it. Run848 retried with
  Run846-sized shards and `2` workers, but also stopped updating after the
  first progress line and was interrupted before artifacts were written. The
  partial logs in both runs still supported early-slice yield, but the collector
  needed per-progress partial save or a more responsive cancellation/timeout
  shape before larger interactive scale-ups. Run849 added the partial-save path:
  `--partial-save-interval-decisions` now writes partial metadata, attempt
  rows, summaries, and partial replay rows when present; the collector forwards
  the flag, and a one-decision smoke verified partial/final attempt counts
  match. Run850 completed the first tiny partial-save early continuation:
  `6` rows from `12` attempts, all selected red, every row with a passed
  red-switch candidate, and all final replay rows at exactly `7 / 7` action
  target and value labels. Run851 then completed eight four-decision shards:
  `12` rows from `32` attempts, `11 / 12` selected red, every row with a
  passed red-switch candidate, and the same `7 / 7` mask audit. Runs852-853
  added `quantum_cat_early_bank_builder.py` and used it for one smoke plus two
  real two-decision batches. Runs854-855 then scaled the same strict exclusive
  early-bank recipe through five more batches. Combined with Run846, the early
  bank is now `64` rows from `154` attempts and remains consistent (`59 / 64`
  selected red, `64 / 64` with a passed red switch, `8 / 64` with passed
  follow-led, and all new rows at `7 / 7` action target/value labels). Next
  should keep scaling early pre-collapse data through the bank builder until
  roughly `150-200` rows, and run the same source-held-out selector gates that
  Runs839-842 failed only once the bank is large enough. Runs938-940 then
  revived the Strawberry/RFT direction with repeated red-augmented slates and
  source-held-out DeepSet/listwise screening; Run940 reached `9 / 9` fold gates
  and `27 / 27` held-out top-1 safe choices on saved slates. Run941 wired that
  selector into the live evaluator as `proposal_veto_top1`, but the best
  augmented threshold smoke was still `11 / 25 = 44.0%` traced paradox rounds,
  above the strict `<40%` gate. Do not promote Run941. The live adapter is
  useful infrastructure, but its deployed features currently omit the
  round-history context used during training (`history_available=0` in the
  smokes). Run942 tested the obvious follow-up: a no-history/live-feature
  DeepSet reached `27 / 27` all-group top-1 safe offline, but its live smoke was
  `14 / 25 = 56.0%` paradox rounds at threshold `0.80`. A simple source-margin
  agreement guard also failed: on the same seed as Run941's `44.0%` near-miss,
  `proposal_veto_min_source_margin=0.0` worsened survival to `14 / 25 =
  56.0%`. Do not keep tuning these simple thresholds; next should mine the live
  false-positive overrides and train contrast labels that explicitly punish bad
  proposal-veto overrides, or reconstruct richer live history before retrying.
  Run943 and Run944 started that next data loop. Run943 wired the live final
  `proposal_veto_top1` stack into `quantum_cat_live_substitution_replay.py` as
  a proposal source and mined `36` real override states from `352` considered
  5p proposal states. The audit exposed the exact missing signal: `16 / 36`
  selected candidates had candidate-any-paradox, only `7 / 36` were
  dense-positive, `10 / 36` slates contained a clean positive candidate, and
  `6 / 36` had both a bad selected candidate and a clean alternative in the
  same slate. The first 10-source contrast retrain failed validation (`10 / 32`
  approved top-1 on the Run943 validation source), so do not promote it.
  Run944 then collected the higher-signal version: repeat-validated mixed
  proposal-veto slates. It accepted `8` rows from `360` considered states, all
  `8 / 8` repeat-validation attempts passed, `5 / 8` rows were relabelled by
  the common-seed validator, and all rows wrote exclusive action targets. A
  tiny source-held-out screen passed at threshold `0.80` with `3` clean
  approvals, but the live 5p same-policy smoke still failed at `15 / 25 =
  60.0%` traced paradox rounds; the same-seed raw policy was also `15 / 25 =
  60.0%`. Treat Run944 as data/infrastructure progress, not a model candidate.
  Run945 added source-confidence penalties/guards and passed the same tiny
  holdout, but live smokes still failed (`56.0%` at threshold `0.70`, `64.0%`
  at threshold `0.60`, and `60.0%` when the hard guard mostly disabled
  overrides), so source-confidence thresholding is not enough. Run946 tested the
  more literal ReST/STaR/expert-iteration analogue: blend repeat-validated
  best-action targets into the base policy target so the verified moves become
  slightly more likely. The plumbing worked and held-out target probability rose
  through steps `10/20/30`, but top-1 did not flip and live survival did not
  improve (`64.0%`, `80.0%`, `60.0%`; raw same-seed control `60.0%`). Treat
  Run946 as proof that best-of-N policy nudging is possible, not as a model
  candidate. Run947 made that objective stronger by transforming policies to a
  positive-value-softmax target over repeat-validated positive actions. It
  flipped the target top action on all train and held-out rows, reached `25%`
  held-out top-1 imitation, and proved the raw policy can be pushed toward
  best-of-N targets. Live transfer still failed: step `20` was only `14 / 25 =
  56.0%` same-policy paradox rounds, while steps `40/60/80` worsened to
  `72.0%`, `72.0%`, and `80.0%`. Treat Run947 as evidence that the algorithmic
  idea applies, but the tiny `21`-row train bank overfits/misgeneralizes. Next
  should scale source-diverse repeat-validated policy-target data first, then
  repeat the advantage-weighted policy update with source-family holdouts and a
  cautious dose screen. Run948 did that first scale-up modestly: the bank grew
  to `34` rows from `12` sources, preserved the repeated red-augmented slate
  invariants, and again flipped transformed targets offline. Live transfer got
  worse, not better: raw same-policy smokes were `72.0%`, `76.0%`, `80.0%`,
  and `80.0%` paradox rounds across steps `10/20/30/40`. Do not promote
  Run948. The next move should not be more epochs on this tiny positive-only
  bank. Either scale to hundreds of rows with source-family holdouts before
  another raw-policy update, or pivot the best-of-N signal into a contrast,
  reranker, or auxiliary objective that also penalizes live bad updates rather
  than only cloning local positive actions. Run949 tested that reranker pivot
  on the same `34` red-augmented repeated rows and `12` sources. Offline it
  looked excellent: the history-feature DeepSet passed a source-held-out gate
  and both trained proposal rerankers chose `34 / 34` safe/approved saved-slate
  top actions. Live transfer still failed: 5p same-policy smokes were `56.0%`
  and `68.0%` for the history-feature model, `52.0%` for the live-feature
  model, and still `52.0%` after guarding overrides with
  `pred_value_delta >= 0`. The key diagnostic is that live overrides often
  contradicted the source scorer itself; at threshold `0.60`, `70 / 73`
  sampled overrides had negative predicted value delta. Do not promote Run949.
  The next move should mine Run949/Run899 live false positives, label the
  actual on-policy override distribution with continuations, and train a
  negative-aware proposal objective that directly penalizes harmful approvals
  while preserving genuinely responsible red/follow-led rescues.
  Run950 did that first live-distribution pass and found two important things.
  First, the proposal-veto trainer had a real feature bug: it was reading
  follow-led/red/token-loss booleans from pair/context feature vectors, not raw
  action features. This is now fixed by carrying `raw_action_feature` through
  `_build_states()` and using it for tactical proposal flags. Second, the
  focused red/follow collector finally produced the right bank shape: `38`
  non-red-led rows where both follow-led and red-switch actions are legal,
  expanding to `289` scored candidates with `57` strict hard-lane approvals and
  `151` forced lane-burn negatives. However, all Run950 verifier screens still
  failed source-held-out gates: the best red/follow hard-lane folds reached only
  `40.9-50.0%` approval precision and still approved too many candidate-any
  paradox lines. Do not promote Run950. The next move is to scale the focused
  red/follow mixed bank across many more source shards and train with
  candidate-any-paradox as a first-class negative, not merely a soft risk
  penalty.
  Run951 executed that scale-up and falsified the easy version. The focused
  bank is now `70` rows, `149` above-threshold non-red-led proposal states, and
  `544` outcome-scored candidates, including `198` clean positives, `339`
  candidate-any-paradox candidates, `206` follow-led candidates, `207`
  red-switch candidates, and `57` responsible red-win switch candidates. But
  five-source hard-target `bce_listwise` still passed only `1 / 5` folds, and
  only on a tiny `3`-approval island; aggregate held-out top-1 was approved in
  only `22 / 70` groups and candidate-any-paradox in `46 / 70`. Do not promote
  Run951. The next move is not more training on the same single-continuation
  labels; collect repeat-validated live red/follow proposals with
  `proposal_scored_plus_red_follow_value`, zero allowed candidate-any rate, and
  explicit passed-follow-led / passed-red-switch coverage.
  Run952 proved that stricter collector is viable but not yet large enough.
  Two live shards produced `24` strict repeat-validated rows from `39`
  repeat-validation attempts, with `24 / 24` exclusive action targets written
  and `14` target relabels. The bank explicitly required zero candidate-any
  rate plus at least one passed follow-led and one passed red-switch candidate.
  But the tiny two-source verifier smoke still failed both held-out gates:
  validation top-1 was approved in only `9 / 24` groups and
  candidate-any-paradox in `15 / 24`. Do not promote Run952. The next move is
  to scale the same strict repeat-validated red/follow collector to at least
  `5-8` independent sources before rerunning the verifier screens or touching
  the raw policy.
  Run953 did that scale-up to six sources and answered the current
  Strawberry-style question more sharply. The approach is possible and it is
  already the family we are testing: sample candidate moves/continuations,
  score them, and distill or rerank toward the verified better move. But the
  latest bridge failed live. The combined strict bank reached `72` rows from
  `6` sources, and a deterministic offline rule
  `red_switch AND score>=0.9477` passed `6 / 6` source holdouts with
  `92.3-100.0%` precision and `0.0-7.7%` candidate-any rate. When wired into
  live same-policy play as a red-switch-only source-score selector, it regressed
  badly: `43 / 50 = 86.0%` paradox rounds versus the raw Run779 same-seed
  control at `28 / 50 = 56.0%`. Do not promote Run953. The next move is not
  more blind winner cloning; mine the Run953 live false positives and train a
  negative-aware best-of-N verifier or multi-guard rule that penalizes bad red
  switches as well as rewarding repeat-validated winners.
  Run954 did that first negative-aware pass. Source-value/margin guards reduced
  the source-score bridge damage but did not beat raw policy (`60.0%` paradox
  rounds for both guarded m10 probes vs raw `56.0%`). A new live negative shard
  then mined `12` candidate-any rows from `23` above-threshold Run953
  source-score proposals, confirming that the failed bridge regularly proposes
  candidate-own any-paradox red switches. The strict mixed deterministic audit
  failed completely (`0 / 7` selected holdout passes), and a first neural
  DeepSet trained on six sources with the new negative source failed held-out
  source `5`: best ordinary threshold still approved `17.65%` candidate-any
  actions, while ultra-high thresholds found only `1` clean held-out approval.
  Do not promote Run954. Next step: collect several more independent
  `candidate_any_paradox` negative shards from the failed Run953 bridge, then
  rerun strict source-held-out verifier screens with both positive and negative
  holdouts.
	  Run955 did that scale-up and closed the easy repair. Three more independent
	  false-positive shards plus Run954 gave `48` candidate-any negative rows from
  `82` above-threshold source-score proposals and `956` considered proposal
  states. The mixed strict rule audit on six positive sources plus four negative
  sources failed every candidate-own no-paradox target (`0 / 10` selected
  holdout passes for `no_candidate_any`, `positive_no_candidate_any`, and
  `positive_safe_no_candidate_any`). A negative-aware DeepSet also failed both
  holdout directions: positive source `5` had only `1` clean approval at
  threshold `0.9999` and top-1 candidate-any in `5 / 12`; negative source `9`
  had `2` approvals at the best threshold and both were candidate-any paradoxes.
	  Do not promote, install, live-screen, or distill Run955. The next
	  Strawberry/RFT step needs either much more source-diverse positive/negative
	  data or a source-invariant same-state contrast objective that explicitly
	  pairs bad live red-switch proposals against safe alternatives.
	  Run956 tested that objective-side path by adding default-off
	  source-adversarial training to `quantum_cat_proposal_veto_trainer.py` and
	  rerunning DeepSet proposal screens. Source adversary alone failed both
	  holdouts. The stronger `group_relative_soft` target with unsafe penalty
	  `4.0` fixed the positive holdout source `5` (`17` clean approvals at
	  threshold `0.90`, top-1 `0 / 12` candidate-any) but still failed the hard
	  negative holdout source `9` (`1` approval, candidate-any; top-1
	  `3 / 8 = 37.5%` candidate-any). Do not promote, install, live-screen, or
	  distill Run956. The next useful move is to add more source-9-like hard live
	  negatives or a still more explicit same-state bad-proposal-vs-safe-alternative
	  contrast before another live bridge.
	  Run957 tested two cheap objective tweaks before spending live-smoke budget:
	  unsafe penalty `8.0`, and source value/positive/margin penalties on top of
	  the Run956 soft target. Both preserved the positive holdout source `5`;
	  the source-metric variant improved it to `35` clean approvals at threshold
	  `0.95` with top-1 `0 / 12` candidate-any. But both still failed hard
	  negative source `9`: no eligible thresholds, and top-1 candidate-any stayed
	  `3 / 8 = 37.5%`. Do not promote, install, live-screen, or distill Run957.
	  The next step should be data/feature-side: mine more independent
	  source-9-like hard-negative source-score red-switch failures and inspect the
	  exact false positive for the missing discriminating feature.
	  Runs958-960 inspected that exact source-9 false positive and found the
	  missing tactical distinction: the bad proposal was a red off-led lane burn
	  that did not win the trick, while the base followed led. A blunt
	  `--force-led-lane-burn-negative` target killed the false approval but
	  removed all source-9 approved targets, so keep it as evidence rather than a
	  candidate. Run959 added backward-compatible win-aware proposal feature modes
	  and improved source-9 grouped top-1 candidate-any from `3 / 8` to `2 / 8`,
	  but the high-confidence false approval remained. More importantly, Run960
	  fixed the live proposal-veto adapter to pass raw action features into
	  `_proposal_feature`, matching the trainer. A small same-seed m5 live smoke
	  with Run957 `srcpen_neg9`, source margin/value/positive guards, and the
	  fixed live path improved from raw Run779 `17 / 25 = 68.0%` traced paradox
	  rounds to `14 / 25 = 56.0%`. This is directional only, still above the
	  `<40%` gate, and not promotable. Next best move: m50/m100-confirm the
	  triple-guarded selector after the raw-feature fix or train an all-source
	  win-aware proposal-veto model with source-agreement guards before any
	  promotion screen.
	  Run961 did the m50 confirmation. The direction held weakly but remained far
	  too small: raw Run779 same-seed 5p control was `139 / 250 = 55.60%`
	  same-policy paradox rounds, while the triple-guarded proposal-veto selector
	  was `128 / 250 = 51.20%` with overlapping CI and only `180 / 8060`
	  proposal overrides (`2.23%`). Do not promote, install, distill, or spend an
	  m100/m200 promotion screen on this exact guarded selector. The next best
	  cycle is source-held-out win-aware proposal-veto training with source
	  agreement in validation, or more hard live negatives for red lane burns that
	  do not win.
	  Run962 added default-off source-agreement filters to proposal-veto
	  validation reports and ran win-aware DeepSet source-held-out screens with
	  those guards. Positive holdout source `5` passed strongly (`30` clean
	  approvals at threshold `0.90`, `0 / 30` candidate-any, top-1 `0 / 12`
	  candidate-any). But negative sources `6-9` all failed: source guards
	  prevented high-confidence false approvals, yet the models either abstained
	  completely or still had bad grouped top-1 candidate-any (`6/9`, `4/7`,
	  `4/5`, `2/8`). Do not live-screen Run962. Next cycle should mine paired
	  hard negatives with safe alternatives: source-9-like red off-led lane burns
	  that do not win, contrasted against follow-led or genuinely responsible red
	  choices that also satisfy source agreement.
	  Run963 added that exact proposal-slate admission bucket, but the live
	  action-set slates accepted `0` clean paired rows: default slates filtered
	  away forced lane burns, while forced-negative slates did include bad red
	  no-win lane burns but mostly in all-bad states with no non-paradox
	  follow-led alternative. The direct lane-burn contrast miner remains the
	  better upstream curriculum source for this grammar. It produced fresh
	  3p/4p/5p banks (`120` rows each), all with follow-led as the best label.
	  The player-count tensors differ (`3p` obs `76`, `4p` obs `97`, `5p` obs
	  `113`), so this is separate per-player training, not one mixed replay. A
	  5p action-aux smoke showed the inherited heads were inverted on this new
	  target, but resetting action-value/paradox heads fixed it: step40 reached
	  action-value validation corr `0.8009`, top-1 `75.0%`, paradox corr
	  `0.7955`, with frozen policy parameters exact. Do not promote Run963 yet;
	  next cycle should scale reset-head lane-burn pretraining per player count
	  and feed those labels/heads into the proposal-veto contrast objective.
	  Run964 executed the first scaled version for the player counts with
	  compatible aux-stack checkpoints. It mined `500` 4p and `500` 5p
	  lane-burn contrast rows, and in both banks the best label followed the led
	  color in `500 / 500` rows. Reset-head aux pretraining learned the contrast
	  while keeping the policy/value trunk exactly frozen: 4p step80 reached
	  action-value validation corr `0.8775`, top-1 `57.0%`, paradox corr
	  `0.8793`; 5p step80 reached value corr `0.7823`, top-1 `61.0%`, paradox
	  corr `0.7754`. Do not promote Run964 directly; no live same-policy gate was
	  run. The next best use is to feed these heads/labels into the
	  negative-aware proposal-veto objective so bad off-led/red no-win lane burns
	  are penalized, not merely outvoted by positive winner cloning.
	  Run965 wired that bridge. Policy-context checkpoint loading now preserves
	  separate action-value/action-paradox aux encoders, legal-set state building
	  annotates each action with aux value/risk predictions, proposal-veto has
	  opt-in `scalar_action_flags_win_aux_*` feature modes, and the live
	  `proposal_veto_top1` selector computes the same aux features from its base
	  model. A two-source 5p smoke using the Run964 aux checkpoint built a
	  `195`-feature proposal-veto model and a one-match live load smoke scored
	  `161` proposal decisions with `proposal_veto_no_rows=0`. This is
	  infrastructure only: the smoke verifier failed validation and is not a
	  model candidate. Next run should train a real source-held-out 5p
	  aux-feature proposal-veto screen across the scaled red/follow hard-negative
	  sources before any live survival claim.
	  Run966 ran that screen on the five scaled 5p red/follow sources with
	  `scalar_action_flags_win_aux_slate_state_round_history_context`. It failed
	  all `5 / 5` source-held-out gates. The aux signal improved some slices
	  compared with the non-aux Run951 baseline, but held-out top-1 still chose
	  candidate-any actions too often (`4/6`, `13/16`, `9/16`, `9/16`, `9/16`)
	  and no threshold reached the precision/candidate-any/safety gate. Do not
	  live-screen or promote Run966. The next useful move is not another same
	  small verifier with extra features; it is stronger grouped negatives for
	  candidate-any and red no-win lane burns, or more same-state hard negatives
	  where a safe follow-led alternative is actually available.
	  Run967 added that grouped negative objective directly: a default-off
	  unsafe-margin loss that pushes clean same-state alternatives above
	  candidate-any / forced lane-burn proposals. The loss was active
	  (`419-519` clean-vs-unsafe pairs per fold), but all `5 / 5`
	  source-held-out gates still failed. Run968 then moved the survival signal
	  into the main listwise target with `group_relative_soft`, lower
	  temperature, and a stronger unsafe penalty. That was directional:
	  all-slate top-1 candidate-any fell to `8.57-18.57%`, but gate stability
	  was only `1 / 5` and held-out validation top-1 still chose candidate-any
	  too often. Do not live-screen or promote Run967/968. Keep
	  survival-shaped listwise targets as the better objective family, but the
	  next cycle needs more source-diverse on-policy same-state hard negatives
	  or stronger source-regularization before another deployment bridge.
	  Run969 mixed six strict repeated red/follow positive sources with four live
	  false-positive negative sources and added two pieces of infrastructure:
	  default-off inverse-source weighting for the main proposal-veto losses, and
	  a negative-only validation gate where abstaining on a pure hard-negative
	  holdout counts as success. The fold-6 positive/negative holdout still
	  failed (`7 / 9` top-1 candidate-any; source-balanced retry `6 / 9`), so
	  source weighting is not a candidate. The fold-7 pure-negative smoke passed
	  only by abstention at strict thresholds (`0` approvals, `0` candidate-any
	  approvals) while grouped top-1 was still bad (`4 / 7` candidate-any).
	  Do not promote, install, distill, or live-screen Run969; keep the gate
	  fix, and collect more source-diverse on-policy same-state hard negatives.
	  Run970 created the first version of that harder source by adding a
	  `bad_red_no_win_and_lane_preserver` prefilter plus a
	  `scored_bad_red_lane_burn_no_win_with_safe_any_alternative` bucket to the
	  live substitution collector. The accepted 5p shard is tiny (`5` rows), but
	  it has the missing blind-spot shape: `33` scored bad red/no-win lane burns,
	  `10` safe lane-preserving alternatives, and `0` safe source-agreeing
	  alternatives. Holding this source out against the Run969-style
	  survival-soft verifier failed: grouped top-1 selected candidate-any in
	  `4 / 5` groups and approved `0 / 5` top choices. Turning on the
	  unsafe-margin loss was directionally better (`3 / 5` candidate-any,
	  `1 / 5` approved top-1) but still failed every live gate and worsened the
	  threshold approval mix. Do not live-screen or promote Run970. Keep the new
	  safe-any bucket, but the next cycle needs several more independent
	  Run970-style sources, ideally with a faster/cheaper proposal feature path,
	  before another verifier or raw-policy update.
	  Run971 confirmed the modern "Strawberry-style" / ReST / GRPO /
	  expert-iteration idea is the right algorithmic family but not yet a
	  solved transfer bridge. The repo has already tried the literal forms:
	  Run946 small best-of-N policy nudging, Run947 stronger target-top policy
	  nudging, Run948 scaled positive-bank nudging, and Run949 saved-slate
	  reranking. They moved offline targets, but live same-policy survival did
	  not pass because the banks were too sparse/source-sensitive and missed
	  on-policy false positives. Run971 fixed the immediate source-collection
	  issue: the hard bad-red/no-win actions were being excluded before scoring.
	  With `--no-action-set-transformer-exclude-forced-led-lane-negatives`, the
	  fixed preproposal filter collected two independent 5p hard-negative
	  shards (`2` rows from seed `20261708`, `4` rows from seed `20261709`)
	  containing bad red/no-win lane burns with safe lane-preserving
	  alternatives. Do not promote from this data directly; collect more
	  independent no-exclude Run970/971 sources, then train source-held-out
	  negative-aware proposal verification before another raw-policy RFT nudge.
	  Run972 collected two more independent no-exclude hard-negative shards
	  (`2` rows each from seeds `20261710` and `20261711`) and added a
	  default-off explicit bad-red/no-win objective to
	  `quantum_cat_proposal_veto_trainer.py`
	  (`--bad-red-no-win-negative`, `--bad-red-no-win-penalty`). This hook
	  correctly names the user's tactical rule: red off-led lane burns that do
	  not win should be rejected when lane-preserving alternatives exist. It
	  worked narrowly (`0` top-1 bad-red/no-win selections and `0.0`
	  bad-red/no-win approval rate in enabled variants), but it did not solve
	  source-held-out transfer: the best verifier passed only `2 / 5` hard
	  source holdouts, and harder sources `10` and `12` still produced
	  candidate-any top-1 selections. Do not promote or live-screen Run972.
	  Keep the explicit bad-red/no-win hook, but the next step needs richer
	  same-state candidate-any contrast labels that distinguish "safe because
	  the base is also doomed" from genuinely non-paradox alternatives.
	  Run973 added exactly that direct contrast bucket:
	  `scored_candidate_any_with_positive_safe_no_candidate_any_alternative`.
	  It is productive (`3`, `8`, and `8` row shards from seeds `20261801-03`,
	  with `82` total positive-safe-no-candidate-any scored candidates), but a
	  compact-feature verifier still failed every source-held-out gate:
	  listwise-only hard diagnostic `0 / 2`, augmented compact verifier `0 / 5`,
	  with `15 / 28` top-1 selections still candidate-any. Do not promote
	  Run973; keep the bucket as useful data infrastructure. Run974 then tested
	  full candidate/base features. That representation fixed hard holdout `12`
	  (`0 / 4` top-1 candidate-any, top-1 reward `0.981205` vs oracle
	  `0.981820`) but worsened hard holdout `10` (`5 / 5` top-1 candidate-any).
	  Do not promote Run974. Next step should focus on source-10-like
	  same-lane rank/liquidity failures: either mine more source-10-shaped
	  contrasts or build a feature family that retains rank/card/liquidity
	  information without overfitting source score/rank.
	  Runs975-977 tested that next fork. Run975 added a `full_no_source`
	  proposal feature mode that keeps candidate/base rank/card/liquidity
	  features but drops proposal source-score/probability scalars; this improved
	  source `10` versus Run974 (`3 / 5` top-1 candidate-any instead of `5 / 5`)
	  while keeping source `12` clean (`0 / 4`). It also added and smoke-mined the
	  exact same-led contrast bucket. Run976 then trained with the first two
	  same-led sources and rejected the result: source `10` regressed to `4 / 5`
	  candidate-any and source `12` to `2 / 4`, even though the new same-led
	  holdouts themselves had signal. Run977 collected six more same-led trick-0
	  rows (`4` completed plus `2` partial) and retrained only hard holdouts
	  `10/12`; source `10` still failed at `4 / 5` candidate-any and source `12`
	  only partially recovered to `1 / 4`. Do not promote, install, distill, or
	  live-screen Runs975-977. The next step should not be another identical
	  same-led augmentation pass; inspect source `10` directly and add a stable
	  public same-lane rank/liquidity feature, or build a larger source-balanced
	  same-led collector with source-family holdouts before another verifier
	  bridge.
	  Run978 added that public feature fork as
	  `full_no_source_public_slate`: full candidate/base action features with no
	  source-score/probability scalars, plus outcome-free rank/color/follow-led
	  and public liquidity percentiles inside the scored candidate slate. It was
	  directionally useful but still rejected: source `10` improved from Run977's
	  `4 / 5` top-1 candidate-any to `3 / 5`, while source `12` stayed clean at
	  `0 / 4`, but no validation gate passed and source `10` still approved too
	  many candidate-any examples. Keep the feature as infrastructure; next needs
	  source-balanced same-led/candidate-any data or a stricter direct
	  candidate-any loss before any live bridge.
	  Run979 tested the obvious no-code stricter-loss fork on top of Run978:
	  `clean_any` unsafe positives, unsafe margin `4.0`, unsafe weight `8.0`,
	  and main source balancing. It failed and regressed source `10` back to
	  `4 / 5` candidate-any top-1 with `0 / 5` approved, while source `12`
	  stayed clean at `0 / 4`. Do not promote or live-screen Run979. The next
	  useful move is not more generic unsafe-margin pressure; it is data-side
	  source-family work: collect/source-balance more source-10-like same-lane
	  candidate-any rows or build a source-family diagnostic before another
	  objective tweak.
	  Run980 built that diagnostic across the current `22` metadata sources. The
	  important result: many sources are structurally source-10-like, but source
	  `10` is much harder because its candidate slate is `80%` candidate-any and
	  only `8%` approved, while broad same-led sources often have `43-66%`
	  approved candidates. This explains why broad same-led augmentation did not
	  transfer. Next collection should require source-10-density constraints:
	  high candidate-any fraction plus sparse approved/no-any alternatives, not
	  merely same-led contrast presence.
	  Run981 implemented that source-10-density collector bucket as
	  `scored_source10_dense_candidate_any_sparse_approved`: non-red led,
	  at least `8` scored candidates, same-led candidate-any present, a sparse
	  positive-safe/no-any alternative present, candidate-any rate `>=70%`, and
	  approved/no-any rate `<=15%`. The first smoke collected `2` rows over `53`
	  rounds and `44` evaluated proposal rows. Both admitted rows matched the
	  hard regime (`88.89%` and `87.50%` candidate-any; `11.11%` and `12.50%`
	  approved/no-any). Keep Run981 as infrastructure; do not train or promote
	  from the 2-row smoke. Next step is parallel multi-shard collection using
	  this bucket, then a hard-holdout verifier retrain with source-density rows
	  treated as their own family.
	  Run982/Run983 did that first retrain and falsified the simple-data
	  hypothesis. Strict Run981 thresholds were too sparse (`0/130` accepted in
	  two follow-up workers), but a near-density bucket (`>=65%` candidate-any,
	  `<=30%` approved/no-any) collected `8` more rows, for `10` density rows
	  total. Adding those rows with source-balanced training regressed source
	  `10` to `4/5` top-1 candidate-any while keeping source `12` clean; adding
	  them without source balancing was worse (`5/5` source `10`, `1/4` source
	  `12`). The density rows are useful infrastructure, but not plug-in
	  positives. They mix same-led clean actions with responsible red-switch
	  winners, so the next lever should be a density-specific contrast/veto loss
	  that explicitly pushes rejected candidate-any actions below sparse clean
	  actions inside these hard slates.
	  Run984-Run988 tested that density-specific contrast/veto loss. The hook
	  learned the intended within-group margin, but no candidate passed the hard
	  holdout gates. Unscoped contrast could improve one source-10-facing number
	  (`40.00%` best-threshold candidate-any in Run986), but it regressed source
	  `12` top-1 from `0/4` to `1/4`. Source-scoped contrast restricted to
	  sources `22,23,24` fixed that source-12 top-1 contamination but did not
	  keep the source-10 threshold gain (`50-60%` candidate-any). Reject
	  Runs984-988; keep the default-off density contrast/source-filter hooks.
	  The next best step is not another tiny same-slate density pass, but a
	  best-of-rollouts / expert-iteration generator: sample multiple
	  continuations from hard states, rank complete trajectories by no-paradox
	  table survival plus score, then train pairwise/listwise first-action
	  preferences from winning branches versus collapse branches.
	  Run989 implemented and smoke-tested that generator as
	  `quantum_cat_best_of_rollouts_replay.py`. It writes normal replay rows
	  plus full per-action rollout metadata. Strict non-red-led survival-contrast
	  smokes succeeded for all player counts: 3p produced `4` rows from `13`
	  evaluated decisions, 4p produced `1 / 1`, and 5p produced `1 / 1`; all
	  accepted rows changed the policy target and had any-paradox spread `1.0`.
	  The examples show the richer missing signal: some fixes are red-switches
	  (`Y6 -> R6`), but others are same-led rank allocation (`Y4 -> Y1`,
	  `B4 -> B8`, `B3 -> B9`). Keep Run989 as infrastructure; do not train or
	  promote from the smoke. Next step is scaled independent 3p/4p/5p
	  best-of-rollouts banks, then table-size-specific policy/action-risk
	  training and Mac paradox gates.
- **Phase 2 - win-rate / score optimization** (where everything has died):
  naive self-play erodes phase 1; the binding constraint is credit
  assignment, not the optimizer. Levers in order: table-wide liveness aux
  targets; trajectory-level causal continuation data; oracle-filtered
  self-play generation; genuinely large-scale self-play if hardware
  permits; exact endgame labeling for tricks 5-7.

## Infrastructure requirements

- Full base game for 3/4/5 players (exists:
  `open_spiel/python/games/quantum_cat.py`).
- Visualizer so a human can play against the bot or with bot suggestions
  (exists: web UI + iOS app). Web completed-game logging exists as of Run729;
  iOS logging/export and real human-match collection are still missing.
- Log every run in `EXPERIMENTS.md` following `QUANTUM_CAT_RUN_METRICS.md`:
  goal, recipe, checkpoint paths + SHA-256, gate artifacts with seeds and
  CIs, and an explicit promote/continue/reject decision.

## Latest Direction - Run991 Best-of-Rollouts Policy Improvement

- Run990 confirmed that the LLM-style loop applies here: sample several legal
  continuations, rank the full trajectories by table survival first and score
  second, and train the base policy to make the better first actions more
  likely.
- This is not new conceptually in the repo: Runs946-947 already tried tiny 5p
  best-of-N / rejection-sampling policy nudges. They moved offline imitation
  but did not improve live same-policy survival because the replay bank was too
  narrow.
- The new Run989/Run990/Run991 path is broader and table-size-aware:
  `quantum_cat_best_of_rollouts_replay.py` now creates rollout-ranked replay
  rows for 3p, 4p, and 5p. The first Run991 screen trained `24/20/16` rows and
  failed badly; the stricter follow-up trained `48/24/16` rows and produced
  real live survival movement.
- Strict Run991 follow-up result:
  3p improved from `61.67%` to `42.78%` same-policy paradox rounds; 4p improved
  from `76.67%` to `68.75%`; 5p improved from `77.67%` to `58.33%`. This is
  positive evidence for historical table-survival policy improvement, but not
  proof of competitive placement; the former `<40%` veto is retired.
- Current decision: focus the next run on a much larger 3p strict bank first,
  with held-out validation and late forced-exit contrast. 3p is now close
  enough to justify scale-up; 4p/5p should wait for more table-size-specific
  red-switch and late-exit coverage before expensive full gates.
