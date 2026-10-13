# Curated bidding validation

This suite validates only the acting player's prediction using their private
hand and public match state. Acceptance is set-valued: when several larger bids
are reasonable, every such bid is accepted. The exact fixture hands are locked
evaluation data and must never enter model training replay.

The root families cover five-player nines, four-player eights, and three-player
sixes. A hand with several maximum-rank cards accepts bids 2, 3, or 4; a hand
without any of the top three ranks accepts only bid 1. An isolated highest card
accepts either bid 1 or bid 2. Four second-highest cards are also recognized as
a strong five-player hand even without a nine. Every position is repeated in
every possible acting seat.

Run schema-only validation with:

    PYTHONPATH=. .venv/bin/python quantum_cat_curated_bidding_validation.py

Run an actual candidate, rather than the non-gating schema-only mode, with:

    PYTHONPATH=. .venv/bin/python quantum_cat_curated_bidding_validation.py \
      --players 5 --candidate '/absolute/checkpoint.pt:policy?neural_prediction_mode=model'

Curated bidding positions are a regression preflight, never a replacement for
seat-rotated official full-match placement and acting-player paradox safety.
