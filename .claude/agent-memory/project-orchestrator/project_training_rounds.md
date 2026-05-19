---
name: project-training-rounds
description: Round 1-4 cumulative training infrastructure: what tokens each round introduces, where scripts live, and how the data is structured
metadata:
  type: project
---

Round structure for OMR model training (cumulative vocabulary expansion):

- Round 1: basic notes (whole–16th), rests, 5 key sigs (C/G/D/F/Bb), 3 time sigs (4/4,3/4,2/4), no accidentals/dynamics. Data: data/train/ (100 samples, .png + .musicxml + .json)
- Round 2: adds all accidentals, all key/time sigs, chords, dynamics, hairpins, articulations, ornaments, fermata, slur, tuplets, ottava, trill ext, repeat barlines. Data: data/Round2/ (10 samples currently; generate_dataset.py outputs this format)
- Round 3: adds grace notes, tremolos, pedal marks, navigation marks (coda/segno/da capo), breath marks, multi-measure rests. Data dir: data/Round3/ (not yet generated as of 2026-05-19)
- Round 4: adds complex tuplets (5/7-let), tempo text, expression text, volta brackets, chord symbols, arpeggios. Data dir: data/Round4/ (not yet generated)

Key files created 2026-05-19:
- data/round_tokens/round{1..4}_tokens.json — per-round token definitions (new_tokens + generation_config)
- data/generate_random_scores.py — round-aware wrapper over omr/data_gen/generate_dataset.py; monkey-patches probabilities per round
- scripts/evaluate_round_accuracy.py — loads .pt checkpoint, runs greedy decode on test images, reports TER/NoteAcc/per-token recall/confusion matrix

**Why:** Training rounds allow progressive curriculum learning; each round fine-tunes from the previous checkpoint (load_checkpoint_with_vocab_expansion in train.py handles vocab growth).

**How to apply:** When user asks about training data or evaluation, reference these files. Round 2 generation uses omr/data_gen/generate_dataset.py directly; Rounds 1/3/4 use data/generate_random_scores.py with --round flag.

[[project_webdemo_state]]
