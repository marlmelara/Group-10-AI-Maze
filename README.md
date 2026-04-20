# COSC 4368 Maze Solver - Group 15 FINAL

Hybrid reinforcement-learning + classical-search agent for an unknown 64x64
maze with rotating fire hazards, teleports, confusion pads, and (on
maze-gamma) directional wind cells.

## Team
Marlon Melara, Ahnaf Murshid, Israel Mummay, Brittany Picazo, Osinachi Enemuoh.

## Code layout

```
src/
    maze_parser.py        # Parses MAZE_0.png / MAZE_1.png into walls + hazards
    environment.py        # Spec-compliant MazeEnvironment (Action / TurnResult)
    agent.py              # HybridRLAgent: Q-learning risk + A* search
    metrics.py            # EpisodeRecord, AggregateMetrics, run_episode
    visualizer.py         # Static maps, trajectories, learning curves, heatmaps
    run_experiments.py    # Top-level train-on-alpha, test-on-{alpha,beta,gamma}
    live_demo.py          # Matplotlib live animation (for in-class demo)
    visualize_parse.py    # Parser sanity check
data/
    alpha/  beta/  gamma/  # walls.png, hazards.png, reference_path.png, parsed.npz
figures/                   # Generated plots
results/                   # metrics.json from run_experiments.py
archive/                   # Prior check-in artifacts
report.tex  presentation.tex
```

## How to run

```bash
# (1) Parse maze PNGs into numpy arrays (already committed under data/*.npz)
python src/maze_parser.py
python src/visualize_parse.py

# (2) Run the full experiment battery: train on alpha, test alpha/beta/gamma
python src/run_experiments.py --episodes 5 --training-episodes 10

# (3) Live demo (uses matplotlib animation)
python src/live_demo.py --maze beta
```

## Method summary

The agent is a two-layer hybrid:
- **Reinforcement-learning risk model**: tabular Q over (cell, rotation-phase),
  updated by TD(0) whenever the agent dies.  Between episodes the Q-table
  is decayed so fear from early exploration doesn't permanently block
  good paths.  Hyperparameters (decay, TD rates, penalty scale) transfer
  between mazes.
- **Classical search planner**: each turn we run A* with a Manhattan
  heuristic, using the learned risk signal plus visit counts.  Unknown
  edges are treated as traversable with a small cost so the planner
  actively explores while respecting known hazards.

Full derivation, metrics tables, ablations, and AI-tool disclosure are in
`report.pdf`.

## AI usage disclosure

This project was developed with AI assistance via Anthropic's Claude Code
(Opus 4.7, 1M context).  Prompts, generated code, and how we iterated on
the AI's suggestions are documented in `report.pdf`, section "AI Usage".
