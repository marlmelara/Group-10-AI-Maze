# Silent Cartographer: Hybrid RL + Phase-Aware BFS Maze Agent

A reinforcement-learning + classical-search agent that solves unknown
64×64 grid mazes containing rotating fire hazards, teleport pads,
confusion pads, and directional wind cells. Built for COSC 4368
(Fundamentals of AI, Spring 2026) by **Group 15** as a spec-compliant
implementation of the *Silent Cartographer: Maze Navigation* project.

| Maze      | Success rate | Avg turns | Death rate | Notes |
|-----------|-------------:|----------:|-----------:|-------|
| alpha     | **100 %** (5/5) | 1015.4 | 0.0030 | Train + test |
| beta      | **100 %** (5/5) | 1178.4 | 0.0090 | Test only; no training per spec |
| gamma EC  | attempted   | — | — | Reached Manhattan 25 from goal; not solved |

Both primary mazes meet or exceed the specification's **stretch goals**
(>95 % success, <500 turns on converged episodes, <1 % death rate).

## Team

**Marlon Melara** • **Ahnaf Murshid**

(Group 15 was originally rostered with five students; only the two
above contributed to this submission. The contribution statement in
`report.pdf` documents this.)

## How It Works

The agent combines three ideas:

1. **Online mapping (classical).** A trit edge map
   (`−1 unknown / 0 open / 1 wall`) is updated from every `TurnResult`.
   Rotating V-fire clusters are stored as four per-phase sets so the
   planner can reason about "cell X is a fire at phase 1 but safe at
   phase 2." Teleport source→destination pairs and confusion pads are
   recorded on first trigger.

2. **Reinforcement-learning risk shaping (RL).** A tabular
   `Q[row, col, phase]` grows on death via TD(0) and propagates to
   Manhattan neighbors with geometric decay, so the planner detours
   around *regions* rather than single cells. Between episodes, Q is
   decayed by `q_episode_decay = 0.25` to prevent stale fear from
   permanently blocking good paths.

3. **Phase-aware BFS with goal-biased frontier (search).** Each turn,
   the planner tries in order:

   1. Phase-aware BFS to the goal over known-open cells.
   2. Step across an unknown edge when the current cell is a frontier.
   3. BFS to the **frontier cell closest to the goal** (this is the
      fix that turned beta from unsolvable to 100 %).
   4. Progressively relaxed attempts if the above fail.

The critical correctness detail is that the environment checks pit
collisions using the **pre-increment** phase
`(action_counter // 5) % 4`, so the agent's BFS must use the same
convention — an off-by-one here was responsible for deaths on every
5th planned action.

## Installation

```bash
git clone <this repo>
cd "Group 15 Maze AI Project"
pip install numpy matplotlib pillow
```

Only needs the standard scientific Python stack. Tested on Python 3.12
and 3.14.

## Quick Start

```bash
# 1. (Optional) Re-parse the maze PNGs -> numpy arrays
python3 src/maze_parser.py
python3 src/visualize_parse.py

# 2. Train on alpha and test on alpha + beta (our full submission run)
python3 src/run_experiments.py \
    --episodes 5 --training-episodes 3 --max-turns 15000 \
    --mazes alpha beta --output results/metrics.json

# 3. Live demo (matplotlib animation) — for the in-class presentation
python3 src/live_demo.py --maze beta
```

## Using the Toolkit on Your Own Maze

The environment and agent are spec-agnostic: drop in a 64×64
`parsed.npz` and they will run.

```python
from environment import MazeEnvironment, Action
from agent import HybridRLAgent

env   = MazeEnvironment("beta")     # loads data/beta/parsed.npz
agent = HybridRLAgent()
agent.set_start_goal(env.start_pos, env.goal_pos)

env.reset()
agent.reset_episode()
agent.pos = env.start_pos

last = None
while True:
    actions = agent.plan_turn(last)         # List[Action], 1–5 entries
    last = env.step(actions)                # -> TurnResult
    if last.is_goal_reached:
        break

print(env.get_episode_stats())
```

The `Action` enum, `TurnResult` dataclass, and `MazeEnvironment`
interface are exactly as defined in the course specification §6.

## Repository Layout

```
src/
  maze_parser.py       PNG → numpy wall + hazard arrays
  environment.py       Spec-compliant MazeEnvironment
  agent.py             HybridRLAgent (RL risk + phase-aware BFS)
  metrics.py           EpisodeRecord, AggregateMetrics, run_episode
  visualizer.py        Static maps, trajectories, learning curves, heatmaps
  run_experiments.py   Top-level train + eval driver
  live_demo.py         Matplotlib live animation
  visualize_parse.py   Parser sanity check
data/
  alpha/ beta/ gamma/  MAZE_0.png walls • MAZE_1.png hazards •
                       MAZE_2.png reference path • parsed.npz
figures/               Auto-generated plots (maps, trajectories, heatmaps)
results/
  metrics.json         5-episode evaluation numbers for both mazes
  experiment.log       Live experiment stdout
report.pdf             8-page final report (method, results, AI disclosure)
presentation.pdf       In-class presentation slides
```

## Reproducibility

All runs are deterministic given a fixed Python version and CPU: the
BFS ties break by explicit priority, and there is no randomness in the
planner. A single end-to-end evaluation on alpha+beta takes under
10 minutes on a 2021 Apple Silicon laptop.

## Limitations and Known Issues

- Maze-gamma's goal is only reachable via a specific teleport and
  through wind-bounded corridors. Our current frontier planner finds
  teleports only by accidentally walking over a teleport source; an
  explicit "teleport source prospector" would be needed to solve
  gamma reliably. Left as future work.
- The agent's wall-clock per turn grows with the number of known fire
  cells because the phase-aware BFS has a larger effective state
  space. A 40 ms time cap keeps each call bounded.

## License

MIT. Feel free to fork, learn from, or extend. Cite this repository if
you use the idea of goal-biased frontier targeting.

## AI Usage Disclosure

Developed with Anthropic Claude Code (Opus 4.7, 1M-context). All
representative prompts, iterations, and the team's evaluation of the
AI's work are documented in **report.pdf** §"AI Usage".
