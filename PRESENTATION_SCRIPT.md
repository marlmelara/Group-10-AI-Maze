# Presentation Script — Silent Cartographer (Group 15)

Direct, blunt, no wasted words. One paragraph per slide. Read or
paraphrase; pauses are between slides.

---

## Slide 1 — Title

"Silent Cartographer. Hybrid reinforcement learning plus A-star
agent. COSC 4368 final submission, Group 15. Marlon Melara and
Ahnaf Murshid."

---

## Slide 2 — Task Recap

"Sixty-four by sixty-four grid maze. Agent has no sensors. Learns
only by bumping into walls and dying on fires. Hazards: rotating
V-shaped fire clusters that turn ninety degrees every five actions,
teleport pads, and confusion pads. Each turn we submit one to five
actions. We train on maze-alpha, then test on maze-beta without any
retraining. We are reporting success rate, average turns, path
length, and death rate, plus four bonus metrics."

---

## Slide 3 — Approach: Why Hybrid RL + A*

"We picked reinforcement learning because the reward signal is
delayed and sparse — you only discover a cell is a fire after you
die on it. TD-zero learning spreads that danger to neighbouring
cells so the agent detours whole regions, not one cell. We pair RL
with classical A-star because once an edge is observed, its state
is deterministic forever — classical search exploits that
immediately, while pure RL would need hundreds of episodes to learn
the same thing. Our hyperparameters transfer from alpha to beta;
the per-cell state does not."

---

## Slide 4 — Agent Architecture

"One loop. Environment returns a TurnResult. We integrate it —
walls, fires by phase, teleport mappings, confusion pads, Q-table.
We run the planner. We submit one to five actions. Repeat. The RL
layer is a sixty-four by sixty-four by four Q-table, decayed between
episodes. The planning layer is phase-aware breadth-first search
with a goal-biased frontier selector."

---

## Slide 5 — Rotating Fire Hazards

"Fires come in V-clusters. They rotate ninety degrees clockwise
every five cumulative actions, pivoting on the tip of the V. We
pre-compute all four rotation states at environment construction.
The agent learns fires by dying. Conservatively, a cell that has
ever been a fire is treated as dangerous across all phases by
default — relaxed only when we need to squeeze through."

---

## Slide 6 — Mapping and Q-Learning

"Per-maze state: a trit edge map — unknown, open, or wall — plus
phase-indexed fire sets, a teleport map, and the Q-table. On death
at cell s in phase phi, Q updates by TD-zero with the death penalty
and propagates to Manhattan neighbours with geometric decay. What
transfers between mazes is the RiskConfig — learning rate, decay
factor, penalty scale. Per-cell values reset."

---

## Slide 7 — Planner: Weighted A*

"Edge cost into cell n at phase phi: base one, plus a fire penalty
if n is a known fire, plus a confusion penalty, plus w-Q times the
learned Q value, plus a visit-count penalty. Two passes — strict
first, blocking every known-ever-fire. Relaxed second, only blocking
fires at the current arrival phase. Frontier fallback when the goal
is unreachable. Batching: we submit actions up to five as long as
each next edge is known-open and the landing cell is known-safe at
its arrival phase."

---

## Slide 8 — Live Transfer Demo: maze-beta

"Beta has an entirely different layout. Two teleports, five
confusion pads, fifty-four fire cells, and — critically — a narrow
single-entry chute to the goal. The only non-walled approach to
zero-thirty-one is from zero-thirty-two, which is only reachable
from one-thirty-two, which is only reachable from two-thirty-two.
No training on beta per spec. Same hyperparameters as alpha. The
goal-biased frontier BFS is what lets the agent find that chute."

---

## Slide 9 — Results

"Five evaluation episodes per maze. Alpha: one hundred percent
success, average one thousand fifteen turns, death rate zero point
zero zero three. Episodes two through five converge to the
eighty-seven-turn BFS-optimal path with zero deaths. Beta: one
hundred percent success, average one thousand one hundred
seventy-eight turns, death rate zero point zero zero nine. Episodes
three through five converge to an eighty-turn path with zero deaths.
Both mazes exceed the specification's stretch goals."

---

## Slide 10 — Bonus Opportunities

"Three bonus targets. One: gamma attempt — we ran a five-episode
evaluation, the agent discovered one of the two teleports, reached
Manhattan twenty-five from the goal, but did not solve within the
turn budget. Five points for attempting. Two: novel visualization —
a six-panel solution dashboard unifies maps, trajectories,
convergence, and metrics for both mazes in one image. Up to five
points. Three: open-source toolkit release — MIT license, complete
README with installation and a public API example, spec-agnostic
code. Up to five points. Spec caps total bonus at ten."

---

## Slide 11 — What Turned Beta from "Unsolvable" to 100%

"Three targeted fixes. One: off-by-one phase fix in BFS. The
environment checks pits at the pre-increment phase. The original
planner used the post-increment phase, so every fifth planned step
landed on a fire. Two: goal-biased frontier targeting. When no known
path to the goal exists, we enumerate known-open cells that have an
unknown edge, sort them by Manhattan distance to the goal, and BFS
to the nearest one. Without this, the nearest-frontier heuristic
kept pulling the agent into a local pocket. Three: A-star cycle
guard. Negative-cost teleport bonuses could produce cycles in the
parent map; the reconstruction loop spun forever. Fixed with
edge-cost flooring and a seen-set guard."

---

## Slide 12 — Conclusion & Lessons Learned

"Hybrid RL plus phase-aware BFS plus frontier planning reaches one
hundred percent success on both alpha and beta, five out of five
each, no training on beta. Episode one is exploratory. Episodes two
through five run the BFS-optimal path in eighty to eighty-seven
turns with zero deaths. Beta's narrow single-entry chute would have
defeated a pure nearest-frontier explorer; goal-biased frontier
targeting was the key. AI usage is fully disclosed in report dot
PDF — Claude Code, Opus four-point-seven, one-million-token context.
Thanks. Questions."

---

## Live Demo (if asked)

Terminal command:
```
python3 src/live_demo.py --maze beta
```

Point out in the animation:
- Agent starts at (63, 32), heads north
- Blue dots are teleport sources; green circle is start; yellow circle is goal
- Around turn 3000 the agent hits the (0, 30) wall and realises the goal needs a different approach
- Watch the green frontier trail work rightward to find the (2, 32) chute
- Agent crosses the chute and reaches the goal

---

## Q&A crib sheet

- **"Why RL over pure search?"** — Delayed sparse rewards, and our hyperparameters transfer between mazes even when the map resets.
- **"Why not pure search?"** — A fresh maze has no known edges; search can't plan until exploration happens.
- **"Why doesn't gamma solve?"** — The goal is only reachable through a specific teleport, and our planner finds teleports only by accidentally walking over a source. An explicit "teleport prospector" would fix it.
- **"How fast is each plan?"** — Alpha about three milliseconds, beta about fifty milliseconds — the phase-aware BFS state space grows with the fire count.
- **"What if the goal were on an island?"** — Frontier search keeps exploring until every reachable region is mapped, then declares failure.
- **"AI usage?"** — Claude Code was used for scaffolding, debugging stack traces, and iterating on Q-decay. All algorithmic decisions — fire marking, BFS layering, the three key bug fixes — were made by us and documented in the report.
