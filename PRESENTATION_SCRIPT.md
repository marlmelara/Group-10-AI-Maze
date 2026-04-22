# Presentation Script — Silent Cartographer (Group 15)

Written in complete sentences but still direct. Read or paraphrase.
Pauses are between slides.

**How to use the plain-language notes.** Every technical term is
followed by an *optional* "plain English" line in italics. If the
audience is with you, skip it. If you see blank faces, drop it in
naturally before moving on.

---

## Slide 1 — Title

"Good morning. This is Silent Cartographer, our final submission for
COSC 4368. We built a hybrid reinforcement-learning and A-star agent
that solves unknown maze environments. I'm Marlon Melara, and my
teammate is Ahnaf Murshid."

> *Plain-English fallback for "hybrid RL + A-star":*
> "In other words, we combined a learning approach that gets better
> through trial and error with a classical path-finding algorithm."

---

## Slide 2 — Task Recap

"The task is to navigate an unknown sixty-four by sixty-four grid
maze. The agent has no sensors, which means it learns about walls by
bumping into them and learns about fires by dying on them. The
hazards include rotating V-shaped fire clusters that turn ninety
degrees every five actions, teleport pads, and confusion pads. Each
turn the agent submits between one and five actions, and the
environment returns a TurnResult describing what happened. We train
on maze-alpha and then test the same agent on maze-beta without any
retraining. The required metrics are success rate, average turns,
path length, and death rate, and there are four bonus metrics we
also report."

> *Plain-English fallback for "TurnResult":*
> "Basically, a summary object telling the agent what just happened —
> did it hit a wall, did it die, did it reach the goal."

> *Plain-English fallback for "retraining":*
> "We're not allowed to let the agent practice on the test maze
> beforehand. It gets dropped in cold."

---

## Slide 3 — Approach: Why Hybrid RL + A*

"We chose reinforcement learning because the reward signal is both
delayed and sparse. You only discover a cell is a fire after you
have already died on it, so TD-zero learning is useful for spreading
that danger to neighboring cells. The agent ends up detouring whole
regions instead of a single cell. We paired reinforcement learning
with classical A-star search because once an edge has been observed,
its state is deterministic forever. Classical search can exploit
that structure immediately, while a pure reinforcement-learning
policy would need hundreds of episodes to learn the same paths. The
learned hyperparameters transfer from alpha to beta, but the
per-cell state is reset between mazes because that's what the spec
requires."

> *Plain-English fallback for "reward signal is delayed and sparse":*
> "You don't find out a cell is bad until you've already stepped on
> it and died. There's no warning beforehand."

> *Plain-English fallback for "TD-zero learning":*
> "A reinforcement learning rule that lets danger information spread
> from the cell you died on to the cells around it, so the agent
> avoids the whole neighborhood next time."

> *Plain-English fallback for "A-star search":*
> "A classic pathfinding algorithm — think Google Maps — that finds
> the shortest route through a known map."

> *Plain-English fallback for "hyperparameters":*
> "The tuning knobs on the learning algorithm: how fast it learns,
> how quickly it forgets, how much it penalizes danger."

---

## Slide 4 — Agent Architecture

"The agent runs a simple loop. The environment returns a TurnResult,
we integrate that into our world model, and then we plan. Integration
updates the wall map, the per-phase fire sets, the teleport mapping,
the confusion pad set, and the Q-table. Planning is a phase-aware
breadth-first search with a goal-biased frontier selector. The
planner returns between one and five actions, and that list is
submitted to the environment. The RL layer is a sixty-four by
sixty-four by four Q-table that decays between episodes so stale
fear doesn't block otherwise-good paths."

> *Plain-English fallback for "Q-table":*
> "A big lookup table where each cell stores a learned danger score.
> Higher score means 'I've died here, avoid it.'"

> *Plain-English fallback for "breadth-first search":*
> "A classic method for finding the shortest route through a known
> map by exploring outward one step at a time."

> *Plain-English fallback for "phase-aware":*
> "The fires rotate on a schedule, so the planner doesn't just check
> where fires are now — it checks where each fire will be when the
> agent actually arrives at each cell along the path."

> *Plain-English fallback for "decays between episodes":*
> "Between runs we reduce the Q-table values so the agent isn't
> permanently scared of cells it died on early while still being
> bad at the game."

---

## Slide 5 — Rotating Fire Hazards

"The fires come in V-shaped clusters. Every five cumulative actions
they rotate ninety degrees clockwise, pivoting around the tip of the
V. We pre-compute all four rotation states at environment
construction time so we never have to rotate them on the fly. The
agent learns about a fire by dying on it, so we mark the exact phase
of each death. To be conservative, we also treat any cell that has
ever been a fire as dangerous across all phases by default, and only
relax that when we need to squeeze through a timed corridor."

> *Plain-English fallback for "four rotation states":*
> "Each V-cluster has exactly four possible positions as it spins,
> and the environment cycles through them every five actions. We
> pre-compute all four so it's instant to look up."

---

## Slide 6 — Mapping and Q-Learning

"Each maze gets its own world model. We maintain a trit-valued edge
map where minus one is unknown, zero is open, and one is wall. We
keep four per-phase fire sets, a teleport source-to-destination
mapping, a confusion pad set, and the Q-table. When the agent dies
at cell s in phase phi, the Q-table is updated by TD-zero using the
death penalty, and the signal propagates to Manhattan neighbors with
a geometric decay. What actually transfers between mazes is the
RiskConfig, which holds the learning rate, the decay factor, and the
penalty scale. The per-cell Q-values reset."

> *Plain-English fallback for "trit-valued edge map":*
> "For every possible step between adjacent cells, we store one of
> three values: we haven't tried it, it's open, or it's a wall."

> *Plain-English fallback for "Manhattan neighbors":*
> "Manhattan distance is just taxicab distance — how many up-down
> and left-right steps between two cells, ignoring diagonals."

> *Plain-English fallback for "geometric decay":*
> "The further you get from the cell where the agent died, the
> smaller the danger score. It fades with distance."

> *Plain-English fallback for "RiskConfig":*
> "Our name for the bundle of tuning knobs that describe how the
> agent reacts to danger. That bundle carries over between mazes."

---

## Slide 7 — Planner: Weighted A*

"The edge cost into cell n at phase phi has five terms. The base
cost is one, and then we add a fire penalty if the cell is a known
fire at the arrival phase, a confusion penalty if it's a known
confusion pad, the learned Q-value scaled by its weight, and a
visit-count penalty to prevent oscillation. The planner runs in two
passes. The strict pass blocks every cell that has ever been a fire
at any phase, and the relaxed pass only blocks fires at the current
arrival phase. If both passes fail to reach the goal, we fall back
to frontier exploration. When submitting actions we batch greedily.
We keep appending actions to the batch as long as each next edge is
known-open and the landing cell is known-safe at its arrival phase,
and we stop at the first uncertain step so we can observe its
outcome."

> *Plain-English fallback for "edge cost":*
> "Basically, how 'expensive' it is to step into a particular cell.
> The planner prefers the path with the lowest total cost."

> *Plain-English fallback for "visit-count penalty to prevent
> oscillation":*
> "If the agent visits the same cell a lot, we start charging extra
> for going back there. That stops it from bouncing forever between
> two cells."

> *Plain-English fallback for "frontier exploration":*
> "If the planner can't find a safe path to the goal, it heads for
> the edge of the known map to discover new territory instead."

---

## Slide 8 — Working Demo: maze-beta (learning → optimal)

"Maze-beta has a completely different layout from alpha. It has two
teleport pads, five confusion pads, fifty-four fire cells across the
V-clusters, and most importantly a narrow approach corridor to the
goal. The goal cell is at row zero, column thirty-one. Three of its
four sides are walls — so there's only one open neighbor it can be
stepped into from, which is the cell to its right. And to get to
that neighbor, the agent has to come up through a four-cell corridor
from below. We never train on beta — same hyperparameters as alpha.
The goal-biased frontier BFS is what lets the agent discover that
corridor.

We have two GIFs on this slide, one for each phase of the agent's
behavior on beta.

On the left is episode one, the reinforcement-learning phase. The
agent starts blind — it knows nothing about the maze layout or where
the fires are. It explores by trying to move, bumps into walls,
steps on fires and dies, respawns at the start, and gradually builds
an internal map. You can watch it discover larger and larger regions
of the maze. It dies forty-nine times and takes five thousand two
hundred turns on this first run, but by the end of the episode it
has found the goal.

On the right is a later episode, after the map has been learned.
Now the classical search layer takes over — the phase-aware
breadth-first search plans the optimal path on the learned map, and
the agent walks it cleanly in about eighty turns with zero deaths.
This is the converged behavior, and it holds steady through
episodes three, four, and five. Together the two GIFs show the
hybrid in action — reinforcement learning to build knowledge, and
classical search to exploit it."

> *Plain-English fallback for "approach corridor":*
> "The goal cell has walls on three of its four sides, so there's
> only one neighbor cell you can step in from. The agent has to
> find that specific neighbor, and the cells leading up to it, by
> exploration."

> *Plain-English fallback for "goal-biased frontier BFS":*
> "When the agent can't find a safe path to the goal, instead of
> picking the nearest unexplored spot, we pick the unexplored spot
> closest to the goal. That nudges exploration in the right
> direction."

> *Plain-English fallback for "converged":*
> "By the third episode the agent has stopped exploring and just
> replays the shortest path it found — same number of turns, same
> route, every time."

*(Left GIF: `figures/beta_demo_exploration.gif` — Episode 1, 5204
turns, RL exploration.)*
*(Right GIF: `figures/beta_demo_converged.gif` — Episode 3+, ~80
turns, BFS-optimal converged path.)*

---

## Slide 9 — Results

"Here are the numbers from a five-episode evaluation on each maze.
On alpha the agent succeeded every single time for a hundred-percent
success rate, with an average of one thousand fifteen turns and a
death rate of zero point zero zero three. Episodes two through five
all converge to the eighty-seven-turn BFS-optimal path with zero
deaths each. On beta the agent also succeeds every single time —
again one hundred percent — with an average of one thousand one
hundred seventy-eight turns and a death rate of zero point zero zero
nine. Episodes three through five converge to an eighty-turn path
with zero deaths. Both mazes comfortably beat the specification's
expected and stretch-goal benchmarks."

> *Plain-English fallback for "converge":*
> "By episode three the agent has stopped exploring and just replays
> the shortest path it discovered. Same number of turns, same path,
> every time."

---

## Slide 10 — Bonus Opportunities

"We targeted three of the bonus opportunities. The first is the
maze-gamma attempt. We ran a full evaluation on gamma, and the agent
discovered one of the two teleport pads and reached a Manhattan
distance of twenty-five from the goal, but it didn't solve within
the turn budget. The spec awards five points just for attempting.
The second is novel visualization. We built a six-panel solution
dashboard that unifies the static maps, agent trajectories,
convergence plots, and final metrics for both mazes into one image,
and the script that generates it is included in the source tree.
That's worth up to five points. The third is open-source toolkit
release. The repository ships with an MIT license, a complete README
including installation steps and a public-API example, and the code
is spec-agnostic so any parsed sixty-four by sixty-four maze can
plug in. That's also worth up to five points. The specification caps
total bonus at ten."

> *Plain-English fallback for "spec-agnostic":*
> "Our code doesn't assume any specific maze. Drop in any
> sixty-four by sixty-four maze file and it runs."

---

## Slide 11 — What Turned Beta from "Unsolvable" to 100%

"Three targeted fixes are what took beta from unsolvable to a clean
one-hundred percent. The first was an off-by-one bug in the phase
calculation. The environment checks pit collisions using the
pre-increment phase, but our original planner was using the
post-increment phase, which meant every fifth planned step landed
directly on a fire. The second was goal-biased frontier targeting.
When no known-safe path to the goal existed, we started enumerating
all known-open cells that still had at least one unknown edge,
sorting them by Manhattan distance to the goal, and running BFS to
the closest one. Without that change, the agent's nearest-frontier
heuristic kept pulling it into a local pocket on the wrong side of
the maze. The third fix was an A-star cycle guard. The negative-cost
teleport bonus could produce cycles in the parent map during path
reconstruction, and the reconstruction loop was spinning forever.
We fixed it by flooring edge costs at zero-point-zero-five and
adding a seen-set guard in path recovery."

> *Plain-English fallback for "off-by-one bug":*
> "A subtle timing mistake where we were checking the fire position
> one action too late. Every fifth action walked right into a fire
> because of it."

> *Plain-English fallback for "negative-cost teleport bonus":*
> "We gave the planner a discount for using teleports that shortcut
> toward the goal. But the discount was too generous — it made some
> routes feel shorter than they actually were — and that created
> infinite loops in the path reconstruction."

> *Plain-English fallback for "seen-set guard":*
> "We now track which cells we've already visited while rebuilding
> the path, and stop if we'd revisit one. Infinite loops are
> impossible."

---

## Slide 12 — Conclusion & Lessons Learned

"To wrap up, our hybrid agent reaches one-hundred percent success on
both alpha and beta across five evaluation episodes each, with no
training on beta. Episode one is the exploration phase on either
maze, and from episode two onward the agent runs the BFS-optimal
path in eighty to eighty-seven turns with zero deaths. The biggest
lesson was that a purely nearest-frontier explorer would have been
defeated by beta's goal cell, which is walled off on three sides
and only approachable through one specific corridor. Goal-biased
frontier targeting is what made the transfer work. Our AI usage is
fully disclosed in the report — we used Claude Code with the Opus
four-point-seven one-million-context model. Thank you, and we're
happy to take questions."

---

## Demo slide — what's shown

The slide displays TWO GIFs side-by-side:

**Left (Episode 1 — RL Exploration):**
`figures/beta_demo_exploration.gif` — cold-start run. Agent knows
nothing about the map. 5204 turns, 49 deaths, eventually reaches
the goal. This is the reinforcement-learning phase — the agent
literally learning the maze.

**Right (Episode 3+ — BFS-Optimal Converged):**
`figures/beta_demo_converged.gif` — after the map is learned.
~80 turns, 0 deaths, clean optimal path. This is the classical-
search payoff — the phase-aware BFS planning on the learned map.

Final-frame still fallbacks (shown automatically if GIFs don't
play): `beta_demo_exploration_final.png` and
`beta_demo_converged_final.png`.

## Q&A Crib Sheet

- **Why reinforcement learning over pure classical search?**
  Because the fire layout is not knowable in advance, the reward
  signal is delayed and sparse, and the RiskConfig hyperparameters
  we learn on alpha actually transfer to beta even when the map
  resets.

- **Why not pure classical search?**
  Because a fresh maze has no known edges, so search has nothing to
  plan over until exploration fills in the edge map.

- **Why doesn't gamma solve?**
  Gamma's goal is only reachable through a specific teleport, and
  our current planner only discovers teleports by accidentally
  walking over a source. Adding an explicit teleport-source
  prospector would fix it; we ran out of time before the submission
  deadline.

- **How fast is each plan?**
  About three milliseconds per turn on alpha, and about fifty
  milliseconds per turn on beta. Beta has more learned fire cells,
  so the phase-aware BFS has a larger effective state space.

- **What if the goal were on an isolated island?**
  Frontier search would keep exploring until every reachable region
  of the map was discovered, then declare failure and terminate the
  episode.

- **AI usage?**
  Claude Code was used for scaffolding, debugging stack traces, and
  iterating on the Q-decay schedule. Every algorithmic decision —
  the fire-marking strategy, the BFS layering, and the three key
  bug fixes we just discussed — was made by us and is documented in
  the report.
