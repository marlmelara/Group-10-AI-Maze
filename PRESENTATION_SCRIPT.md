# Presentation Script — Silent Cartographer (Group 15)

Read straight through. Every technical term is explained in the sentence where it first appears, so you don't need to stop or insert anything.

---

## Slide 1

Good morning. This is Silent Cartographer, Group 15's final project for COSC 4368, presented by Marlon Melara and Ahnaf Murshid. We built an agent that navigates an unknown sixty-four by sixty-four grid maze using a combination of reinforcement learning, which is a trial-and-error approach where the agent learns from the consequences of its actions, and A-star search, which is a classical pathfinding algorithm that finds the shortest route through a map it already knows. Over the next twelve slides we'll walk through how the agent works, how it performs on both test mazes, and what we learned along the way.

---

## Slide 2

The task is to navigate an unknown sixty-four by sixty-four grid maze. The agent has no sensors, so it learns about walls only by bumping into them and learns about fires only by dying on them. The hazards are rotating V-shaped fire clusters that turn ninety degrees every five actions, teleport pads that move the agent to a fixed destination, and confusion pads that invert the agent's controls for a turn. Each turn the agent submits between one and five actions, and the environment returns a summary called a TurnResult that tells the agent what happened, including whether it hit a wall, died, or reached the goal. The rule is that we train on maze-alpha and then test the same agent on maze-beta without retraining, meaning the agent is dropped cold into the test maze and has to figure it out live. The four required metrics are success rate, average turns to solution, average path length, and death rate, and we also report four bonus metrics.

---

## Slide 3

We chose this hybrid for three reasons. First, reinforcement learning handles delayed and sparse feedback, which is exactly our situation because the agent only discovers a cell is a fire after it has already died on it. A technique called TD-zero, which stands for temporal difference learning, lets that danger information spread from the cell of death to nearby cells, so the agent avoids the whole dangerous region rather than just one specific cell. Second, once the agent has observed an edge between two cells, that edge is deterministic forever, and classical A-star search can exploit that structure immediately, while a pure reinforcement-learning policy would need hundreds of episodes to learn the same paths. Third, the learned hyperparameters, which are the tuning knobs that control how the agent reacts to danger, transfer from alpha to beta even though the map itself resets. So we get the trial-and-error learning of reinforcement learning and the optimal-path-finding of classical search working together.

---

## Slide 4

The agent runs one simple loop. The environment returns a TurnResult, the agent updates its internal world model based on what just happened, the planner produces the next one to five actions, and those are submitted back to the environment. The integration step updates the wall map, the per-phase fire sets that track which cells are fires at which rotation, the teleport mapping, the confusion pad set, and the Q-table, which is a sixty-four by sixty-four by four lookup table where each cell at each rotation phase stores a learned danger score. A higher score means the agent has died there before and wants to avoid it. Between episodes we decay the Q-table, meaning we reduce every value by a fixed factor, so the agent isn't permanently afraid of cells where it died early while it was still bad at the game. The planner itself is a phase-aware breadth-first search, which is a classical method that finds the shortest route through a known map by exploring outward one step at a time, plus a goal-biased frontier selector that we'll explain later.

---

## Slide 5

The fires come in V-shaped clusters, and every five cumulative actions they rotate ninety degrees clockwise around the tip of the V. So each cluster has exactly four possible configurations, and the environment cycles through all four every twenty actions. We pre-compute all four rotation states when the environment is built, so the look-up is instant during planning. The agent learns about a fire by dying on it, and when that happens we mark the exact rotation phase at which the death occurred. To be conservative by default, we also treat any cell that has ever been a fire as dangerous at every phase, and we only relax that rule when the agent absolutely needs to squeeze through a timed corridor. The figure on the right shows maze-beta's parsed hazard map in red.

---

## Slide 6

Every maze gets its own internal world model. We keep a trit-valued edge map, which just means every possible step between adjacent cells is one of three values: unknown, open, or wall. We keep four per-phase fire sets for the rotating clusters, a teleport source-to-destination mapping, a confusion pad set, and the Q-table. When the agent dies at some cell at some phase, we apply a TD-zero update, meaning we raise the Q-value at that cell and also propagate the danger signal to the Manhattan neighbors, which is a taxicab-distance measure ignoring diagonals, with a geometric decay, so cells further away get a smaller share of the danger. What actually transfers between mazes is a bundle we call the RiskConfig, which holds the learning rate, the decay factor, and the penalty scale. The per-cell Q-values reset between mazes because each maze has a different layout.

---

## Slide 7

The planner treats each possible step as having an edge cost, which is how expensive it is to move into that cell. The formula is one base point, plus a fire penalty if the cell is a known fire at the arrival phase, plus a confusion penalty if it's a confusion pad, plus the learned Q-value scaled by its weight, plus a visit-count penalty that charges extra for revisiting the same cell to stop the agent from bouncing back and forth. The planner runs in two passes. The strict pass blocks every cell that has ever been a fire at any phase, and the relaxed pass only blocks fires at the actual arrival phase. If both passes fail to find the goal, we fall back to frontier exploration, which means we head for the edge of the known map to discover new territory. When submitting actions we batch greedily: we keep appending moves up to five as long as each next step is known to be safe at its arrival phase, and we stop at the first uncertain step so we can observe its outcome before planning further.

---

## Slide 8

This slide has two animations, both on maze-beta. On the left is episode one, the reinforcement-learning phase. The agent starts knowing nothing about the maze, so it bumps around, steps on fires, dies and respawns at the start, and gradually builds its internal map. By the end of the episode it has explored enough territory to find the goal, but it took five thousand two hundred turns and forty-nine deaths to get there. On the right is a later episode, after the map has been learned. Now the classical search layer takes over, the phase-aware breadth-first search plans the optimal path directly, and the agent walks it cleanly in eighty turns with zero deaths. We call this converged behavior, meaning the agent has stopped exploring and just replays the shortest path it found, and it holds steady through episodes three, four, and five. Together the two animations are the whole hybrid in action: reinforcement learning to build knowledge, and classical search to exploit it.

---

## Slide 9

On both mazes the agent succeeds every single time across five evaluation episodes, for a one-hundred percent success rate. On alpha, the average across all five episodes is one thousand fifteen turns with a death rate of zero point zero zero three, and episodes two through five all converge to an eighty-seven-turn optimal path with zero deaths each. On beta, where we are not allowed to train, the average is one thousand one hundred seventy-eight turns with a death rate of zero point zero zero nine, and episodes three through five converge to an eighty-turn path with zero deaths. The specification lists an expected-performance bar and a stretch-goal bar, and we comfortably beat the stretch goals on both mazes.

---

## Slide 10

We targeted three of the bonus opportunities the specification lists. First, we ran a five-episode evaluation on maze-gamma, the extra-credit maze with wind hazards. The agent discovered one of the two teleport pads and reached a Manhattan distance of twenty-five from the goal, but it did not solve within the turn budget. The specification awards five points just for attempting gamma. Second, we built a novel visualization, which is the six-panel dashboard on the right side of this slide. It unifies the static maps, agent trajectories, convergence plots, and final metrics for both mazes into one image. Third, we packaged the project as an open-source toolkit release, with an MIT license, a complete README including installation steps and a public-API example, and spec-agnostic code, which means any parsed sixty-four by sixty-four maze file can be dropped in and our code will run on it. The specification caps total bonus points at ten.

---

## Slide 11

Three targeted fixes took beta from unsolvable to a clean one-hundred percent. The first was an off-by-one bug in the phase calculation, meaning we were checking the fire position one action too late. The environment checks pit collisions using the phase before it increments, but our planner was using the phase after, so every fifth planned step walked directly into a fire. The second fix was goal-biased frontier targeting. When the agent had no known-safe path to the goal, instead of picking the nearest unexplored cell, which was dragging the agent into a local pocket on the wrong side of the maze, we started enumerating all unexplored cells and picking the one closest to the goal by Manhattan distance. The third fix was an A-star cycle guard. We had given the planner a discount for using teleports that shortcut toward the goal, but the discount was too generous and produced negative-weight cycles in the parent map, meaning the path-reconstruction loop could spin forever because the costs cancelled out. We fixed it by flooring every edge cost at a small positive number and adding a seen-set check during path recovery that breaks the loop if we ever try to revisit any cell.

---

## Slide 12

To wrap up, our hybrid reinforcement-learning-plus-A-star agent reaches one-hundred percent success on both maze-alpha and maze-beta, five out of five evaluation episodes each, with no training on beta. Episode one is the reinforcement-learning exploration phase, and from episode two onward the agent runs the classical-search optimal path in eighty to eighty-seven turns with zero deaths. The biggest lesson was that a purely nearest-frontier explorer would have been defeated by beta's walled-off goal, which can only be reached through one specific four-cell corridor, and goal-biased frontier targeting is what made the transfer from alpha to beta work. Our AI usage is fully disclosed in the report: we used Claude Code with the Opus four-point-seven one-million-context model. Thank you, and we're happy to take questions.
