"""5-episode beta evaluation with the working planner."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import MazeEnvironment
from agent import HybridRLAgent

MAX_TURNS = 15000
NUM_EPS = 5

env = MazeEnvironment('beta')
agent = HybridRLAgent()
agent.set_start_goal(env.start_pos, env.goal_pos)

successes = 0
results = []
for ep in range(1, NUM_EPS + 1):
    env.reset()
    agent.reset_episode()
    agent.pos = env.start_pos
    last = None
    t_start = time.time()
    best_d = 999
    for t in range(MAX_TURNS):
        actions = agent.plan_turn(last)
        last = env.step(actions)
        d = abs(env.agent_pos[0]-env.goal_pos[0]) + abs(env.agent_pos[1]-env.goal_pos[1])
        best_d = min(best_d, d)
        if last.is_goal_reached:
            break
    el = time.time() - t_start
    if env.goal_reached:
        successes += 1
    line = (f"ep={ep} goal={env.goal_reached} turns={env.turn_counter} "
            f"actions={env.action_counter} deaths={env.death_counter} "
            f"explored={len(env.cells_explored)} known={len(agent.known_empty)} "
            f"closest={best_d} wall_s={el:.1f}")
    print(line, flush=True)
    results.append(line)

print(f"\nSUMMARY: {successes}/{NUM_EPS} beta episodes solved")
