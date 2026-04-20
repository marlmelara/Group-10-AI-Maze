"""Quick sanity: 1 beta ep, 15k turns, with new frontier-first planner."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import MazeEnvironment
from agent import HybridRLAgent

MAX_TURNS = 15000

env = MazeEnvironment('beta')
agent = HybridRLAgent()
agent.set_start_goal(env.start_pos, env.goal_pos)
env.reset()
agent.reset_episode()
agent.pos = env.start_pos

last = None
best_d = 999
t_start = time.time()

for t in range(MAX_TURNS):
    actions = agent.plan_turn(last)
    last = env.step(actions)
    d = abs(env.agent_pos[0]-env.goal_pos[0]) + abs(env.agent_pos[1]-env.goal_pos[1])
    best_d = min(best_d, d)
    if t % 1000 == 0 or last.is_goal_reached:
        print(f"t={t:5d} pos={env.agent_pos} deaths={env.death_counter} "
              f"explored={len(env.cells_explored)} "
              f"known_empty={len(agent.known_empty)} "
              f"fires_any={len(agent.fires_any_phase)} "
              f"closest={best_d}", flush=True)
    if last.is_goal_reached:
        print(f"GOAL! t={env.turn_counter} actions={env.action_counter}")
        break

elapsed = time.time() - t_start
print(f"FINAL: goal={env.goal_reached} turns={env.turn_counter} "
      f"actions={env.action_counter} deaths={env.death_counter} "
      f"closest={best_d} wall_s={elapsed:.1f}")
