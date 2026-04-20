"""Verify alpha still solves with new planner."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import MazeEnvironment
from agent import HybridRLAgent

env = MazeEnvironment('alpha')
agent = HybridRLAgent()
agent.set_start_goal(env.start_pos, env.goal_pos)

for ep in range(1, 4):
    env.reset()
    agent.reset_episode()
    agent.pos = env.start_pos
    last = None
    t_start = time.time()
    for t in range(10000):
        actions = agent.plan_turn(last)
        last = env.step(actions)
        if last.is_goal_reached:
            break
    elapsed = time.time() - t_start
    print(f"ep={ep} goal={env.goal_reached} turns={env.turn_counter} "
          f"actions={env.action_counter} deaths={env.death_counter} "
          f"explored={len(env.cells_explored)} wall_s={elapsed:.2f}")
