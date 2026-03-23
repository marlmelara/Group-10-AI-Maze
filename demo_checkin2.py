from environment import MazeEnvironment, Action
from maze_config import START_POS, GOAL_POS
from solver import bfs_path, positions_to_actions, chunk_actions
from visualizer import draw_maze_base


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_no_hazard_demo() -> None:
    print_header("1) LOAD MAZE WITHOUT HAZARDS")
    env = MazeEnvironment(hazards_enabled=False)

    print(f"Using START_POS = {START_POS}")
    print(f"Using GOAL_POS  = {GOAL_POS}")

    draw_maze_base(
        right_blocked=env.right_blocked,
        down_blocked=env.down_blocked,
        start=START_POS,
        goal=GOAL_POS,
        title="Maze Without Hazards",
        save_path="maze_no_hazards.png",
    )

    print_header("2) SOLVE THE MAZE USING BFS")
    path = bfs_path(env.cells, env.right_blocked, env.down_blocked, START_POS, GOAL_POS)
    if path is None:
        print("No path found. Check maze extraction and start/goal coordinates.")
        return

    actions = positions_to_actions(path)
    turns = chunk_actions(actions, chunk_size=5)

    print(f"Path found with {len(path)} cells.")
    print(f"Moves required: {len(actions)}")
    print(f"Turns of up to 5 actions: {len(turns)}")

    print_header("3) VISUALIZE THE SOLUTION")
    draw_maze_base(
        right_blocked=env.right_blocked,
        down_blocked=env.down_blocked,
        start=START_POS,
        goal=GOAL_POS,
        path=path,
        title="BFS Solution",
        save_path="maze_solution.png",
    )


def run_hazard_demo() -> None:
    print_header("4) LOAD HAZARDS")
    env = MazeEnvironment(hazards_enabled=True)

    print("Hazard summary:")
    for k, v in env.hazard_summary.items():
        print(f"  {k}: {v}")

    draw_maze_base(
        right_blocked=env.right_blocked,
        down_blocked=env.down_blocked,
        start=START_POS,
        goal=GOAL_POS,
        death_pits=env.death_pits,
        confusion_pads=env.confusion_pads,
        teleport_map=env.teleport_map,
        title="Maze With Hazards",
        save_path="maze_with_hazards.png",
    )

    print_header("5) DEMONSTRATE HAZARDS")
    featured = env.featured_hazards

    # Death pit demo
    env.reset()
    death_path = featured["pit_path"]
    death_actions = positions_to_actions(death_path)

    print("\nDeath pit demo:")
    print("Target pit:", featured["pit"])
    print("Actions:", [a.name for a in death_actions])

    for turn_index, action_chunk in enumerate(chunk_actions(death_actions, 5), start=1):
        result = env.step(action_chunk)
        print(f"Turn {turn_index}: actions={[a.name for a in action_chunk]}")
        print("  Result:", result)
        print("  Position after turn:", env.agent_pos)
        if result.is_dead:
            print("  Death pit triggered correctly.")
            break

    # Teleport demo
    env.reset()
    teleport_path = featured["teleport_path"]
    teleport_actions = positions_to_actions(teleport_path)

    print("\nTeleport demo:")
    print("Teleport source:", featured["teleport_source"])
    print("Teleport destination:", featured["teleport_dest"])
    print("Actions:", [a.name for a in teleport_actions])

    for turn_index, action_chunk in enumerate(chunk_actions(teleport_actions, 5), start=1):
        result = env.step(action_chunk)
        print(f"Turn {turn_index}: actions={[a.name for a in action_chunk]}")
        print("  Result:", result)
        print("  Position after turn:", env.agent_pos)
        if result.teleported:
            print("  Teleport triggered correctly.")
            break

    # Confusion demo
    env.reset()
    confusion_path = featured["confusion_path"]
    confusion_actions = positions_to_actions(confusion_path)

    print("\nConfusion demo:")
    print("Confusion pad:", featured["confusion"])
    print("Actions to confusion:", [a.name for a in confusion_actions])

    reached_confusion = False
    for turn_index, action_chunk in enumerate(chunk_actions(confusion_actions, 5), start=1):
        result = env.step(action_chunk)
        print(f"Turn {turn_index}: actions={[a.name for a in action_chunk]}")
        print("  Result:", result)
        print("  Position after turn:", env.agent_pos)
        if env.agent_pos == featured["confusion"] or result.is_confused:
            reached_confusion = True
            break

    if reached_confusion:
        turn_same = [Action.MOVE_UP]
        turn_next = [Action.MOVE_UP]

        result_same = env.step(turn_same)
        print("Next turn after confusion is triggered:")
        print("  Actions:", [a.name for a in turn_same])
        print("  Result:", result_same)
        print("  Position after turn:", env.agent_pos)

        result_next = env.step(turn_next)
        print("Turn after confusion expires:")
        print("  Actions:", [a.name for a in turn_next])
        print("  Result:", result_next)
        print("  Position after turn:", env.agent_pos)

    print("\nEpisode stats:")
    print(env.get_episode_stats())


if __name__ == "__main__":
    run_no_hazard_demo()
    run_hazard_demo()