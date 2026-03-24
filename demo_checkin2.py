from environment import MazeEnvironment, Action
from maze_config import START_POS, GOAL_POS
from solver import bfs_path, positions_to_actions, chunk_actions
from visualizer import draw_maze_base, animate_path


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def collect_live_path(env: MazeEnvironment, action_chunks):
    """
    Step through the environment one action at a time and collect
    the visited positions for animation.
    """
    live_path = [env.agent_pos]

    for action_chunk in action_chunks:
        for action in action_chunk:
            before = env.agent_pos
            result = env.step([action])

            # If moved, append new position
            if env.agent_pos != before:
                live_path.append(env.agent_pos)

            # If teleported, also append the teleport destination
            elif result.teleported:
                live_path.append(result.current_position)

            # If death occurs, show death cell then respawn
            if result.is_dead:
                if result.current_position != live_path[-1]:
                    live_path.append(result.current_position)
                if env.agent_pos != live_path[-1]:
                    live_path.append(env.agent_pos)
                return live_path

    return live_path


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

    print_header("3) LIVE BFS SOLUTION")
    animate_path(
        right_blocked=env.right_blocked,
        down_blocked=env.down_blocked,
        start=START_POS,
        goal=GOAL_POS,
        path=path,
        title="Live BFS Solution",
        delay=0.01,
        reverse=False,      # change to False if you want actual start->goal direction
        close_to_skip=True,
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

    print_header("5) LIVE HAZARD DEMOS")
    featured = env.featured_hazards

    # -------------------------
    # Death pit demo
    # -------------------------
    env.reset()
    death_path = featured["pit_path"]
    death_actions = positions_to_actions(death_path)
    death_chunks = chunk_actions(death_actions, 5)

    print("\nDeath pit demo:")
    print("Target pit:", featured["pit"])
    print("Actions:", [a.name for a in death_actions])

    death_live_path = collect_live_path(env, death_chunks)

    animate_path(
        right_blocked=env.right_blocked,
        down_blocked=env.down_blocked,
        start=START_POS,
        goal=GOAL_POS,
        path=death_live_path,
        death_pits=env.death_pits,
        confusion_pads=env.confusion_pads,
        teleport_map=env.teleport_map,
        title="Live Death Pit Demo",
        delay=0.08,
        reverse=False,
        close_to_skip=True,
    )

    # -------------------------
    # Teleport demo
    # -------------------------
    env.reset()
    teleport_path = featured["teleport_path"]
    teleport_actions = positions_to_actions(teleport_path)
    teleport_chunks = chunk_actions(teleport_actions, 5)

    print("\nTeleport demo:")
    print("Teleport source:", featured["teleport_source"])
    print("Teleport destination:", featured["teleport_dest"])
    print("Actions:", [a.name for a in teleport_actions])

    teleport_live_path = collect_live_path(env, teleport_chunks)

    animate_path(
        right_blocked=env.right_blocked,
        down_blocked=env.down_blocked,
        start=START_POS,
        goal=GOAL_POS,
        path=teleport_live_path,
        death_pits=env.death_pits,
        confusion_pads=env.confusion_pads,
        teleport_map=env.teleport_map,
        title="Live Teleport Demo",
        delay=0.10,
        reverse=False,
        close_to_skip=True,
    )

    # -------------------------
    # Confusion demo
    # -------------------------
    env.reset()
    confusion_path = featured["confusion_path"]
    confusion_actions = positions_to_actions(confusion_path)
    confusion_chunks = chunk_actions(confusion_actions, 5)

    print("\nConfusion demo:")
    print("Confusion pad:", featured["confusion"])
    print("Actions to confusion:", [a.name for a in confusion_actions])

    confusion_live_path = collect_live_path(env, confusion_chunks)

    # Show one move while confusion is active
    result_same = env.step([Action.MOVE_UP])
    confusion_live_path.append(result_same.current_position)

    # Show next move after that
    result_next = env.step([Action.MOVE_UP])
    confusion_live_path.append(result_next.current_position)

    animate_path(
        right_blocked=env.right_blocked,
        down_blocked=env.down_blocked,
        start=START_POS,
        goal=GOAL_POS,
        path=confusion_live_path,
        death_pits=env.death_pits,
        confusion_pads=env.confusion_pads,
        teleport_map=env.teleport_map,
        title="Live Confusion Demo",
        delay=0.14,
        reverse=False,
        close_to_skip=True,
    )

    print("\nEpisode stats:")
    print(env.get_episode_stats())


if __name__ == "__main__":
    run_no_hazard_demo()
    run_hazard_demo()