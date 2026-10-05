# Reinforcement Learning environment configuration

ACTIONS = ["Up", "Down", "Left", "Right"]

NORMAL_REWARD = -0.1
INVALID_MOVE_REWARD = -1.0
WALL_REWARD = -2.0
DANGER_REWARD = -5.0
GOAL_REWARD = 10.0

MAX_STEPS = 200


# 10x10 environment map
GRID = [
    "Aoo#oooooD",
    "Doo#oo#ooo",
    "D#oooo#oDo",
    "#oo###oooo",
    "Doo#oooo#o",
    "o#oDoo#ooo",
    "ooo#oDoooo",
    "o#ooo###oD",
    "Dooo##oooo",
    "oooooooDoT"
]

GRID_SIZE = 10

START_STATE = (0, 0)
GOAL_STATE = (9, 9)



from collections import Counter


def validate_grid():
    if len(GRID) != GRID_SIZE:
        raise ValueError("The grid must have exactly 10 rows.")

    if any(len(row) != GRID_SIZE for row in GRID):
        raise ValueError("Each grid row must have exactly 10 columns.")

    counts = Counter("".join(GRID))

    expected_counts = {
        "A": 1,
        "T": 1,
        "o": 68,
        "#": 20,
        "D": 10
    }

    if counts != expected_counts:
        raise ValueError(
            f"Invalid grid counts. Expected {expected_counts}, got {dict(counts)}"
        )

    return True

from collections import deque


def find_path():
    """Find a valid path from A to T using breadth-first search."""
    queue = deque([(START_STATE, [START_STATE])])
    visited = {START_STATE}

    directions = [
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1)    # Right
    ]

    while queue:
        current_state, path = queue.popleft()

        if current_state == GOAL_STATE:
            return path

        row, col = current_state

        for row_change, col_change in directions:
            next_row = row + row_change
            next_col = col + col_change
            next_state = (next_row, next_col)

            # Check grid boundaries
            if not (0 <= next_row < GRID_SIZE and 0 <= next_col < GRID_SIZE):
                continue

            # Do not cross walls
            if GRID[next_row][next_col] == "#":
                continue

            # Do not visit the same state twice
            if next_state in visited:
                continue

            visited.add(next_state)
            queue.append((next_state, path + [next_state]))

    return None


def step(state, action):
    """
    Execute one action in the environment.

    Returns:
        next_state: Tuple (row, column)
        cell_type: Type of cell reached
        reward: Reward received for the action
        done: Whether the episode has ended
    """

    row, col = state

    movements = {
        "Up": (-1, 0),
        "Down": (1, 0),
        "Left": (0, -1),
        "Right": (0, 1)
    }

    # Check if the action is valid
    if action not in movements:
        return state, "invalid", INVALID_MOVE_REWARD, False

    row_change, col_change = movements[action]

    next_row = row + row_change
    next_col = col + col_change

    # Check grid boundaries
    if not (0 <= next_row < GRID_SIZE and 0 <= next_col < GRID_SIZE):
        return state, "invalid", INVALID_MOVE_REWARD, False

    next_state = (next_row, next_col)
    cell_type = GRID[next_row][next_col]

    # Wall
    if cell_type == "#":
        return state, "#", WALL_REWARD, False

    # Goal
    if cell_type == "T":
        return next_state, "T", GOAL_REWARD, True

    # Danger zone
    if cell_type == "D":
        return next_state, "D", DANGER_REWARD, False

    # Normal movement
    return next_state, "o", NORMAL_REWARD, False


def validate_environment():
    """Validate the main rules of the reinforcement learning environment."""

    # Validate grid structure and cell counts
    validate_grid()

    # Validate that a path from A to T exists
    path = find_path()
    if path is None:
        raise ValueError("No valid path exists from A to T.")

    # Validate normal movement
    next_state, cell_type, reward, done = step((0, 0), "Right")
    if next_state != (0, 1):
        raise ValueError("Normal movement is incorrect.")
    if cell_type != "o":
        raise ValueError("Normal cell type is incorrect.")
    if reward != NORMAL_REWARD:
        raise ValueError("Normal movement reward is incorrect.")
    if done:
        raise ValueError("Normal movement should not end the episode.")

    # Validate invalid movement
    next_state, cell_type, reward, done = step((0, 0), "Up")
    if next_state != (0, 0):
        raise ValueError("Invalid movement should keep the current state.")
    if cell_type != "invalid":
        raise ValueError("Invalid movement cell type is incorrect.")
    if reward != INVALID_MOVE_REWARD:
        raise ValueError("Invalid movement reward is incorrect.")

    # Validate wall
    next_state, cell_type, reward, done = step((0, 2), "Right")
    if next_state != (0, 2):
        raise ValueError("Wall movement should keep the current state.")
    if cell_type != "#":
        raise ValueError("Wall cell type is incorrect.")
    if reward != WALL_REWARD:
        raise ValueError("Wall reward is incorrect.")

    # Validate danger zone
    next_state, cell_type, reward, done = step((0, 0), "Down")
    if next_state != (1, 0):
        raise ValueError("Danger movement is incorrect.")
    if cell_type != "D":
        raise ValueError("Danger cell type is incorrect.")
    if reward != DANGER_REWARD:
        raise ValueError("Danger reward is incorrect.")
    if done:
        raise ValueError("Entering a danger zone should not end the episode.")

    # Validate goal
    next_state, cell_type, reward, done = step((9, 8), "Right")
    if next_state != GOAL_STATE:
        raise ValueError("Goal movement is incorrect.")
    if cell_type != "T":
        raise ValueError("Goal cell type is incorrect.")
    if reward != GOAL_REWARD:
        raise ValueError("Goal reward is incorrect.")
    if not done:
        raise ValueError("Reaching the goal should end the episode.")

    return True