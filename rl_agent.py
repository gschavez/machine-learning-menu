
import random

import numpy as np
from sklearn.linear_model import SGDRegressor

import rl_environment as env

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
EPISODES = 400            # enough to converge, fast enough for Render
INITIAL_EPSILON = 1.0     # start by exploring everything
MIN_EPSILON = 0.05        # never stop exploring completely
EPSILON_DECAY = 0.985     # multiplicative decay applied after each episode
GAMMA = 0.95              # discount factor
LEARNING_RATE = 0.1       # eta0 of SGDRegressor
TRAIN_MAX_STEPS = 120     # step limit per training episode

ACTIONS = env.ACTIONS
N_STATES = env.GRID_SIZE * env.GRID_SIZE
N_ACTIONS = len(ACTIONS)
N_FEATURES = N_STATES * N_ACTIONS

# Pre-computed feature matrix: row (s * N_ACTIONS + a) is the one-hot vector
# for the pair (state s, action a). Reading rows is much faster than
# building vectors at every step.
_FEATURES = np.eye(N_FEATURES)


def get_params():
    """Return the training parameters (shown in the parameters panel)."""
    return {
        "initial_epsilon": INITIAL_EPSILON,
        "min_epsilon": MIN_EPSILON,
        "epsilon_decay": EPSILON_DECAY,
        "gamma": GAMMA,
        "episodes": EPISODES,
    }


def _state_index(state):
    row, col = state
    return row * env.GRID_SIZE + col


def _state_features(state):
    """Feature rows of the four actions for one state (shape 4 x 400)."""
    start = _state_index(state) * N_ACTIONS
    return _FEATURES[start:start + N_ACTIONS]


class QLearningAgent:
    def __init__(self):
        self.model = SGDRegressor(
            penalty=None,
            learning_rate="constant",
            eta0=LEARNING_RATE,
            fit_intercept=False,
            random_state=SEED,
        )
        # partial_fit must be called once before predict(); start all Q at 0.
        self.model.partial_fit(_FEATURES[:N_ACTIONS], np.zeros(N_ACTIONS))
        self.model.coef_[:] = 0.0

        self.epsilon = INITIAL_EPSILON
        self.rng = random.Random(SEED)

    # -- Q estimation ------------------------------------------------------
    def q_values(self, state):
        """Estimated Q for the four actions in a state (predict)."""
        return self.model.predict(_state_features(state))

    def best_action(self, state):
        """Greedy action; ties are broken by the fixed action order."""
        return ACTIONS[int(np.argmax(self.q_values(state)))]

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        return self.best_action(state)

    # -- learning ----------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """One Q-learning update using partial_fit."""
        target = reward
        if not done:
            target += GAMMA * float(np.max(self.q_values(next_state)))
        x = _state_features(state)[ACTIONS.index(action)].reshape(1, -1)
        self.model.partial_fit(x, [target])

    def decay_epsilon(self):
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(agent):
    """Run all training episodes. Returns per-episode statistics."""
    successes = 0
    total_rewards = []

    for _ in range(EPISODES):
        state = env.START_STATE
        episode_reward = 0.0

        for _ in range(TRAIN_MAX_STEPS):
            action = agent.choose_action(state)
            next_state, _cell, reward, done = env.step(state, action)
            agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            if done:
                successes += 1
                break

        total_rewards.append(episode_reward)
        agent.decay_epsilon()

    return successes, total_rewards


# ---------------------------------------------------------------------------
# Evaluation (no exploration)
# ---------------------------------------------------------------------------
def evaluate(agent):
    """Follow the greedy policy step by step (epsilon = 0)."""
    state = env.START_STATE
    steps = []
    path = [list(state)]
    total_reward = 0.0
    goal_reached = False

    for number in range(1, env.MAX_STEPS + 1):
        action = agent.best_action(state)
        next_state, cell_type, reward, done = env.step(state, action)
        total_reward += reward
        steps.append({
            "step": number,
            "state": state,
            "action": action,
            "next_state": next_state,
            "cell_type": cell_type,
            "reward": round(reward, 2),
        })
        path.append(list(next_state))
        state = next_state
        if done:
            goal_reached = True
            break

    return steps, path, round(total_reward, 2), goal_reached


# ---------------------------------------------------------------------------
# Q-values of every valid (non-wall) state
# ---------------------------------------------------------------------------
def collect_q_values(agent):
    rows = []
    for r in range(env.GRID_SIZE):
        for c in range(env.GRID_SIZE):
            if env.GRID[r][c] == "#":
                continue
            q = agent.q_values((r, c))
            rows.append({
                "state": (r, c),
                "up": round(float(q[0]), 2),
                "down": round(float(q[1]), 2),
                "left": round(float(q[2]), 2),
                "right": round(float(q[3]), 2),
            })
    return rows


# ---------------------------------------------------------------------------
# Single entry point
# ---------------------------------------------------------------------------
def run_training():
    """Train the agent and return everything the page needs."""
    random.seed(SEED)
    np.random.seed(SEED)

    agent = QLearningAgent()
    successes, total_rewards = train(agent)
    evaluation_steps, path, eval_reward, goal_reached = evaluate(agent)

    results = {
        "total_episodes": EPISODES,
        "successful_episodes": successes,
        "success_percentage": round(100 * successes / EPISODES, 1),
        "average_reward": round(sum(total_rewards) / EPISODES, 2),
        "final_epsilon": round(agent.epsilon, 4),
        "eval_moves": len(evaluation_steps),
        "eval_total_reward": eval_reward,
        "goal_reached": goal_reached,
    }

    return {
        "results": results,
        "path": path,
        "evaluation_steps": evaluation_steps,
        "q_values": collect_q_values(agent),
        "params": get_params(),
    }


if __name__ == "__main__":
    out = run_training()
    print(out["results"])
    print("Path length:", len(out["path"]) - 1)