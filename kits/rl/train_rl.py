import logging
import os
import time
from typing import Any, Dict, List, Tuple, Union

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from policy import PolicyNetwork, create_policy, sample_action, update_step

from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
from luxai_s3.state import EnvObs, EnvState
from lux.utils import find_max_in_filename

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train.log"), logging.StreamHandler()],
)
policy_logger = logging.getLogger("policy")
policy_logger.setLevel(logging.INFO)

ppath = os.path.dirname(__file__)
max_number, max_filename = find_max_in_filename(ppath)
model_path = os.path.join(os.path.dirname(__file__), max_filename)
model_path_dst = os.path.join(
    os.path.dirname(__file__), f"model_params_{max_number+1}.npz"
)


def train_basic_env(
    num_episodes: int = int(os.environ.get("TRAIN_EPISODES", "100")),
    seed: int = int(os.environ.get("TRAIN_SEED", str(int(time.time())))),
) -> None:
    """Train a policy gradient agent for the Lux AI Season 3 environment.

    Args:
        num_episodes: Number of episodes to train for. Can be set via TRAIN_EPISODES environment variable.
                     Defaults to 100 if not set.
        seed: Random seed for training. Can be set via TRAIN_SEED environment variable.
              Defaults to current timestamp if not set.
    """
    # Initialize wandb
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        logging.warning(
            "WANDB_API_KEY not found in environment variables. Wandb logging disabled."
        )
        use_wandb = False
    else:
        wandb.login(key=wandb_api_key)
        # Create descriptive run name with timestamp and parameters
        run_name = f"train_{time.strftime('%Y%m%d_%H%M%S')}_ep{num_episodes}_seed{seed}"
        wandb.init(
            project="lux-s3-rl",
            name=run_name,
            config={
                "num_episodes": num_episodes,
                "seed": seed,
                "buffer_size": 1000,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hostname": os.uname().nodename,
            },
        )
        use_wandb = True

    # Initialize environment and policy
    env: LuxAIS3Env = LuxAIS3Env()
    params: EnvParams = env.default_params

    # Initialize random key for JAX
    logging.info(f"Using random seed: {seed}")
    rng = jax.random.PRNGKey(seed)
    rng, policy_rng = jax.random.split(rng)

    # load from existing model
    if os.path.exists(model_path):
        logging.info(f"加载已有模型: {model_path}")
        data = np.load(model_path)
        mean_reward = float(data["mean_reward"])
        hidden_dims = tuple(data["hidden_dims"])

        policy = PolicyNetwork(hidden_dims=hidden_dims)

        # Reconstruct nested dictionary for parameters
        flat_params = {
            k: jnp.array(v)
            for k, v in data.items()
            if k not in ["mean_reward", "hidden_dims", "max_units", "num_actions"]
        }
        nested_params = flax.traverse_util.unflatten_dict(
            {tuple(k.split(".")): v for k, v in flat_params.items()}
        )
        policy_params = flax.core.frozen_dict.freeze(nested_params)

        policy, policy_state, optimizer = create_policy(
            policy_rng, hidden_dims=hidden_dims
        )
    else:
        logging.info("未找到已有模型,将从头开始训练")
        policy, policy_state, optimizer = create_policy(policy_rng)

    # Initialize metrics and buffers
    total_rewards: List[float] = []
    episode_losses: List[float] = []
    buffer_size = 1000

    # Initialize experience buffers
    episode_observations: List[Dict[str, Dict[str, Any]]] = []
    episode_actions: List[jnp.ndarray] = []
    episode_rewards: List[float] = []

    all_observations: List[Dict[str, Dict[str, Any]]] = []
    all_actions: List[jnp.ndarray] = []
    all_rewards: List[float] = []

    # Training loop
    for episode in range(num_episodes):
        # Reset environment
        rng, reset_rng = jax.random.split(rng)
        raw_obs, state = env.reset(reset_rng, params)

        # Get observation dictionary from environment
        obs_dict = env.get_obs(state, params)

        # Convert observation to policy format
        obs = convert_obs_to_dict(obs_dict)

        # Alternate between training as player_0 and player_1
        current_player = "player_0" if episode % 2 == 0 else "player_1"
        opponent_player = "player_1" if current_player == "player_0" else "player_0"
        current_team_idx = 0 if current_player == "player_0" else 1
        opponent_team_idx = 1 - current_team_idx

        episode_reward = 0.0
        done = False
        step_count = 0

        # Episode loop
        while not done and step_count < params.max_steps_in_match:
            step_count += 1

            # Generate keys for action sampling
            rng, action_rng, opponent_rng = jax.random.split(rng, 3)

            # Sample actions for current player using the policy network
            current_actions = sample_action(
                policy, policy_state, obs, current_player, action_rng
            )

            # Convert to full action format (movement + sap direction)
            # Remove batch dimension since we're processing one step at a time
            current_actions_unbatched = current_actions[0]  # Shape: (max_units,)
            current_full_actions = jnp.zeros((params.max_units, 3), dtype=jnp.int32)
            current_full_actions = current_full_actions.at[:, 0].set(
                current_actions_unbatched
            )

            # TODO Random opponent actions
            opponent_full_actions = jnp.zeros((params.max_units, 3), dtype=jnp.int32)
            opponent_full_actions = opponent_full_actions.at[:, 0].set(
                jax.random.randint(opponent_rng, (params.max_units,), 0, 5)
            )

            # Create action dictionary for both players
            actions = {
                current_player: np.array(current_full_actions),
                opponent_player: np.array(opponent_full_actions),
            }

            # Store experience
            episode_observations.append(
                {
                    "player_0": {
                        "units": {
                            "position": jnp.array(obs["player_0"]["units"]["position"]),
                            "energy": jnp.array(obs["player_0"]["units"]["energy"]),
                        },
                        "units_mask": jnp.array(obs["player_0"]["units_mask"]),
                        "map_features": {
                            "energy": jnp.array(
                                obs["player_0"]["map_features"]["energy"]
                            ),
                            "tile_type": jnp.array(
                                obs["player_0"]["map_features"]["tile_type"]
                            ),
                        },
                        "sensor_mask": jnp.array(obs["player_0"]["sensor_mask"]),
                        "team_points": jnp.array(obs["player_0"]["team_points"]),
                        "team_wins": jnp.array(obs["player_0"]["team_wins"]),
                        "steps": obs["player_0"]["steps"],
                        "match_steps": obs["player_0"]["match_steps"],
                        "relic_nodes": jnp.array(obs["player_0"]["relic_nodes"]),
                        "relic_nodes_mask": jnp.array(
                            obs["player_0"]["relic_nodes_mask"]
                        ),
                    },
                    "player_1": {
                        "units": {
                            "position": jnp.array(obs["player_1"]["units"]["position"]),
                            "energy": jnp.array(obs["player_1"]["units"]["energy"]),
                        },
                        "units_mask": jnp.array(obs["player_1"]["units_mask"]),
                        "map_features": {
                            "energy": jnp.array(
                                obs["player_1"]["map_features"]["energy"]
                            ),
                            "tile_type": jnp.array(
                                obs["player_1"]["map_features"]["tile_type"]
                            ),
                        },
                        "sensor_mask": jnp.array(obs["player_1"]["sensor_mask"]),
                        "team_points": jnp.array(obs["player_1"]["team_points"]),
                        "team_wins": jnp.array(obs["player_1"]["team_wins"]),
                        "steps": obs["player_1"]["steps"],
                        "match_steps": obs["player_1"]["match_steps"],
                        "relic_nodes": jnp.array(obs["player_1"]["relic_nodes"]),
                        "relic_nodes_mask": jnp.array(
                            obs["player_1"]["relic_nodes_mask"]
                        ),
                    },
                }
            )
            episode_actions.append(current_actions)

            # Step environment
            rng, step_rng = jax.random.split(rng)
            step_result = env.step(step_rng, state, actions, params)
            raw_obs, state, rewards, done_flags = step_result[
                :4
            ]  # Unpack only what we need

            # Get observation dictionary and convert to policy format
            obs_dict = env.get_obs(state, params)
            obs = convert_obs_to_dict(obs_dict)

            # Update reward based on team points, unit counts, and exploration for current player
            current_team_points = float(
                obs[current_player]["team_points"][current_team_idx]
            )
            current_unit_count = float(np.sum(obs[current_player]["units_mask"]))

            # Calculate exploration bonus based on unit positions
            unit_positions = np.array(
                obs[current_player]["units"]["position"][current_team_idx]
            )
            valid_mask = np.array(obs[current_player]["units_mask"][current_team_idx])
            valid_positions = unit_positions[valid_mask]
            # Convert positions to tuples for set operation
            position_tuples = [tuple(pos) for pos in valid_positions]
            unique_positions = len(set(position_tuples))
            # Decay exploration weight over time
            exploration_weight = max(
                0.05 * (1.0 - episode / num_episodes), 0.01
            )  # Minimum weight of 0.01
            exploration_bonus = (
                exploration_weight * unique_positions
            )  # Decaying bonus for exploring unique positions

            # Balance between points, units, and exploration
            current_reward = (
                current_team_points + 0.1 * current_unit_count + exploration_bonus
            )
            episode_reward += current_reward

            # Check termination
            done = jnp.any(done_flags[current_player])

        # Store episode data
        total_rewards.append(episode_reward)
        step_rewards = [episode_reward / len(episode_observations)] * len(
            episode_observations
        )

        # Add episode data to overall buffers
        all_observations.extend(episode_observations)
        all_actions.extend(episode_actions)
        all_rewards.extend(step_rewards)

        # Clear episode buffers
        episode_observations = []
        episode_actions = []

        # Update policy if we have enough experience
        if len(all_observations) >= buffer_size:
            # Stack observations into batched format
            obs_batch = create_batched_obs(all_observations)
            action_array = jnp.array(all_actions)  # Shape: [batch_size, max_units]
            reward_array = jnp.array(all_rewards)  # Shape: [batch_size]

            # Normalize rewards
            reward_array = (reward_array - reward_array.mean()) / (
                reward_array.std() + 1e-8
            )

            # Update policy
            policy_state, loss = update_step(
                policy, policy_state, obs_batch, action_array, reward_array, optimizer
            )

            logging.info(f"Update loss: {float(loss)}")
            episode_losses.append(float(loss))

            # Clear buffers
            all_observations = []
            all_actions = []
            all_rewards = []

        # Log progress
        if (episode + 1) % 10 == 0:
            mean_reward = np.mean(total_rewards[-10:])
            logging.info(f"Episode {episode + 1}/{num_episodes}")
            logging.info(f"Mean reward (last 10): {mean_reward:.2f}")
            logging.info(f"Latest episode reward: {episode_reward:.2f}")
            if episode_losses:
                logging.info(f"Latest loss: {episode_losses[-1]:.4f}")

            # Log metrics to wandb
            if use_wandb:
                wandb.log(
                    {
                        "episode": episode + 1,
                        "mean_reward_last_10": mean_reward,
                        "episode_reward": episode_reward,
                        "loss": episode_losses[-1] if episode_losses else None,
                    },
                    step=episode + 1,
                )

    # Save trained policy parameters
    # Convert policy parameters to flat numpy arrays
    # Extract kernel and bias from each layer
    policy_params_dict = flax.traverse_util.flatten_dict(policy_state.params)
    numpy_params = {}
    for key, value in policy_params_dict.items():
        numpy_params[".".join(str(k) for k in key)] = np.array(value)

    # Save parameters as individual arrays
    save_dict = {
        **numpy_params,
        "mean_reward": np.array(np.mean(total_rewards)),
        "hidden_dims": np.array(policy.hidden_dims),
        "max_units": np.array(params.max_units),
        "num_actions": np.array(5),
    }
    np.savez(model_path_dst, **save_dict)
    logging.info(f"Training complete. Mean reward: {np.mean(total_rewards):.2f}")
    logging.info(f"Saved model parameters to kits/rl/{model_path_dst}")

    if use_wandb:
        # Log final metrics only
        wandb.log({"final_mean_reward": np.mean(total_rewards)}, step=num_episodes)
        wandb.finish()


def convert_obs_to_dict(
    raw_obs: Union[EnvObs, Dict[str, EnvObs]]
) -> Dict[str, Dict[str, Any]]:
    """Convert raw observation to policy network format.

    Args:
        raw_obs: Either a single EnvObs object or a dictionary mapping player to EnvObs
    """
    obs_dict = {}

    def _convert_single_obs(obs: EnvObs) -> Dict[str, Any]:
        """Convert a single EnvObs to dictionary format."""
        return {
            "units": {
                "position": jnp.array(obs.units.position),
                "energy": jnp.array(obs.units.energy),
            },
            "units_mask": jnp.array(obs.units_mask),
            "map_features": {
                "energy": jnp.array(obs.map_features.energy),
                "tile_type": jnp.array(obs.map_features.tile_type),
            },
            "sensor_mask": jnp.array(obs.sensor_mask),
            "team_points": jnp.array(obs.team_points),
            "team_wins": jnp.array(obs.team_wins),
            "steps": obs.steps,
            "match_steps": obs.match_steps,
            "relic_nodes": jnp.array(obs.relic_nodes),
            "relic_nodes_mask": jnp.array(obs.relic_nodes_mask),
        }

    if isinstance(raw_obs, dict):
        # Handle Dict[str, EnvObs] case
        for player in ["player_0", "player_1"]:
            obs_dict[player] = _convert_single_obs(raw_obs[player])
    else:
        # Handle single EnvObs case
        obs_dict["player_0"] = _convert_single_obs(raw_obs)
        obs_dict["player_1"] = _convert_single_obs(raw_obs)

    return obs_dict


def create_batched_obs(
    observations: List[Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Create batched observations from a list of dictionary format observations."""
    return {
        player: {
            "units": {
                "position": jnp.stack(
                    [obs[player]["units"]["position"] for obs in observations]
                ),
                "energy": jnp.stack(
                    [obs[player]["units"]["energy"] for obs in observations]
                ),
            },
            "units_mask": jnp.stack(
                [obs[player]["units_mask"] for obs in observations]
            ),
            "map_features": {
                "energy": jnp.stack(
                    [obs[player]["map_features"]["energy"] for obs in observations]
                ),
                "tile_type": jnp.stack(
                    [obs[player]["map_features"]["tile_type"] for obs in observations]
                ),
            },
            "sensor_mask": jnp.stack(
                [obs[player]["sensor_mask"] for obs in observations]
            ),
            "team_points": jnp.stack(
                [obs[player]["team_points"] for obs in observations]
            ),
            "team_wins": jnp.stack([obs[player]["team_wins"] for obs in observations]),
            "steps": jnp.array([obs[player]["steps"] for obs in observations]),
            "match_steps": jnp.array(
                [obs[player]["match_steps"] for obs in observations]
            ),
            "relic_nodes": jnp.stack(
                [obs[player]["relic_nodes"] for obs in observations]
            ),
            "relic_nodes_mask": jnp.stack(
                [obs[player]["relic_nodes_mask"] for obs in observations]
            ),
        }
        for player in ["player_0", "player_1"]
    }


if __name__ == "__main__":
    train_basic_env()
