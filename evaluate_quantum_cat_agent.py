"""Script to evaluate a saved Quantum Cat PPO agent."""

import random
import numpy as np
import torch
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.vector_env import SyncVectorEnv
from quantum_cat_ppo import evaluate_agent

def load_and_evaluate_agent(
    agent_path,
    num_players=3,
    player_id=0,
    num_episodes=100,
    seed=1234
):
    """Load and evaluate a saved agent."""
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environments
    game = pyspiel.load_game("python_quantum_cat", {"players": num_players})
    def make_env():
        return rl_environment.Environment(game=game, seed=seed)
    num_envs = 2
    envs = SyncVectorEnv([make_env() for _ in range(num_envs)])
    
    # Get observation shape from environment
    time_step = envs.reset()
    obs_spec = time_step[0].observations["info_state"][player_id]
    info_state_shape = (len(obs_spec),)
    
    # Create and load agent
    from open_spiel.python.pytorch.ppo import PPO, PPOAgent
    agent = PPO(
        input_shape=info_state_shape,
        num_actions=game.num_distinct_actions(),
        num_players=num_players,
        player_id=player_id,
        num_envs=num_envs,
        steps_per_batch=16,  # Doesn't matter for evaluation
        agent_fn=PPOAgent,
    )
    agent.load_state_dict(torch.load(agent_path))
    agent.eval()
    
    print(f"\nEvaluating agent from {agent_path}")
    print("1. vs Random opponents:")
    evaluate_agent(agent, envs, num_episodes=num_episodes, player_id=player_id)
    
    print("\n2. Self-play (all seats use this agent):")
    evaluate_agent(agent, envs, num_episodes=num_episodes, 
                  player_id=player_id, opponent_agent=agent)

if __name__ == "__main__":
    load_and_evaluate_agent("quantum_cat_agent_p0.pth")
