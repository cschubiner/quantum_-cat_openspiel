"""Simple AlphaZero quantum cat example.

Take a look at the log-learner.txt in the output directory.
"""
import pyspiel
from absl import app
from absl import flags

from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.utils import spawn

from open_spiel.python.games import quantum_cat


flags.DEFINE_string("path", None, "Where to save checkpoints.")
FLAGS = flags.FLAGS


def main(unused_argv):
    game = pyspiel.load_game("python_quantum_cat")

    config = alpha_zero.Config(
        game="python_quantum_cat",  # Our registered game name
        path=FLAGS.path,
        learning_rate=0.002,  # Lower learning rate due to game complexity
        weight_decay=1e-4,
        train_batch_size=256,  # Larger batch size for more stable learning
        replay_buffer_size=2**16,  # Larger buffer for more diverse experience
        replay_buffer_reuse=4,
        max_steps=1000,  # More steps due to game complexity
        checkpoint_freq=100,

        actors=1,  # More actors for parallel experience collection
        evaluators=2,
        uct_c=2.2,  # Higher exploration constant
        max_simulations=100,  # More simulations per move due to game complexity
        policy_alpha=0.3,
        policy_epsilon=0.25,
        temperature=1,
        temperature_drop=10,
        evaluation_window=100,
        eval_levels=7,

        nn_model="mlp",
        nn_width=128,  # Wider network for more capacity
        nn_depth=2,    # Deeper network for more complex patterns
        # observation_shape=None,  # Let the game specify this
        # output_size=None,       # Let the game specify this
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions(),

        quiet=True,
    )
    alpha_zero.alpha_zero(config)


if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
