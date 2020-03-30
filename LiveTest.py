from collections import deque

import tensorflow as tf  # Deep learning library
import numpy as np  # Handle matrices
import matplotlib.pyplot as plt
from datetime import datetime
import gym
import random
import sys

#####################################
# Environment parameters
#####################################
# Number of episodes to run for
NUM_EPISODES = 20
# Size of frame stack
# Early instances had single frame input so may be val 1
STACK_SIZE = 1
# Render game environment
RENDER = True
# Folder to read from e.g. "./results/20200323135024"
# PATH = "./results/20200323135024"
PATH = "./results/LAMBDA_TEST/v2/LAMBDA_00010"
# Items in TF array - 2 vars / 2 vars * 4 frames
STATE_COUNT = 2 * STACK_SIZE
# Number of possible actions - need to verify from training
ACTION_COUNT = 3

print("Running LIVE TEST")

class RLAgent:
    # Constructor
    def __init__(self):
        # self._timestamp = TIMESTAMP
        self._path = PATH
        self._saver = None
        self._tf_states = None
        self._state_count = STATE_COUNT
        self._tf_qsa = None
        self._action_count = ACTION_COUNT
        self._tf_logits = None
        self._loss = None
        self._tf_optimise = None
        self._tf_variable_initializer = None

        # Call model setup
        self.create_model()

    def create_model(self):
        # Still need to define model so trained model weightings can be loaded on to it
        # Create TF placeholders for inputs
        # State data
        self._tf_states = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._state_count],
            name="tf_states",
        )

        # Q(s,a)
        self._tf_qsa = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._action_count],
            name="tf_qsa",
        )

        # Create TF layers
        # Layer 1, 50 nodes, relu activation
        layer_1 = tf.layers.dense(
            self._tf_states,
            50,  # default 50
            activation=tf.nn.relu,
            name="Layer_1",
        )

        # Layer 2, 50 nodes, relu activation
        layer_2 = tf.layers.dense(
            layer_1,
            50,  # default 50
            activation=tf.nn.relu,
            name="Layer_2",
        )

        # Output layer, limited nodes (number of actions)
        self._tf_logits = tf.layers.dense(
            layer_2,
            self._action_count,
            activation=None,
            name="Layer_Output",
        )

        # Loss function - how wrong were we
        # Predicted values, actual values
        self._loss = tf.losses.mean_squared_error(
            self._tf_qsa,
            self._tf_logits,
        )

        # Set Loss optimiser process to use Adam process - aims to improve accuracy of predictions
        self._tf_optimise = tf.train.AdamOptimizer().minimize(self._loss)

        # Initialise TF global variables
        self._tf_variable_initializer = tf.global_variables_initializer()

        # Get ref for saving NN model
        self._saver = tf.train.Saver()



    def load_model(self, session):
        path = "%s/model.ckpt" % PATH

        try:
            self._saver.restore(session, path)

        except Exception as e:
            print("Exception:", e)
            print("Unable to load TF checkpoint ", path)
            print("KILLING APP..")
            sys.exit()

    def predict_single(self, state, session):
        return session.run(
            self._tf_logits,
            feed_dict={
                self._tf_states: state.reshape(1, self._state_count)
            }
        )


class TestRunner:
    def __init__(self, sess, model, env, render=True):
        self._sess = sess
        self._model = model
        self._env = env
        self._render = render
        self._steps = 0
        self._episode = 0
        self._reward_store = []
        self._max_x_store = []
        self._stack = deque([np.zeros((2), dtype=np.int) for i in range(STACK_SIZE)], maxlen=STACK_SIZE)

    def run(self):

        # increment episode
        self._episode += 1

        # Reset environment and get initial state
        initial_state = self._env.reset()

        # Initialise stack for episode
        stack = self._stack_frames(initial_state, new_episode=True)

        # Init total reward
        total_reward = 0

        # Init max x to minimum
        max_x = -100

        while True:
            # Display environment if needed
            if self._render:
                self._env.render()

            # Choose an action using the RL agent
            action = self._choose_action(stack)

            # Take action and get output values
            next_state, reward, done, info = self._env.step(action)

            # Manually adjust rewards based on x value of car
            if next_state[0] >= -0.1:
                reward += 1
            elif next_state[0] >= 0.1:
                reward += 10
            elif next_state[0] >= 0.25:
                reward += 20
            elif next_state[0] >= 0.5:
                reward += 100
                print("Reached top of hill")

            # Track highest x achieved
            if next_state[0] > max_x:
                max_x = next_state[0]

            # If game finished, set the next state to None for storage sake
            if done:
                next_state = None
            else:
                next_state = self._stack_frames(next_state)

            # Increment step
            self._steps += 1

            # move the agent to the next state and accumulate the reward
            stack = next_state
            total_reward += reward

            # if the game is done, store episode results and break loop
            if done:
                print("Episode done, saving values: {}, {}".format(total_reward, max_x))
                self._reward_store.append(total_reward)
                self._max_x_store.append(max_x)
                break

    def _choose_action(self, state):
        # Get prediction from RL agent and return highest value
        predictions = self._model.predict_single(state, self._sess)
        return np.argmax(predictions)

    def _stack_frames(self, state, new_episode=False):
        if new_episode:
            # Clear stack
            self._stack = deque([np.zeros((2), dtype=np.int) for i in range(STACK_SIZE)], maxlen=STACK_SIZE)
            # Reset with four copies of state
            self._stack.append(state)
            self._stack.append(state)
            self._stack.append(state)
            self._stack.append(state)

            # Not new episode - add state to stack
        else:
            self._stack.append(state)

            # Create Numpy 2D array
        stack = np.array(self._stack)
        # Flatten into 1D array
        stack = stack.flatten()

        return stack

    def save_results(self):
        # Save output of max_x and rewards to image file
        # timestamp = TIMESTAMP
        path = PATH

        # Record rewards to CSV
        i = 0
        with open('%s/LIVE_rewards.csv' % path, 'w') as f:
            for item in self._reward_store:
                f.write("%s, %s\n" % (i, item))
                i += 1

        # Record max_x to CSV
        i = 0
        with open('%s/LIVE_max_x.csv' % path, 'w') as f:
            for item in self.max_x_store:
                f.write("%s, %s\n" % (i, item))
                i += 1

        # Graph of episode results
        plt.plot(self._reward_store)
        plt.suptitle("REWARDS")
        plt.savefig("%s/LIVE_rewards.png" % path)
        plt.show()
        plt.close("all")

        # Graph of max X value
        plt.plot(self._max_x_store)
        plt.suptitle("MAX X")
        plt.savefig("%s/LIVE_max_x.png" % path)
        plt.show()

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._max_x_store


if __name__ == "__main__":
    # Select and load test environment
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    model = RLAgent()

    # Scoped TF Session for automated clean up when out of scope
    with tf.Session() as sess:
        # Initialise variables
        # sess.run(model.tf_variable_initializer)
        # Initialise game runner

        # Load saved model into memory
        model.load_model(sess)

        tr = TestRunner(
            sess,
            model,
            env,
        )
        num_episodes = NUM_EPISODES
        episode_count = 0

        # Loop through training for max number of episodes
        while episode_count < num_episodes:
            # Print interval log
            if episode_count % 10 == 0:
                print('Episode {} of {}'.format(episode_count + 1, num_episodes))

            # Run game runner
            tr.run()
            episode_count += 1

        # Save TF model, and result graphs
        tr.save_results()
