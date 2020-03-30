from collections import deque

import tensorflow as tf  # Deep learning library
import numpy as np  # Handle matrices
import matplotlib.pyplot as plt
from datetime import datetime
import gym
import random

#############################
# Hyper Parameters
# Not learnt by RL process
#############################
# Max explore rate
MAX_EPSILON = 1
# Min explore rate
MIN_EPSILON = 0.01
# Decay rate for exploration
LAMBDA = 0.00001  # Default 0.00001
# Max batch size for memory buffer
BATCH_SIZE = 64
# Decay rate for future rewards Q(s',a')
GAMMA = 0.9  # Default 0.9
# Nodes in a layer of TF - default 50
NODES = 50
#############################
# Environment Variables
#############################
# Max memory buffer size - default 50000
MAX_MEMORY = 50000
# Stack size - default 4
STACK_SIZE = 10
# Timestamp for log output per session
TIMESTAMP = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# Folder suffix if needed
SUFFIX = "_STACK_10"
# Number of episodes to train on -  default 500
NUM_EPISODES = 500
# Number of live episodes to test - default 20
LIVE_EPISODES = 20
# Render game env
RENDER = False


class RLAgent:
    # Constructor
    def __init__(self, state_count, action_count, batch_size, stack_size, time_stamp):
        # Declare local variables
        self._state_count = state_count * stack_size
        self._action_count = action_count
        self._batch_size = batch_size

        self._timestamp = time_stamp
        self._write_op = None
        self._writer = None
        self._saver = None
        self._tb_summary = None
        self._loss = None
        self._tf_loss_ph = None
        self._tf_loss_summary = None

        self._tf_states = None
        self._tf_qsa = None
        self._tf_actions = None
        self._tf_output = None
        self._tf_optimise = None
        self._tf_variable_initializer = None
        self._tf_logits = None

        # Call setup
        self.create_model()

    def create_model(self):
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
            NODES,
            activation=tf.nn.relu,
            name="Layer_1",
        )

        # Layer 2, 50 nodes, relu activation
        layer_2 = tf.layers.dense(
            layer_1,
            NODES,
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

    # Input a single state to get a prediction - used for env action choice
    # Reshape to ensure data size is numpy array (1 x num_states)
    # Returns arr[3] with predicted q values for move left, do nothing, move right
    def predict_single(self, state, session):
        # Run given state input against logits layer
        return session.run(
            self._tf_logits,
            feed_dict={
                self._tf_states: state.reshape(1, self._state_count)
            }
        )

    # Predict from a batch - used by replay
    def predict_batch(self, states, session):
        # Run given batch of states against logits layer
        return session.run(
            self._tf_logits,
            feed_dict={
                self._tf_states: states
            }
        )

    # Update model for given states and Q(s,a) values
    def train_batch(self, session, states, qsa):
        # Run given states and QsAs values against optimise layer
        session.run(
            self._tf_optimise,
            feed_dict={
                self._tf_states: states,
                self._tf_qsa: qsa
            }
        )

    # Save model to local storage
    def save_model(self, sess, path):
        # Save TF model to given path
        self._saver.save(sess, path)

    @property
    def tf_variable_initializer(self):
        return self._tf_variable_initializer

    @property
    def action_count(self):
        return self._action_count

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def state_count(self):
        return self._state_count


# Stores tuples of (state, action, reward, next_state)
class Memory:
    def __init__(self, max_memory):
        # Declare memory variables
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        # Adds sample to end of samples array
        self._samples.append(sample)

        # If exceeded length, remove first from memory
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):

        # If requested number of samples is greater than size of memory buffer, return all samples in memory
        if no_samples > len(self._samples):
            return random.sample(
                self._samples,
                len(self._samples)
            )
        else:
            # Else, return batch of given size of randomly selected samples
            return random.sample(
                self._samples,
                no_samples
            )


def _make_result_dir(time_stamp):
    from errno import EEXIST
    from os import makedirs

    # Set path to create folder
    path = "./results/%s/" % time_stamp

    try:
        # Make folder
        makedirs(path)
    except OSError as exc:
        # If folder exists, ignore error
        if exc.errno == EEXIST or 183 and path.isdir(path):
            pass
        else:
            raise


class GameRunner:
    def __init__(self, sess, model, env, memory, number_of_states, numberOfActions, stack_size, max_eps, min_eps,
                 lambda_, render=True):
        # Set up game runner variables
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._number_of_states = number_of_states
        self._number_of_actions = numberOfActions
        self._stack_size = stack_size
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._lambda = lambda_
        self._eps = self._max_eps
        self._steps = 0
        self._episode = 0
        self._reward_store = []
        self._max_x_store = []
        self._stack = deque([np.zeros(2, dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

    def clear_memory(self):
        self._reward_store = []
        self._max_x_store = []

    def run(self, training):
        # Increment episode counter
        self._episode += 1

        # Reset environment and get initial state
        initial_state = self._env.reset()

        # Initialise stack for episode
        stack = self._stack_frames(initial_state, new_episode=True)

        # Init total reward
        tot_reward = 0

        # Init max x to minimum
        max_x = -100

        while True:
            # Display environment if needed
            if self._render:
                self._env.render()

            # Choose on action to take based on current state
            action = self._choose_action(stack, training)

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

            # Increment steps
            self._steps += 1

            # Track highest x achieved
            if next_state[0] > max_x:
                max_x = next_state[0]

            # If game finished, set the next state to None for storage sake
            if done:
                next_state = None
            else:
                next_state = self._stack_frames(next_state)

            # If training, store experience to memory, run training and update learning curve
            if training:
                # Add step values to memory bank
                self._memory.add_sample((
                    stack,
                    action,
                    reward,
                    next_state
                ))

                # Relearn from memory
                self._replay()

                # Exponentially decay the eps value
                self._eps = self._min_eps + (self._max_eps - self._min_eps) * np.exp(-self._lambda * self._steps)

            # move the agent to the next state and accumulate the reward
            stack = next_state
            tot_reward += reward

            # if the game is done, store episode results and break loop
            if done:
                self._reward_store.append(tot_reward)
                self._max_x_store.append(max_x)
                break

        print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state, training):
        # If random number < exploit threshold, choose a random action
        if training or random.random() < self._eps:
            return random.randint(0, self._model.action_count - 1)
        else:
            # Else, get predicted action from RL agent
            predictions = self._model.predict_single(state, self._sess)
            return np.argmax(predictions)

    def _stack_frames(self, state, new_episode=False):

        if new_episode:
            # Clear stack
            self._stack = deque([np.zeros(2, dtype=np.int) for i in range(STACK_SIZE)], maxlen=STACK_SIZE)

            # Reset with copies of state
            for i in range(STACK_SIZE):
                self._stack.append(state)

        # Not new episode - add state to stack
        else:
            self._stack.append(state)

        # Create Numpy 2D array
        stack = np.array(self._stack)
        # Flatten into 1D array
        stack = stack.flatten()

        return stack

    def _replay(self):
        # Get a batch from memory
        batch = self._memory.sample(self._model.batch_size)

        # Draw out states for all in batch
        states = np.array([val[0] for val in batch])

        # Draw out next states from batch
        next_states = np.array([(np.zeros(self._model.state_count)
                                 if val[3] is None else val[3]) for val in batch])

        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)

        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)

        # setup training arrays
        state_x = np.zeros((len(batch), self._model.state_count))
        q_value_y = np.zeros((len(batch), self._model.action_count))

        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]

            # get the current q values for all actions in state
            current_q = q_s_a[i]

            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            state_x[i] = state
            q_value_y[i] = current_q
        self._model.train_batch(self._sess, state_x, q_value_y)

    def save_results(self, training):

        # Create folder name
        timestamp = TIMESTAMP + SUFFIX

        # Only need to call during training - in live run will already exist
        if training:
            _make_result_dir(timestamp)

        # Create file suffix if needed
        live = '' if training else 'LIVE_'

        # Save TF model
        self._model.save_model(sess, "./results/%s/model.ckpt" % timestamp)

        # Record session params
        with open('./results/%s/HYPERPARAMS.TXT' % timestamp, 'w') as f:
            f.write('# Max explore rate\nMAX_EPSILON = %.2f\n' % MAX_EPSILON)
            f.write('# Min explore rate\nMIN_EPSILON = %.2f\n' % MIN_EPSILON)
            f.write('# Decay rate for exploration\nLAMBDA = %.6f\n' % LAMBDA)
            f.write('# Max batch size for memory buffer\nBATCH_SIZE = %.0f\n' % BATCH_SIZE)
            f.write('# Decay rate for future rewards Q(s,a)\nGAMMA = %.2f\n' % GAMMA)
            f.write('# Nodes in TF layers \n NODES = %.0f\n' % NODES)
            f.write('# Stack size\n STACK_SIZE = %.0f\n' % STACK_SIZE)
            f.write('# Number of training episodes\n NUM_EPISODES = %.0f\n' % NUM_EPISODES)
            f.write('# TF state size\n tf.state_size = %.0f\n' % (self._stack_size * self._number_of_states))
            f.write('# TF action count\n tf.action_count = %.0f\n' % self._number_of_actions)

        # Record rewards to CSV
        i = 0
        with open("./results/%s/%srewards.csv" % (timestamp, live), 'w') as f:
            for item in self._reward_store:
                f.write("%s, %s\n" % (i, item))
                i += 1

        # Record max_x to CSV
        i = 0
        with open('./results/%s/%smax_x.csv' % (timestamp, live), 'w') as f:
            for item in self._max_x_store:
                f.write("%s, %s\n" % (i, item))
                i += 1

        # Graph of episode results
        plt.plot(self._reward_store)
        plt.suptitle("REWARDS")
        plt.savefig("./results/%s/%srewards.png" % (timestamp, live))
        plt.close("all")

        # Graph of max X value
        plt.plot(self._max_x_store)
        plt.suptitle("MAX X")
        plt.savefig("./results/%s/%smax_x.png" % (timestamp, live))

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

    # Get number of states and actions from environment
    numberOfStates = env.env.observation_space.shape[0]
    numberOfActions = env.env.action_space.n

    # Instantiate RLAgent and memory buffer
    model = RLAgent(numberOfStates, numberOfActions, BATCH_SIZE, STACK_SIZE, TIMESTAMP)
    mem = Memory(MAX_MEMORY)

    # Scoped TF Session for automated clean up when out of scope
    with tf.Session() as sess:
        # Initialise variables
        sess.run(model.tf_variable_initializer)
        # Initialise game runner
        gr = GameRunner(
            sess,
            model,
            env,
            mem,
            numberOfStates,
            numberOfActions,
            STACK_SIZE,
            MAX_EPSILON,
            MIN_EPSILON,
            LAMBDA,
            RENDER,
        )
        episode_count = 0

        # Loop through training for max number of episodes
        while episode_count < NUM_EPISODES:
            # Print interval log
            if episode_count % 10 == 0:
                print('Episode {} of {}'.format(episode_count + 1, NUM_EPISODES))

            # Run game runner
            gr.run(training=True)
            episode_count += 1

        # Save TF model, and result graphs
        gr.save_results(training=True)

        # Wipe stored training results ready for live test
        gr.clear_memory()

        # Run live test of model
        episode_count = 0
        while episode_count < LIVE_EPISODES:
            gr.run(training=False)
            episode_count += 1

        # Save test results
        gr.save_results(training=False)
