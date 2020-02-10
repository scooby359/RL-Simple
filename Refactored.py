import tensorflow as tf  # Deep learning library
import numpy as np  # Handle matrices
import matplotlib.pyplot as plt
from datetime import datetime
import gym
import random

############
# Hyper Parameters
# Not learnt by RL process
############

# Max explore rate
MAX_EPSILON = 1
# Min explore rate
MIN_EPSILON = 0.01
# Decay rate for exploration
LAMBDA = 0.00001
# Max batch size for memory buffer
BATCH_SIZE = 64
# Decay rate for future rewards Q(s',a')
GAMMA = 0.9
# Timestamp for log output per session
TIMESTAMP = datetime.utcnow().strftime("%Y%m%d%H%M%S")


class RLAgent:
    # Constructor
    def __init__(self, state_count, action_count, batch_size):

        # Declare local variables
        self._state_count = state_count
        self._action_count = action_count
        self._batch_size = batch_size

        self._timestamp = TIMESTAMP
        self._write_op = None
        self._writer = None
        self._saver = None # tf.train.Saver() # to save / load models
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
        self._tf_states = tf.placeholder(dtype=tf.float32, shape=[None, self._state_count], name="tf_states")
        # Q(s,a)
        self._tf_qsa = tf.placeholder(dtype=tf.float32, shape=[None, self._action_count], name="tf_qsa")
        
        # Create TF layers
        # Layer 1, 50 nodes, relu activation
        layer_1 = tf.layers.dense(self._tf_states, 50, activation=tf.nn.relu, name="Layer_1")

        # Layer 2, 50 nodes, relu activation
        layer_2 = tf.layers.dense(layer_1, 50, activation=tf.nn.relu, name="Layer_2")

        # Output layer, limited nodes (number of actions)
        self._tf_logits = tf.layers.dense(layer_2, self._action_count, activation=None, name="Layer_Output")

        # Loss function - how wrong were we
        # Predicted values, actual values
        self._loss = tf.losses.mean_squared_error(self._tf_qsa, self._tf_logits)

        # Set Loss optimiser process to use Adam process - adaptive moment estimation
        self._tf_optimise = tf.train.AdamOptimizer().minimize(self._loss)

        # Setup TensorBoard writer - create new folder for instance
        # Saves a diagram of the graph
        self._writer = tf.summary.FileWriter("./tensorboard/" + self._timestamp, tf.Session().graph)

        # Initialise TF global variables
        self._tf_variable_initializer = tf.global_variables_initializer()

    # Input a single state to get a prediction
    # Reshape to ensure data size is numpy array (1 x num_states)
    def predict_single(self, state, session):
        return session.run(self._tf_logits, feed_dict={self._tf_states: state.reshape(1, self._state_count)})

    # Predict from a batch
    def predict_batch(self, states, session):
        return session.run(self._tf_logits, feed_dict={self._tf_states: states})

    # Update model for given states and Q(s,a) values
    def train_batch(self, session, states, qsas):
        print("Starting training")
        session.run(self._tf_optimise, feed_dict={self._tf_states: states, self._tf_qsa: qsas})
        print("Training done")

    # Save model to local storage
    def save_model(self, session):
        save_path = self._saver.save(session, "./models/%s/model.ckpt" % self._timestamp)
        print("Model saved")

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
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class GameRunner:
    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0
        max_x = -100
        while True:
            if self._render:
                self._env.render()

            action = self._choose_action(state)
            next_state, reward, done, info = self._env.step(action)
            if next_state[0] >= -0.1:
                reward += 1
            elif next_state[0] >= 0.1:
                reward += 10
            elif next_state[0] >= 0.25:
                reward += 20
            elif next_state[0] >= 0.5:
                reward += 100

            if next_state[0] > max_x:
                max_x = next_state[0]
            # is the game complete? If so, set the next state to
            # None for storage sake
            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            self._replay()

            # exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._reward_store.append(tot_reward)
                self._max_x_store.append(max_x)
                break

        print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.action_count - 1)
        else:
            return np.argmax(self._model.predict_single(state, self._sess))

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
        x = np.zeros((len(batch), self._model.state_count))
        y = np.zeros((len(batch), self._model.action_count))

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
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)

    def _make_result_dir(self, time_stamp):
        from errno import EEXIST
        from os import makedirs, path

        path = "./results/%s/" % time_stamp

        try:
            makedirs(path)
        except OSError as exc:
            if exc.errno == EEXIST and path.isdir(path):
                pass
            else:
                raise

    def save_results(self):
        # Save output of max_x and rewards to image file
        timestamp = TIMESTAMP
        self._make_result_dir(timestamp)
        plt.plot(self._reward_store)
        plt.savefig("./results/%s/rewards.png" % timestamp)
        plt.show()
        plt.close("all")
        plt.plot(self._max_x_store)
        plt.savefig("./results/%s/max_x.png" % timestamp)
        plt.show()

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._max_x_store


if __name__ == "__main__":
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    numberOfStates = env.env.observation_space.shape[0]
    numberOfActions = env.env.action_space.n

    model = RLAgent(numberOfStates, numberOfActions, BATCH_SIZE)
    mem = Memory(50000)

    with tf.Session() as sess:
        sess.run(model.tf_variable_initializer)
        gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON,
                        LAMBDA)
        num_episodes = 2
        cnt = 0

        while cnt < num_episodes:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt + 1, num_episodes))
            gr.run()
            cnt += 1

        gr.save_results()
