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
LAMBDA = 0.00001
# Max batch size for memory buffer
BATCH_SIZE = 64
# Decay rate for future rewards Q(s',a')
GAMMA = 0.9

#############################
# Environment Variables
#############################
# Timestamp for log output per session
TIMESTAMP = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# Number of episodes to train on
NUM_EPISODES = 500
# Render game env
RENDER = False


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
            50,
            activation=tf.nn.relu,
            name="Layer_1",
        )

        # Layer 2, 50 nodes, relu activation
        layer_2 = tf.layers.dense(
            layer_1,
            50,
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

        # Setup TensorBoard writer - create new folder for instance
        # self._writer = tf.summary.FileWriter("./tensorboard/" + self._timestamp, tf.Session().graph)
        self._writer = tf.summary.FileWriter("./tensorboard/" + self._timestamp)

        # Summary to record losses
        tf.summary.scalar("Loss", self._loss)

        # Write op to record all summaries (only one given above)
        self._write_op = tf.summary.merge_all()

        # Initialise TF global variables
        self._tf_variable_initializer = tf.global_variables_initializer()

        # Get ref for saving NN model
        self._saver = tf.train.Saver()

    def write_summary(self, states, qsa, episode):
        summary = sess.run(
            self._write_op,
            feed_dict={
                self._tf_states: states,
                self._tf_qsa: qsa
            }
        )
        self._writer.add_summary(summary, episode)
        self._writer.flush()

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
    def train_batch(self, session, states, qsas):
        # Run given states and QsAs values against optimise layer
        session.run(
            self._tf_optimise,
            feed_dict={
                self._tf_states: states,
                self._tf_qsa: qsas
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

class GameRunner:
    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay, render=True):
        # Set up game runner variables
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
        self._episode = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        # Increment episode counter
        self._episode += 1

        # Reset environment and get initial state
        state = self._env.reset()

        # Init total reward
        tot_reward = 0

        # Init max x to minimum
        max_x = -100

        while True:
            # Display environment if needed
            if self._render:
                self._env.render()

            # Choose on action to take based on current state
            action = self._choose_action(state)

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

            # Add step values to memory bank
            self._memory.add_sample((
                state,
                action,
                reward,
                next_state
            ))

            # Relearn from memory
            self._replay()

            # Exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, store episode results and break loop
            if done:
                self._reward_store.append(tot_reward)
                self._max_x_store.append(max_x)
                break

        print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state):
        # If random number < exploit threshold, choose a random action
        if random.random() < self._eps:
            return random.randint(0, self._model.action_count - 1)
        else:
            # Else, get predicted action from RL agent
            predictions = self._model.predict_single(state, self._sess)
            return np.argmax(predictions)

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
        states = np.zeros(
            (len(batch),
             self._model.state_count)
        )
        q_val = np.zeros(
            (len(batch),
             self._model.action_count)
        )

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
            states[i] = state
            q_val[i] = current_q
        self._model.train_batch(self._sess, states, q_val)
        self._model.write_summary(states, q_val, self._episode)

    def _make_result_dir(self, time_stamp):
        from errno import EEXIST
        from os import makedirs, path

        # Set path to create folder
        path = "./results/%s/" % time_stamp

        try:
            # Make folder
            makedirs(path)
        except OSError as exc:
            # If folder exists, ignore error
            if exc.errno == EEXIST and path.isdir(path):
                pass
            else:
                raise

    def save_results(self):
        # Save output of max_x and rewards to image file
        timestamp = TIMESTAMP
        self._make_result_dir(timestamp)

        # Save NN model
        self._model.save_model(sess, "./results/%s/model.ckpt" % timestamp)

        # Produce graph of results
        # Graph of episode results
        plt.plot(self._reward_store)
        plt.suptitle("REWARDS")
        plt.savefig("./results/%s/rewards.png" % timestamp)
        plt.show()
        plt.close("all")
        # Graph of max X value
        plt.plot(self._max_x_store)
        plt.suptitle("MAX X")
        plt.savefig("./results/%s/max_x.png" % timestamp)
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

    # Get number of states and actions from environment
    numberOfStates = env.env.observation_space.shape[0]
    numberOfActions = env.env.action_space.n

    # Instantiate RLAgent and memory buffer
    model = RLAgent(numberOfStates, numberOfActions, BATCH_SIZE)
    mem = Memory(50000)

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
            MAX_EPSILON,
            MIN_EPSILON,
            LAMBDA,
            RENDER
        )
        num_episodes = NUM_EPISODES
        episode_count = 0

        # Loop through training for max number of episodes
        while episode_count < num_episodes:
            # Print interval log
            if episode_count % 10 == 0:
                print('Episode {} of {}'.format(episode_count + 1, num_episodes))

            # Run game runner
            gr.run()
            episode_count += 1

        # Save TF model, and result graphs
        gr.save_results()
