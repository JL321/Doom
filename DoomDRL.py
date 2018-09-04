import tensorflow as tf
import numpy as np
from vizdoom import *
import random
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

stack_size = 4

stacked_frames = deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen=4)

def create_environment():
    
    game = DoomGame()
    
    game.load_config(r'C:\Users\james\Anaconda3\envs\tensorflow\Lib\site-packages\vizdoom\scenarios\deadly_corridor.cfg') #Path to deadly_corridor.cfg
    
    game.set_doom_scenario_path(r'C:\Users\james\Anaconda3\envs\tensorflow\Lib\site-packages\vizdoom\scenarios\deadly_corridor.wad')
    
    game.init()
    
    possible_actions = np.identity(7, dtype = int).tolist()
    
    return game, possible_actions

game, possible_actions = create_environment()

def preprocess_frame(frame):
    
    cropped_frame = frame[15:-5,20:-20]
    
    normalized_frame = cropped_frame/255
    
    preprocessed_frame = transform.resize(cropped_frame, [100,120])
    
    return preprocessed_frame

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

#Initialize hyper parameters
    
state_size = [100,120,4] #Changed from 100,120,4

action_size = game.get_available_buttons_size()
learning_rate = 0.00025

total_episodes = 5000
max_steps = 5000
batch_size = 64

max_tau = 10000 #Steps before target update

explore_start = 1
explore_stop = 0.01
decay_rate = 0.00005

gamma = 0.95

replay_max = 50000
replay_min = 50000

training = False

episode_render = False

class DuelDQNet:
    
    def __init__(self, state_space, action_space, learning_rate, name):
        
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.name = name
        
        with tf.variable_scope(self.name):
            
            self.inputs = tf.placeholder(tf.float32, shape = (None, *state_size))
            
            self.ISWeights = tf.placeholder(tf.float32, shape = (None, 1))
            
            self.actions = tf.placeholder(tf.float32, shape = (None, action_size))
            
            self.target_Q = tf.placeholder(tf.float32, shape = (None))
            
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 32,
                                         kernel_size = (8,8),
                                         strides = [4,4],
                                         padding = 'same')
            
            self.conv1_out= tf.nn.elu(self.conv1)
            
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                         filters = 64,
                                         kernel_size = (4,4),
                                         strides = (2,2),
                                         kernel_initializer = 
                                         tf.contrib.layers.xavier_initializer()
                                         )
            
            self.conv2_out = tf.nn.elu(self.conv2)
            
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                          filters = 128,
                                          kernel_size = (4,4),
                                          strides = (2,2),
                                          kernel_initializer =
                                          tf.contrib.layers.xavier_initializer()
                                          )
            
            self.conv3_out = tf.nn.elu(self.conv3)
            
            self.flatten = tf.layers.flatten(self.conv3_out)
            
            self.fc1_v = tf.layers.dense(inputs = self.flatten, 
                                        units = 512, 
                                        activation = tf.nn.elu,
                                        kernel_initializer =
                                        tf.contrib.layers.xavier_initializer()
                                        )
            
            self.fc2_v = tf.layers.dense(inputs = self.fc1_v,
                                         units = 1,
                                         activation = tf.nn.elu,
                                         kernel_initializer = 
                                         tf.contrib.layers.xavier_initializer()
                                         )
            
            self.fc1_a = tf.layers.dense(inputs = self.flatten,
                                         units = 512,
                                         activation = tf.nn.elu,
                                         kernel_initializer = 
                                         tf.contrib.layers.xavier_initializer()
                                         )
            
            self.fc2_a = tf.layers.dense(inputs = self.fc1_a,
                                         units = self.action_space,
                                         activation = tf.nn.elu,
                                         kernel_initializer =
                                         tf.contrib.layers.xavier_initializer()
                                         )
            
            self.output = self.fc2_v + (self.fc2_a - tf.reduce_mean(self.fc2_a, axis = 1, keepdims = True))
            
            self.Q = tf.reduce_sum(tf.multiply(self.output,self.actions), axis = 1)
            
            self.absolute_errors = tf.abs(self.target_Q - self.Q)
            
            self.loss = tf.reduce_mean(self.ISWeights*tf.squared_difference(self.target_Q, self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            
tf.reset_default_graph()
    
DQNetwork = DuelDQNet(state_size, action_size, learning_rate, 'DQN')

TargetNet = DuelDQNet(state_size, action_size, learning_rate, 'TDQN')
    
class SumTree(object):

    data_pointer = 0
    

    def __init__(self, capacity):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
    
    

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        

        # Update data frame
        self.data[self.data_pointer] = data
        
        # Update the leaf
        self.update (tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
            
    
    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change through tree (as this is a sum tree)
        while tree_index != 0:    #
            
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, v):
    
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node
    
class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree 

        self.tree = SumTree(capacity)
        
    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)   # set the max p for new p
        
    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []
        
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        
        for i in range(n):
  
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
            
            #Ensure that IS weights are smaller than 1
            
            b_idx[i]= index
            
            experience = [data]
            
            memory_b.append(experience)

        
        return b_idx, memory_b, b_ISWeights
    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
        
game, possible_actions = create_environment()        
        
memory = Memory(replay_max)

game.new_episode()

#Fill Initial Memory

for i in range(replay_min):
    
    if i%1000 == 0 :
        print(i)
    
    if i == 0:
        
        state = game.get_state().screen_buffer

        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    action = random.choice(possible_actions)
    
    reward= game.make_action(action)
    
    done = game.is_episode_finished()
        
    if done:
        
        next_state = np.zeros(state.shape)
        
        experience = state,action, reward, next_state, done
        
        memory.store(experience)
        
        game.new_episode()
        state = game.get_state().screen_buffer
        
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
    else:
        
        next_state = game.get_state().screen_buffer
        
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        
        experience = state, action, reward, next_state, done
        
        memory.store(experience)
        
        state = next_state
        
writer = tf.summary.FileWriter("/tensorboard/dddqn/1")

tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()
        
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    
    exp_exp_tradeoff = np.random.rand()
    
    explore_probability = explore_stop + (explore_start - explore_stop)* np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        
        action = random.choice(possible_actions)
        
    else:
        
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs: state.reshape((1, *state.shape))}) #Try removing * after
        
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
        
    return action, explore_probability 
        
def update_target_graph():
    
    op_holder = []
    
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQN")
    
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TDQN")
                
    for from_var, to_var in zip(from_vars, to_vars):
        
        op_holder.append(to_var.assign(from_var))
        
    return op_holder


# Saver will help us to save our model# Saver 
saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        
        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0
        
        # Set tau = 0
        tau = 0

        # Init the game
        game.init()
        
        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)
        
        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            game.new_episode()
            
            state = game.get_state().screen_buffer
            
            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        
            while step < max_steps:
                step += 1
                
                # Increase the C step
                tau += 1
                
                # Increase decay_step
                decay_step +=1
                
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                # Do the action
                reward = game.make_action(action)

                # Look if the episode is finished
                done = game.is_episode_finished()
                
                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((120,140), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)

                else:
                    # Get the next state
                    next_state = game.get_state().screen_buffer
                    
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)
                    
                    # st+1 is now our current state
                    state = next_state


                ### LEARNING PART            
                # Obtain random mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                
                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch]) 
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)              
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []

                
                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')
                
                # Get Q values for next_state 
                q_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs: next_states_mb})
                
                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNet.output, feed_dict = {TargetNet.inputs: next_states_mb})
                
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a') 
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    
                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                
                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                    feed_dict={DQNetwork.inputs: states_mb,
                                               DQNetwork.target_Q: targets_mb,
                                               DQNetwork.actions: actions_mb,
                                              DQNetwork.ISWeights: ISWeights_mb})
              
                
                
                # Update priority
                memory.batch_update(tree_idx, absolute_errors)
                
                
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions: actions_mb,
                                              DQNetwork.ISWeights: ISWeights_mb})
                writer.add_summary(summary, episode)
                writer.flush()
                
                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")
                
with tf.Session() as sess:
    
    game = DoomGame()
    
    # Load the correct configuration (TESTING)
    game.load_config(r'C:\Users\james\Anaconda3\envs\tensorflow\Lib\site-packages\vizdoom\scenarios\deadly_corridor.cfg')
    
    # Load the correct scenario (in our case deadly_corridor scenario)
    game.set_doom_scenario_path(r'C:\Users\james\Anaconda3\envs\tensorflow\Lib\site-packages\vizdoom\scenarios\deadly_corridor.wad')
    
    game.init()    
    
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    
    for i in range(10):
        
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
        while not game.is_episode_finished():
            ## EPSILON GREEDY STRATEGY
            # Choose action a from state s using epsilon greedy.
            ## First we randomize a number
            exp_exp_tradeoff = np.random.rand()
            

            explore_probability = 0.01
    
            if (explore_probability > exp_exp_tradeoff):
                # Make a random action (exploration)
                action = random.choice(possible_actions)
        
            else:
                # Get action from Q-network (exploitation)
                # Estimate the Qs values state
                Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs: state.reshape((1, *state.shape))})
        
                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]
            
            game.make_action(action)
            done = game.is_episode_finished()
        
            if done:
                break  
                
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        
        score = game.get_total_reward()
        print("Score: ", score)
    
    game.close()