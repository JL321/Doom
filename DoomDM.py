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

stacked_frames = deque([np.zeros((100,120), dtype = np.int) for i in range(stack_size)], maxlen=4)

def create_environment():
    
    game = DoomGame()
    
    game.load_config("deadly_corridor.cfg")
    
    game.set_doom_scenario_path('deadly_corridor.wad')
    
    game.init()
    
    possible_actions = np.identity(5, dtype = int).tolist()
    
    return game, possible_actions

game, possible_actions = create_environment()

def preprocess_frame(frame):
    
    cropped_frame = frame[15:-5,20:-20]
    
    normalized_frame = cropped_frame/255
    
    preprocessed_frame = transform.resize(cropped_frame, [100,120])
    
    return preprocessed_frame

def stack_frames(stacked, state, is_new_episode):
    
    frame = preprocess_frame
    
    if is_new_episode:
        
        stacked_frames = deque([np.zeros((100,120), dtype = np.int) for i in range(stack_size)], maxlen = 4)
        
        for i in range(4):
            stacked_frames.append(frame)
            
        stacked_state = np.stack(stacked_frames, axis = 2)
        
    else:
        
        stacked_frames.append(frame)
        
        stacked_state = np.stack(stacked_frames, axis = 2)
        
    return stacked_state, stacked_frames

#Initialize hyper parameters
    
state_size = [4,100,120] #Changed from 100,120,4

action_size = game.get_available_buttons_size()
learning_rate = 0.00025

total_episodes = 5000
max_steps = 5000
batch_size = 64

target_tau = 10000

explore_starts = 1
explore_ends = 0.01
decay_rate = 0.00005

gamma = 0.95

replay_max = 100000
replay_min = 100000

training = False #Visualization

episode_render = False

class DuelDQNet:
    
    def __init__(self, state_space, action_space, learning_rate, name):
        
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.name = name
        
        with tf.variable_scope(self.name):
            
            self.inputs = tf.placeholder(tf.float32, shape = (None, state_size))
            
            self.ISweights = tf.placeholder(tf.float32, shape = (None, 1))
            
            self.actions = tf.placeholder(tf.float32, shape = (None, action_size))
            
            self.target_Q = tf.placeholder(tf.float32, shape = (None))
            
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 32,
                                         kernel = (8,8),
                                         strides = [4,4],
                                         padding = 'same')
            
            self.conv1_out= tf.nn.relu(self.conv1)
            
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                         filters = 64,
                                         kernel = (4,4),
                                         strides = (2,2),
                                         kernel_initializer = 
                                         tf.contrib.layers.xavier_initializer_conv2d(),
                                         )
            
            self.conv2_out = tf.nn.relu(self.conv2)
            
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                          filters = 128,
                                          kernel = (4,4),
                                          strides = (2,2),
                                          kernel_initializer =
                                          tf.contrib.layers.xavier_initializer_conv2d())
            
            self.conv3_out = tf.nn.relu(self.conv3)
            
            self.flatten = tf.layers.flatten(self.conv3_out)
            
            self.fc1_v = tf.layers.dense(inputs = self.flatten, 
                                        units = 512, 
                                        activation = tf.nn.relu,
                                        kernel_initializer =
                                        tf.contrib.layers.xavier_initializer_conv2d())
            
            self.fc2_v = tf.layers.dense(inputs = self.fc1_v,
                                         units = 1,
                                         activation = tf.nn.relu,
                                         kernel_initializer = 
                                         tf.contrib.layers.xavier_initializer_conv2())
            
            self.fc1_a = tf.layers.dense(inputs = self.flatten,
                                         units = 512,
                                         activation = tf.nn.relu,
                                         kernel_initializer = 
                                         tf.contrib.layers.xavier_initializer_conv2d())
            
            self.fc2_a = tf.layers.dense(inputs = self.fc1_a,
                                         units = self.action_space,
                                         activation = tf.nn.relu,
                                         kernel_initializer =
                                         tf.contrib.layers.xavier_initializer_conv2d())
            
            self.output = self.fc2_v + (self.fc2_a - tf.reduce_mean(self.fc2_a, axis = 1, keepdims = True))
            
            self.Q = tf.reduce_sum(self.output*tf.one_hot(self.actions, depth = self.action_space))
            
            self.loss = tf.square(self.target_Q - self.Q)
            


            
            
            