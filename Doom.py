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
    
state_size = [100,120,4]

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
