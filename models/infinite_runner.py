#imports

import random
import pandas as pd
import numpy as np
# import plotly.express as px
# import plotly.graph_objs as go
pd.options.plotting.backend = "plotly"
from dataclasses import dataclass
import copy
from dataclasses import field
from radcad import Model, Simulation, Experiment
from radcad.engine import Engine, Backend

#definitions

METERS = int
COINS = int
PERCENTAGE = float   


#utils
def default(obj):
    return field(default_factory=lambda: copy.copy(obj))

#params
@dataclass
class Parameters:
    # crash_chance is the chance of crashing in the beginning
    crash_chance: PERCENTAGE = default([10])

    # crash_increase is by how much the difficulty_factor will increase each timestep
    crash_increase: PERCENTAGE = default([1])


# Initialize Parameters instance with default values
system_params = Parameters().__dict__



#state variables
@dataclass
class StateVariables:
    distance: METERS = 0
    coins: COINS = 0
    difficulty_factor: int =0
    player_crashes: int = 0

initial_state = StateVariables().__dict__


#policy functions
def p_sprint(params, substep, state_history, prev_state, **kwargs):
    '''Calculates the amount of distance covered per timestep'''

    if prev_state['player_crashes']==1:
        distance_covered = 0
    else:
        distance_covered=5
    
    return {'distance_covered': distance_covered}

def p_difficulty(params, substep, state_history, prev_state, **kwargs):
    '''Calculates the increase in difficulty every timestep'''

    if prev_state['timestep']<1:
        difficulty_increase=0

    # if player crashed in previous step then dont increase difficulty factor
    elif prev_state['player_crashes'] == 1:
        difficulty_increase=0
    
    else:
        #Every second timestep increase difficulty by 1
        difficulty_increase = 1 if prev_state['timestep']%2==1 else 0

    return {'difficulty_increase': difficulty_increase}


def p_generate_coins(params, substep, state_history, prev_state, **kwargs):
    '''Calculates the amount of coins generated'''

    if prev_state['distance']<1:
        new_coins=0

    # if player crashed in previous step then dont increase coins
    elif prev_state['player_crashes'] == 1:
        new_coins=0
    
    else:
        # if distance is less than 50 mint 1 coin, if its less than 100 mint 2, if its above 100 mint 3
        distance = prev_state['distance']
        new_coins = 1 if distance < 50 else 2 if distance < 100 else 3

    return {'new_coins': new_coins}


def p_crash(params, substep, state_history, prev_state, **kwargs):
    '''Calculates the probability of crash'''

    if prev_state['difficulty_factor']<2:
        player_crashed=0

    elif prev_state['player_crashes']==1:
        player_crashed=1
    
    else: 
        # Take the initial chance of crashing and adding it with the current difficulty factor to update the chance of crash
        crash_chance = (params['crash_chance'] + prev_state['difficulty_factor'])/100
        # If the random number generated between 0 and 1 is smaller than the percentage chance of crashing then we assume the player crashed
        if random.random()< crash_chance:
            player_crashed = 1 
        else:
            player_crashed=0

    return {'player_crashed': player_crashed}


#state updates
def s_update_distance(params, substep, state_history, prev_state, policy_input, **kwargs):
    '''Update the state of the distance variable by the amount of distance sprinted'''

    updated_distance = np.ceil(prev_state['distance'] + policy_input['distance_covered'])
    return ('distance', max(updated_distance, 0))


def s_update_coins(params, substep, state_history, prev_state, policy_input, **kwargs):
    '''Update the state of the coins variable by the amount of new coins generated'''

    updated_coins = np.ceil(prev_state['coins'] + policy_input['new_coins'])
    return ('coins', max(updated_coins, 0))


def s_update_difficulty(params, substep, state_history, prev_state, policy_input, **kwargs):
    '''Update the state of the difficulty variable by the amount of difficulty increase'''

    updated_difficulty = np.ceil(prev_state['difficulty_factor'] + policy_input['difficulty_increase'])

    return ('difficulty_factor', max(updated_difficulty, 0))

def s_update_crash(params, substep, state_history, prev_state, policy_input, **kwargs):
    '''Update the state of the crash variable'''

    updated_crash = policy_input['player_crashed']

    return ('player_crashes', max(updated_crash, 0))


#state update blocks
state_update_blocks = [
    {
        'policies': {
            'p_sprint': p_sprint,
            'p_difficulty': p_difficulty,
            'p_generate_coins':p_generate_coins,
            'p_crash':p_crash,
        },
        'variables': {
            'distance': s_update_distance,
            'coins': s_update_coins,
            'difficulty_factor':s_update_difficulty,
            'player_crashes': s_update_crash

        }
    },

]




# config and run

#number of timesteps
TIMESTEPS = 40
#number of monte carlo runs
RUNS = 1


model = Model(initial_state=initial_state, state_update_blocks=state_update_blocks, params=system_params)
simulation = Simulation(model=model, timesteps=TIMESTEPS, runs=RUNS)

experiment = Experiment(simulation)
# Select the Pathos backend to avoid issues with multiprocessing and Jupyter Notebooks
experiment.engine = Engine(backend=Backend.PATHOS)

