import gym
import numpy as np
import time
import random

# constants
nb_rows = 5
nb_cols = 5
nb_locs = 4
locations = [(0,0),(0,4),(4,0),(4,3)]




# Selects the max of an numPy array or a random value (index) if this array contains multiple maxima
def argMaxRandomIndex(array):
    maxIndex = np.argmax(array)
    maxValue = array[maxIndex]
    maxSet = set()
    for i in range(len(array)):
        if array[i] == maxValue:
            maxSet.add(i)
    return random.choice(list(maxSet))

# Encodes the state variables into a state number
def encode(taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i



# calculates all states where a dropoff is unsafe
illegalDropoffStates = set()
for row in range(nb_rows):
    for col in range(nb_cols):
        passenger = 4
        for dest in range(nb_locs):
            if (row, col) not in locations:
                state = encode(row, col, passenger, dest)
                illegalDropoffStates.add(state)

    

# calculates potential function over all states.
# currently potential is finalReward (20) for end state, finalReward/2 (10) for passenger in taxi, 0 everywhere else
def calculatePotential(finalReward):
    Potential = np.zeros(500)
    for passenger in range(nb_locs+1):
        for dest in range(nb_locs):
            if passenger == dest:   # passenger at destination (and unreachable states)
                for row in range(nb_rows):
                    for col in range(nb_cols):
                        state = encode(row, col, passenger, dest)
                        Potential[state] = finalReward
            elif passenger == 4:         #passenger in taxi
                for row in range(nb_rows):
                    for col in range(nb_cols):
                        state = encode(row, col, passenger, dest)
                        Potential[state] = finalReward/2
            # else:                     # passenger not picked up yet
    return Potential 



# Q learning function. returns the array with accumulated reward for each episode
def doQLearning(shielding = False, rewardShaping= False, alpha = 0.618, amountOfEpisodes = 500, timestepReward = -1, illegalReward = -10, finalReward = 20):
    if rewardShaping:
        PotentialF = calculatePotential(finalReward)
        
    
    env = gym.make('Taxi-v3')
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    graphData = np.zeros(amountOfEpisodes, dtype=int)
    

    print("Starting training")
    print(" - Shielding: "        + str(shielding))
    print(" - Reward Shaping: "   + str(rewardShaping))
    print(" - Alpha: "            + str(alpha))
    print(" - Nb Episodes: "      + str(amountOfEpisodes))
    for episode in range(0,amountOfEpisodes):
        done = False
        G, reward = 0, 0
        state = env.reset()
        while done != True:
                action = argMaxRandomIndex(Q[state]) #1
                
                # Shielding
                if shielding and action==5 and state in illegalDropoffStates: # If dropoff, block
                    reducedQArray = Q[state][:5]
                    action = argMaxRandomIndex(reducedQArray)

                state2, reward, done, info = env.step(action) #2
                if reward == -1:
                    reward = timestepReward
                elif reward == 20:
                    reward = finalReward
                elif reward == -10:
                    reward = illegalReward

                # Reward Shaping
                if rewardShaping:
                    if reward == finalReward: 
                        reward=0  #remove original final reward
                    reward += PotentialF[state2] - PotentialF[state] # replace with reward shaping

                Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
                G += reward
                state = state2
        graphData[episode] = G

    print("done")
    print()
    return graphData