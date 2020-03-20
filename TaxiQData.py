import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import random

nb_rows = 5
nb_cols = 5
nb_locs = 4
locations = [(0,0),(0,4),(4,0),(4,3)]

env = gym.make('Taxi-v3')


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




illegalDropoffStates = set()
for row in range(nb_rows):
    for col in range(nb_cols):
        passenger = 4
        for dest in range(nb_locs):
            if (row, col) not in locations:
                state = encode(row, col, passenger, dest)
                illegalDropoffStates.add(state)

                




Potential = np.zeros(500)
for passenger in range(nb_locs+1):
    for dest in range(nb_locs):
        if passenger ==4:   #passenger in taxi
            for row in range(nb_rows):
                for col in range(nb_cols):
                    state = encode(row, col, passenger, dest)
                    dist = abs(locations[dest][0] - row) + abs(locations[dest][1] - col)
                    Potential[state] = 18 - dist
        elif dest == passenger:   # passenger at destination (and unreachable states)
            for row in range(nb_rows):
                for col in range(nb_cols):
                    state = encode(row, col, passenger, dest)
                    Potential[state] = 20
        else:
            for row in range(nb_rows):
                for col in range(nb_cols):
                    state = encode(row, col, passenger, dest)
                    dist = abs(locations[passenger][0] - row) + abs(locations[passenger][1] - col)
                    Potential[state] = 8 - dist      

def doQLearning(shielding = False, rewardShaping= False, alpha = 0.618, amountOfEpisodes = 500):

    Q3 = np.zeros([env.observation_space.n, env.action_space.n])
    graphData3 = np.zeros(amountOfEpisodes, dtype=int)


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
                action = argMaxRandomIndex(Q3[state]) #1
                
                # Shielding
                if shielding and action==5 and state in illegalDropoffStates: # If dropoff, block
                    reducedQArray = Q3[state][:5]
                    action = argMaxRandomIndex(reducedQArray)

                state2, reward, done, info = env.step(action) #2

                # Reward Shaping
                if rewardShaping:
                    reward += Potential[state2] - Potential[state]

                Q3[state,action] += alpha * (reward + np.max(Q3[state2]) - Q3[state,action]) #3
                G += reward
                state = state2
        graphData3[episode] = G

    print(" done")
    return graphData3

    
          
          
            
                   
                



#Plot the non-shield and shield data

plt.plot(doQLearning(), ".")
plt.plot(doQLearning(shielding=True), ".")
plt.plot(doQLearning(rewardShaping=True), ".")

plt.title('Reward versus episodes with and without shielding')
plt.show()



