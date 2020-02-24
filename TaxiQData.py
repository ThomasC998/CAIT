import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import random

env = gym.make('Taxi-v3')
Q = np.zeros([env.observation_space.n, env.action_space.n])
G = 0
alpha = 0.618
amountOfEpisodes = 1000

# Selects the max of an numPy array or a random value (index) if this array contains multiple maxima
def argMaxRandomIndex(array):
    sortedA = np.sort(array)[::-1]
    i=0
    while i != len(sortedA)-1 and sortedA[i] == sortedA[i+1]:
        i+=1
    if i==0:
        return 0
    else:
        return random.randint(0, i)

graphData = np.zeros(amountOfEpisodes, dtype=int)
print("starting training without shielding")
for episode in range(0,amountOfEpisodes):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
            action = np.argmax(Q[state]) #1
            state2, reward, done, info = env.step(action) #2
            Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
            G += reward
            state = state2
#    if episode % 50 == 0:
#        print('Episode {} Total Reward: {}'.format(episode,G))
    graphData[episode] = G

print("training finished without shielding")

illegalDropoffStates = set()
for r in range(4+1):
    for c in range(4+1):
        p = 4
        for d in range(3+1):
            if (r,c) not in {(0,0),(0,4),(4,0),(4,3)}:
                illegalDropoffStates.add(((((r*5)+c)*5+p)*5)+d)



Q2 = np.zeros([env.observation_space.n, env.action_space.n])

graphData2 = np.zeros(amountOfEpisodes, dtype=int)

print("starting training with shielding")
for episode in range(0,amountOfEpisodes):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
            action = np.argmax(Q2[state]) #1
            if action==5 & state in illegalDropoffStates: # If dropoff, block
                indexRow = np.argsort(-Q2[state], axis=1)
                print(indexRow)
                for i in range(len(indexRow)):
                    if indexRow[i]==1:
                        print(i)
                        action = i


                # Copy row of Q matrix of current state
                # Make new row with indices that indicate the highest numbers in Q row
                # Select the index where a "1" is placed in this new index row (for loop met i en return i als value is 1) set action to i


            state2, reward, done, info = env.step(action) #2
            Q2[state,action] += alpha * (reward + np.max(Q2[state2]) - Q2[state,action]) #3
            G += reward
            state = state2
#    if episode % 50 == 0:
#        print('Episode {} Total Reward: {}'.format(episode,G))
    graphData2[episode] = G

print("training finished with shielding")


#Plot the non-shield and shield data
t = np.arange(0, amountOfEpisodes, 1)
plt.plot(t, graphData, '.')
plt.plot(t, graphData2, '.')

plt.title('Reward versus episodes with and without shielding')
plt.show()



