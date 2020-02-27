import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import random

env = gym.make('Taxi-v3')
G = 0
alpha = 0.618
amountOfEpisodes = 1000

maxR = 4
maxC = 4
maxP = 4
maxD = 3

# HELPER FUNCTIONS

# Selects the max of an numPy array or a random value (index) if this array contains multiple maxima
def argMaxRandomIndex(array):
    maxIndex = np.argmax(array)
    maxValue = array[maxIndex]
    maxSet = set()
    for i in range(len(array)):
        if array[i] == maxValue:
            maxSet.add(i)
    return random.choice(list(maxSet))

# Selects the max of an numPy array or a random value (index) if this array contains multiple maxima with shield flags
def argMaxRandomIndexShieldedHandler(array, flags):
    maxIndex = np.argmax(array)
    if not flags[maxIndex]:  # index is blocked by shield
        max = -999999.0
        index = 0
        for i in range(len(array)):
            if array[i] > max and flags[i]:  # only select non-shielded actions
                max = array[i]
                index = i
        # Nu we de max index hebben gevonden, rekening gehouden met shield,
        # moeten we hieruit een random kiezen (als ze dezelfde maxValue hebben), zoals in de else block
        maxValue = array[index]
        maxSet = set()
        for i in range(len(array)):
            if array[i] == maxValue:
                maxSet.add(i)
        return random.choice(list(maxSet))
    else:
        maxValue = array[maxIndex]
        maxSet = set()
        for i in range(len(array)):
            if array[i] == maxValue:
                maxSet.add(i)
        return random.choice(list(maxSet))

def argMaxRandomIndexShielded(array, flags):
    index = argMaxRandomIndexShieldedHandler(array, flags)
    while not flags[index]:
        index = argMaxRandomIndexShieldedHandler(array, flags)
    return index

def makeCodesDropoff(maxR,maxC,maxP,maxD):
    codeSet = set()
    for r in range(maxR+1):
        for c in range(maxC+1):
            p = 4
            for d in range(maxD+1):
                if (r,c) not in {(0,0),(0,4),(4,0),(4,3)}:
                    codeSet.add(((((r*5)+c)*5+p)*4)+d)
    return codeSet

def makeCodesNorthWall(maxR, maxC, maxP, maxD):
    codeSet = set()
    for r in range(maxR + 1):
        for c in range(maxC + 1):
            for p in range(maxP + 1):
                for d in range(maxD + 1):
                    if (r, c) in {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)}:
                        codeSet.add(((((r * 5) + c) * 5 + p) * 4) + d)
    return codeSet

def makeCodesSouthWall(maxR, maxC, maxP, maxD):
    codeSet = set()
    for r in range(maxR + 1):
        for c in range(maxC + 1):
            for p in range(maxP + 1):
                for d in range(maxD + 1):
                    if (r, c) in {(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)}:
                        codeSet.add(((((r * 5) + c) * 5 + p) * 4) + d)
    return codeSet

def makeCodesWestWall(maxR, maxC, maxP, maxD):
    codeSet = set()
    for r in range(maxR + 1):
        for c in range(maxC + 1):
            for p in range(maxP + 1):
                for d in range(maxD + 1):
                    if (r, c) in {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (3, 1), (4, 1), (0, 2), (1, 2), (3, 3), (4, 3)}:
                        codeSet.add(((((r * 5) + c) * 5 + p) * 4) + d)
    return codeSet

def makeCodesEastWall(maxR, maxC, maxP, maxD):
    codeSet = set()
    for r in range(maxR + 1):
        for c in range(maxC + 1):
            for p in range(maxP + 1):
                for d in range(maxD + 1):
                    if (r, c) in {(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (3, 0), (4, 0), (0, 1), (1, 1), (3, 2), (4, 2)}:
                        codeSet.add(((((r * 5) + c) * 5 + p) * 4) + d)
    return codeSet

# END HELPER FUNCTIONS







Q = np.zeros([env.observation_space.n, env.action_space.n])
graphData = np.zeros(amountOfEpisodes, dtype=int)
print("starting training without shielding")
for episode in range(0,amountOfEpisodes):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
            action = argMaxRandomIndex(Q[state]) #1
            state2, reward, done, info = env.step(action) #2
            Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
            G += reward
            state = state2
    graphData[episode] = G

print("training finished without shielding")

# Make the illegal states for the shield
illegalDropoffStates = makeCodesDropoff(maxR, maxC, maxP, maxD)
illegalNorthWallStates = makeCodesNorthWall(maxR, maxC, maxP, maxD)
illegalSouthWallStates = makeCodesSouthWall(maxR, maxC, maxP, maxD)
illegalEastWallStates = makeCodesEastWall(maxR, maxC, maxP, maxD)
illegalWestWallStates = makeCodesWestWall(maxR, maxC, maxP, maxD)
#TODO: kan cleaner met bvb een dictionary met states als keys en actions als values (paren van 2)



Q2 = np.zeros([env.observation_space.n, env.action_space.n])
graphData2 = np.zeros(amountOfEpisodes, dtype=int)

print("starting training with shielding")
for episode in range(0,amountOfEpisodes):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
            action = argMaxRandomIndex(Q2[state]) #1
            # Sommige state/actie paren gaan nooit uitgevoerd worden dus die Q-waarde blijft altijd op 0,
            # De andere Q waarden worden negatief wanneer updated dus is de illegale state altijd de hoogste (0)
            # Zou normaal geen probleem mogen zijn als deze state/actie combinatie er altijd uitgefilterd wordt,
            # maar dit geeft wel veel stress op de if statements (worden altijd uitgevoerd wanneer deze state er is)

            # K heb het systeem waar we de Q2[state] afkapten als we een illegal actie hadden veranderd
            # naar een systeem me flags, omdat dit eerste te complex werd me walls erbij

            # t wilt nog altijd ni werken me de walls (convergeert naar -200)

            # Shield start
            flags = []
            flags[:6] = [True] * 6

            firstRun = True
            illegalDetected = False
            while (firstRun or illegalDetected):
                if action == 5 and state in illegalDropoffStates:  # If dropoff, block and choose 2nd best action
                    flags[5] = False  # can't use this action anymore in the future when getting max Q value
                    action = argMaxRandomIndexShielded(Q[state], flags)
                    illegalDetected = True
                elif action == 0 and state in illegalSouthWallStates:
                    flags[0] = False
                    action = argMaxRandomIndexShielded(Q[state], flags)
                    illegalDetected = True
                elif action == 1 and state in illegalNorthWallStates:
                    flags[1] = False
                    action = argMaxRandomIndexShielded(Q[state], flags)
                    illegalDetected = True
                elif action == 2 and state in illegalEastWallStates:
                    flags[2] = False
                    action = argMaxRandomIndexShielded(Q[state], flags)
                    illegalDetected = True
                elif action == 3 and state in illegalWestWallStates:
                    flags[3] = False
                    action = argMaxRandomIndexShielded(Q[state], flags)
                    illegalDetected = True
                else:
                    illegalDetected = False
                firstRun = False
            # print(action)
            # Shield end
            state2, reward, done, info = env.step(action) #2
            Q2[state,action] += alpha * (reward + np.max(Q2[state2]) - Q2[state,action]) #3
            G += reward
            state = state2
    graphData2[episode] = G

print("training finished with shielding")


#Plot the non-shield and shield data
t = np.arange(0, amountOfEpisodes, 1)
plt.plot(t, graphData, '.')
plt.plot(t, graphData2, '.')

plt.title('Reward versus episodes with and without shielding')
plt.show()



