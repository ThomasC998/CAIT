import gym
import numpy as np
import time
import random

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
            if array[i] == maxValue and flags[i]:
                maxSet.add(i)
        return random.choice(list(maxSet))
    else:
        maxValue = array[maxIndex]
        maxSet = set()
        for i in range(len(array)):
            if array[i] == maxValue and flags[i]:
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

# Make the illegal states for the shield
illegalDropoffStates = makeCodesDropoff(maxR, maxC, maxP, maxD)
illegalNorthWallStates = makeCodesNorthWall(maxR, maxC, maxP, maxD)
illegalSouthWallStates = makeCodesSouthWall(maxR, maxC, maxP, maxD)
illegalEastWallStates = makeCodesEastWall(maxR, maxC, maxP, maxD)
illegalWestWallStates = makeCodesWestWall(maxR, maxC, maxP, maxD)

env = gym.make('Taxi-v3')
Q = np.zeros([env.observation_space.n, env.action_space.n])
G = 0
alpha = 0.618
print("starting training")
for episode in range(0,1000):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
        action = argMaxRandomIndex(Q[state])  # 1
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
        state2, reward, done, info = env.step(action)  # 2
        Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action])  # 3
        G += reward
        state = state2
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))

print("training finished")

for episode in range(0,30):
    done = False
    G = 0
    state = env.reset()
    env.render()
    while done != True:
        time.sleep(0.2)
        action = argMaxRandomIndex(Q[state])
        # Shield start
        flags = []
        flags[:6] = [True] * 6

        firstRun = True
        illegalDetected = False
        while(firstRun or illegalDetected):
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
        print("=========================")
        print(Q[state])
        print(flags)
        # Shield end
        print(action)
        print(state)
        print("=========================")
        state, reward, done, info = env.step(action) #2
        G += reward
        env.render()
    print('Episode {} Total Reward: {}'.format(episode,G))
    time.sleep(1)