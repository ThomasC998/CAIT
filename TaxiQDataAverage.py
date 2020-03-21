import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import random

nb_rows = 5
nb_cols = 5
nb_locs = 4
locations = [(0, 0), (0, 4), (4, 0), (4, 3)]

amountEpisodes = 500
amountSeries = 20

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
for passenger in range(nb_locs + 1):
    for dest in range(nb_locs):
        if passenger == 4:  # passenger in taxi
            for row in range(nb_rows):
                for col in range(nb_cols):
                    state = encode(row, col, passenger, dest)
                    dist = abs(locations[dest][0] - row) + abs(locations[dest][1] - col)
                    Potential[state] = 18 - dist
        elif dest == passenger:  # passenger at destination (and unreachable states)
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


def doQLearning(shielding=False, rewardShaping=False, alpha=0.618, amountOfEpisodes=amountEpisodes):
    Q3 = np.zeros([env.observation_space.n, env.action_space.n])
    graphData3 = np.zeros(amountOfEpisodes, dtype=int)

    '''
    print("Starting training")
    print(" - Shielding: " + str(shielding))
    print(" - Reward Shaping: " + str(rewardShaping))
    print(" - Alpha: " + str(alpha))
    print(" - Nb Episodes: " + str(amountOfEpisodes))
    '''
    for episode in range(0, amountOfEpisodes):
        done = False
        G, reward = 0, 0
        state = env.reset()
        while done != True:
            action = argMaxRandomIndex(Q3[state])  # 1

            # Shielding
            if shielding and action == 5 and state in illegalDropoffStates:  # If dropoff, block
                reducedQArray = Q3[state][:5]
                action = argMaxRandomIndex(reducedQArray)

            state2, reward, done, info = env.step(action)  # 2

            # Reward Shaping
            if rewardShaping:
                reward += Potential[state2] - Potential[state]

            Q3[state, action] += alpha * (reward + np.max(Q3[state2]) - Q3[state, action])  # 3
            G += reward
            state = state2
        graphData3[episode] = G

    print(" done")
    return graphData3



# Generate the data
rewardMatrix = np.zeros((amountSeries, amountEpisodes))
rewardMatrixShielding = np.zeros((amountSeries, amountEpisodes))
rewardMatrixRewardShaping = np.zeros((amountSeries, amountEpisodes))

for i in range(0, amountSeries):
    print('serie {}'.format(i))
    rewardMatrix[i] = doQLearning()
    rewardMatrixShielding[i] = doQLearning(shielding=True)
    rewardMatrixRewardShaping[i] = doQLearning(rewardShaping=True)


# Plot the non-shield and shield data
#plt.plot(doQLearning(), ".")
#plt.plot(doQLearning(shielding=True), ".")
#plt.plot(doQLearning(rewardShaping=True), ".")



#average out rewardMatrixes
averageGraph1 = np.zeros(amountEpisodes, dtype=int)
for i in range(0, amountEpisodes-1):
    sum = 0
    for j in range(0, amountSeries-1):
        sum = sum + rewardMatrix[j][i]
    averageGraph1[i] = sum / amountSeries

averageGraph2 = np.zeros(amountEpisodes, dtype=int)
for i in range(0, amountEpisodes-1):
    sum = 0
    for j in range(0, amountSeries-1):
        sum = sum + rewardMatrixShielding[j][i]
    averageGraph2[i] = sum / amountSeries

averageGraph3 = np.zeros(amountEpisodes, dtype=int)
for i in range(0, amountEpisodes-1):
    sum = 0
    for j in range(0, amountSeries-1):
        sum = sum + rewardMatrixRewardShaping[j][i]
    averageGraph3[i] = sum / amountSeries

# Exponential curves
from scipy.optimize import curve_fit
import sympy as sym
import math

t = np.arange(0, amountEpisodes, 1)

plt.plot(t, averageGraph1, '.',label="Original Data")
plt.plot(t, averageGraph2, '.',label="Original Data with shielding")
plt.plot(t, averageGraph3, '.',label="Original Data with reward shaping")

t = np.array(t, dtype=float) #transform your data in a numpy array of floats
averageGraph1 = np.array(averageGraph1, dtype=float) #so the curve_fit can work
averageGraph2 = np.array(averageGraph2, dtype=float) #so the curve_fit can work
averageGraph3 = np.array(averageGraph3, dtype=float) #so the curve_fit can work


def func(x, a, b, c):
    return -a*(math.e**(-b*x))+c

popt1, pcov1 = curve_fit(func, t, averageGraph1)
popt2, pcov2 = curve_fit(func, t, averageGraph2)
popt3, pcov3 = curve_fit(func, t, averageGraph3)


print("a = %s , b = %s, c = %s" % (popt1[0], popt1[1], popt1[2]))
print("a = %s , b = %s, c = %s" % (popt2[0], popt2[1], popt2[2]))
print("a = %s , b = %s, c = %s" % (popt3[0], popt3[1], popt3[2]))

"""
Use sympy to generate the latex sintax of the function
"""
xs = sym.Symbol('\lambda')
tex = sym.latex(func(xs,*popt1)).replace('$', '')
tex2 = sym.latex(func(xs,*popt2)).replace('$', '')
tex3 = sym.latex(func(xs,*popt3)).replace('$', '')
plt.title(r'$f(\lambda)= %s$' %(tex),fontsize=16)
plt.title(r'$f(\lambda)= %s$' %(tex2),fontsize=16)
plt.title(r'$f(\lambda)= %s$' %(tex3),fontsize=16)

"""
Print the coefficients and plot the funcion.
"""

plt.plot(t, func(t, *popt1), label="Fitted Curve                           y=" + str(np.around(popt1[2],3)) + "-"+ str(np.around(popt1[0],3)) + "e^(-" + str(np.around(popt1[1],3)) +"x)") #same as line above \/
plt.plot(t, func(t, *popt2), label="Fitted Curve with shielding    y=" + str(np.around(popt2[2],3)) + "-"+ str(np.around(popt2[0],3)) + "e^(-" + str(np.around(popt2[1],3)) +"x)") #same as line above \/
plt.plot(t, func(t, *popt3), label="Fitted Curve with reward shaping  y=" + str(np.around(popt3[2],3)) + "-"+ str(np.around(popt3[0],3)) + "e^(-" + str(np.around(popt3[1],3)) +"x)") #same as line above \/
#plt.plot(x, popt[0]*x**3 + popt[1]*x**2 + popt[2]*x + popt[3], label="Fitted Curve")

plt.legend(loc='lower right')

plt.title('Reward versus episodes with/without shielding and reward shaping')
plt.show()



