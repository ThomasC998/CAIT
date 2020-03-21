import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import math

env = gym.make('Taxi-v3')
G = 0
alpha = 0.618
amountOfEpisodes = 10000
amountOfSeries = 10

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




rewardMatrix1 = np.zeros((amountOfSeries, amountOfEpisodes))

for k in range(0,amountOfSeries):
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
    rewardMatrix1[k] = graphData


print("training finished without shielding")

# Make the illegal states for the shield
illegalDropoffStates = makeCodesDropoff(maxR, maxC, maxP, maxD)
illegalNorthWallStates = makeCodesNorthWall(maxR, maxC, maxP, maxD)
illegalSouthWallStates = makeCodesSouthWall(maxR, maxC, maxP, maxD)
illegalEastWallStates = makeCodesEastWall(maxR, maxC, maxP, maxD)
illegalWestWallStates = makeCodesWestWall(maxR, maxC, maxP, maxD)
#TODO: kan cleaner met bvb een dictionary met states als keys en actions als values (paren van 2)


rewardMatrix2 = np.zeros((amountOfSeries, amountOfEpisodes))

Q2 = np.zeros([env.observation_space.n, env.action_space.n])
graphData2 = np.zeros(amountOfEpisodes, dtype=int)

print("starting training with shielding")
for k in range(0, amountOfSeries):
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
                    #elif action == 0 and state in illegalSouthWallStates:
                     #   flags[0] = False
                      #  action = argMaxRandomIndexShielded(Q[state], flags)
                    #    illegalDetected = True
                    #elif action == 1 and state in illegalNorthWallStates:
                    #    flags[1] = False
                    #    action = argMaxRandomIndexShielded(Q[state], flags)
                    #    illegalDetected = True
                    #elif action == 2 and state in illegalEastWallStates:
                      #  flags[2] = False
                     #   action = argMaxRandomIndexShielded(Q[state], flags)
                     #   illegalDetected = True
                    #elif action == 3 and state in illegalWestWallStates:
                    #    flags[3] = False
                    #    action = argMaxRandomIndexShielded(Q[state], flags)
                    #    illegalDetected = True
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
    rewardMatrix2[k] = graphData2

print("training finished with shielding")

#average out rewardMatrixes
averageGraph1 = np.zeros(amountOfEpisodes, dtype=int)
for i in range(0, amountOfEpisodes-1):
    sum = 0
    for j in range(0, amountOfSeries-1):
        sum = sum + rewardMatrix1[j][i]
    averageGraph1[i] = sum / amountOfSeries

averageGraph2 = np.zeros(amountOfEpisodes, dtype=int)
for i in range(0, amountOfEpisodes-1):
    sum = 0
    for j in range(0, amountOfSeries-1):
        sum = sum + rewardMatrix2[j][i]
    averageGraph2[i] = sum / amountOfSeries


# PLOTS
plotNormal = False
plotExponential = True
plotPolynomial = False


#Plot the non-shield and shield data
t = np.arange(0, amountOfEpisodes, 1)
if plotNormal:
    plt.plot(t, averageGraph1, '-', label="Original Data")
    plt.plot(t, averageGraph2, '-',label="Original Data with shielding")

if plotExponential:
    # Exponential
    from scipy.optimize import curve_fit
    import sympy as sym

    plt.plot(t, averageGraph1, '.',label="Original Data")
    plt.plot(t, averageGraph2, '.',label="Original Data with shielding")

    t = np.array(t, dtype=float) #transform your data in a numpy array of floats
    averageGraph1 = np.array(averageGraph1, dtype=float) #so the curve_fit can work
    averageGraph2 = np.array(averageGraph2, dtype=float) #so the curve_fit can work


    def func(x, a, b, c):
        return -a*(math.e**(-b*x))+c

    popt1, pcov1 = curve_fit(func, t, averageGraph1)
    popt2, pcov2 = curve_fit(func, t, averageGraph2)


    print("a = %s , b = %s, c = %s" % (popt1[0], popt1[1], popt1[2]))
    print("a = %s , b = %s, c = %s" % (popt2[0], popt2[1], popt2[2]))

    """
    Use sympy to generate the latex sintax of the function
    """
    xs = sym.Symbol('\lambda')
    tex = sym.latex(func(xs,*popt1)).replace('$', '')
    tex2 = sym.latex(func(xs,*popt2)).replace('$', '')
    plt.title(r'$f(\lambda)= %s$' %(tex),fontsize=16)
    plt.title(r'$f(\lambda)= %s$' %(tex2),fontsize=16)

    """
    Print the coefficients and plot the funcion.
    """

    plt.plot(t, func(t, *popt1), label="Fitted Curve                           y=" + str(np.around(popt1[2],3)) + "-"+ str(np.around(popt1[0],3)) + "e^(-" + str(np.around(popt1[1],3)) +"x)") #same as line above \/
    plt.plot(t, func(t, *popt2), label="Fitted Curve with shielding    y=" + str(np.around(popt2[2],3)) + "-"+ str(np.around(popt2[0],3)) + "e^(-" + str(np.around(popt2[1],3)) +"x)") #same as line above \/
    #plt.plot(x, popt[0]*x**3 + popt[1]*x**2 + popt[2]*x + popt[3], label="Fitted Curve")

    plt.legend(loc='lower right')

if plotPolynomial:
    graad = 100

    # calculate polynomial 1
    z1 = np.polyfit(t, graphData, graad)
    f1 = np.poly1d(z1)

    # calculate new x's and y's
    x_new1 = np.linspace(t[0], t[-1], 50)
    y_new1 = f1(x_new1)

    plt.plot(t, graphData, '.', x_new1, y_new1, label="Fitted Curve without shield")
    plt.xlim([t[0] - 1, t[-1] + 1])

    # calculate polynomial 2
    z2 = np.polyfit(t, graphData2, graad)
    f2 = np.poly1d(z2)

    # calculate new x's and y's
    x_new2 = np.linspace(t[0], t[-1], 50)
    y_new2 = f2(x_new2)

    plt.plot(t, graphData2, '.', x_new2, y_new2, label="Fitted Curve with shield")
    plt.xlim([t[0] - 1, t[-1] + 1])

plt.title('Accumulated reward versus episodes with and without shielding (average over 10 series)')
plt.show()



