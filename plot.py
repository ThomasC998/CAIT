import numpy as np
import matplotlib.pyplot as plt
import qlearning as ql
import time

amountEpisodes = 300
amountSeries = 50
alpha = 0.618

regularGraph = True
shieldingGraph = True
rewardshapeGraph = True
combinedGraph = True

legend = list()

t0 = time.process_time()

if regularGraph:
    rewardMatrix = np.zeros((amountSeries, amountEpisodes))
    for i in range(0, amountSeries):
        print('---Serie {} ---'.format(i))
        rewardMatrix[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, alpha = alpha)
    averageGraph1 = rewardMatrix.mean(0)    
    plt.plot(averageGraph1)
    legend.append('Regular QLearning')

t1 = time.process_time()

if shieldingGraph:
    rewardMatrixShielding = np.zeros((amountSeries, amountEpisodes))
    for i in range(0, amountSeries):
        print('---Serie {} ---'.format(i))
        rewardMatrixShielding[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, alpha = alpha, shielding=True)
    averageGraph2 = rewardMatrixShielding.mean(0)
    plt.plot(averageGraph2)
    legend.append('Shielding')

t2 = time.process_time()

if rewardshapeGraph:
    rewardMatrixRewardShaping = np.zeros((amountSeries, amountEpisodes))
    for i in range(0, amountSeries):
        print('---Serie {} ---'.format(i))
        rewardMatrixRewardShaping[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, alpha = alpha, rewardShaping=True)
    averageGraph3 = rewardMatrixRewardShaping.mean(0)   
    plt.plot(averageGraph3)
    legend.append('Reward Shaping')

t3 = time.process_time()

if combinedGraph:
    rewardMatrixBoth = np.zeros((amountSeries, amountEpisodes))
    for i in range(0, amountSeries):
        print('---Serie {} ---'.format(i))
        rewardMatrixBoth[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, alpha = alpha, shielding=True, rewardShaping=True)
    averageGraph4 = rewardMatrixBoth.mean(0) 
    plt.plot(averageGraph4)
    legend.append('Shielding + Reward Shaping')

t4 = time.process_time()

print('Time1: ', (t1-t0))
print('Time2: ', (t2-t1))
print('Time3: ', (t3-t2))
print('Time4: ', (t4-t3))

plt.xlabel("Episodes")
plt.ylabel("Accumulated Reward")
plt.legend(legend)
plt.show()