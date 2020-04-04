import numpy as np
import matplotlib.pyplot as plt
import qlearning as ql


amountEpisodes = 400
amountSeries = 5
alpha = 0.618

regularGraph = False
shieldingGraph = True
rewardshapeGraph = True
combinedGraph = False
legend = list()

if regularGraph:
    rewardMatrix = np.zeros((amountSeries, amountEpisodes))
    for i in range(0, amountSeries):
        print('---Serie {} ---'.format(i))
        rewardMatrix[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes)
    averageGraph1 = rewardMatrix.mean(0)    
    plt.plot(averageGraph1)
    legend.append('Regular QLearning')


if shieldingGraph:
    rewardMatrixShielding = np.zeros((amountSeries, amountEpisodes))
    for i in range(0, amountSeries):
        print('---Serie {} ---'.format(i))
        rewardMatrixShielding[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, alpha = alpha, shielding=True)
    averageGraph2 = rewardMatrixShielding.mean(0)
    plt.plot(averageGraph2)
    legend.append('Shielding')

if rewardshapeGraph:
    rewardMatrixRewardShaping = np.zeros((amountSeries, amountEpisodes))
    for i in range(0, amountSeries):
        print('---Serie {} ---'.format(i))
        rewardMatrixRewardShaping[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, alpha = alpha, rewardShaping=True)
    averageGraph3 = rewardMatrixRewardShaping.mean(0)   
    plt.plot(averageGraph3)
    legend.append('Reward Shaping')



if combinedGraph:
    rewardMatrixBoth = np.zeros((amountSeries, amountEpisodes))
    for i in range(0, amountSeries):
        print('---Serie {} ---'.format(i))
        rewardMatrixBoth[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, alpha = alpha, shielding=True, rewardShaping=True)
    averageGraph4 = rewardMatrixBoth.mean(0) 
    plt.plot(averageGraph4)
    legend.append('Shielding + Reward Shaping')


plt.xlabel("Episodes")
plt.ylabel("Accumulated Reward")
plt.legend(legend)
plt.show()

