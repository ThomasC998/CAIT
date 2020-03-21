import numpy as np
import matplotlib.pyplot as plt
import qlearning as ql


amountEpisodes = 300
amountSeries = 50
alpha = 0.618


rewardMatrix = np.zeros((amountSeries, amountEpisodes))
rewardMatrixShielding = np.zeros((amountSeries, amountEpisodes))
rewardMatrixRewardShaping = np.zeros((amountSeries, amountEpisodes))

for i in range(0, amountSeries):
    print('---Serie {} ---'.format(i))
    rewardMatrix[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes)
    rewardMatrixShielding[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, alpha = alpha, shielding=True)
    rewardMatrixRewardShaping[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, alpha = alpha, rewardShaping=True)


#average out rewardMatrixes
averageGraph1 = rewardMatrix.mean(0)
averageGraph2 = rewardMatrixShielding.mean(0)
averageGraph3 = rewardMatrixRewardShaping.mean(0)

plt.plot(averageGraph1)
plt.plot(averageGraph2)
plt.plot(averageGraph3)

plt.xlabel("Episodes")
plt.ylabel("Accumulated Reward")
plt.legend(('Regular QLearning', 'Shielding', 'Reward Shaping'))
plt.show()

