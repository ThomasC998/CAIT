import numpy as np
import matplotlib.pyplot as plt
import qlearning as ql


amountEpisodes = 300
amountSeries = 3
alpha = 0.618


legend = list()


rewardMatrix1 = np.zeros((amountSeries, amountEpisodes))
for i in range(0, amountSeries):
    print('---Serie {} ---'.format(i))
    rewardMatrix1[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes)
averageGraph1 = rewardMatrix1.mean(0)    
plt.plot(averageGraph1)
legend.append('-10')


rewardMatrix2 = np.zeros((amountSeries, amountEpisodes))
for i in range(0, amountSeries):
    print('---Serie {} ---'.format(i))
    rewardMatrix2[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes,illegalReward=-20)
averageGraph2 = rewardMatrix2.mean(0)    
plt.plot(averageGraph2)
legend.append('-20')



plt.xlabel("Episodes")
plt.ylabel("Accumulated Reward")
plt.legend(legend)
plt.show()

