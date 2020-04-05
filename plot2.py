import numpy as np
import matplotlib.pyplot as plt
import qlearning as ql


amountEpisodes = 300
amountSeries = 50
alpha = 0.618


legend = list()


rewardMatrix3 = np.zeros((amountSeries, amountEpisodes))
for i in range(0, amountSeries):
    print('---Serie {} ---'.format(i))
    rewardMatrix3[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes)
averageGraph3 = rewardMatrix3.mean(0)    
plt.plot(averageGraph3)
legend.append('regular')

rewardMatrix4 = np.zeros((amountSeries, amountEpisodes))
for i in range(0, amountSeries):
    print('---Serie {} ---'.format(i))
    rewardMatrix4[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, illegalReward=-20)
averageGraph4 = rewardMatrix4.mean(0)    
plt.plot(averageGraph4)
legend.append('regular with -20')


rewardMatrix1 = np.zeros((amountSeries, amountEpisodes))
for i in range(0, amountSeries):
    print('---Serie {} ---'.format(i))
    rewardMatrix1[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, shielding =True)
averageGraph1 = rewardMatrix1.mean(0)    
plt.plot(averageGraph1)
legend.append('normal shield')


rewardMatrix2 = np.zeros((amountSeries, amountEpisodes))
for i in range(0, amountSeries):
    print('---Serie {} ---'.format(i))
    rewardMatrix2[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, shielding= True, illegalReward=-20)
averageGraph2 = rewardMatrix2.mean(0)    
plt.plot(averageGraph2)
legend.append('shield with -20')


plt.xlabel("Episodes")
plt.ylabel("Accumulated Reward")
plt.legend(legend)
plt.show()

