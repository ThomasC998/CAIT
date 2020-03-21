import numpy as np
import time
import matplotlib.pyplot as plt
import qlearning as ql


amountEpisodes = 500
amountSeries = 3



# Generate the data
rewardMatrix = np.zeros((amountSeries, amountEpisodes))
rewardMatrixShielding = np.zeros((amountSeries, amountEpisodes))
rewardMatrixRewardShaping = np.zeros((amountSeries, amountEpisodes))

for i in range(0, amountSeries):
    print('serie {}'.format(i))
    rewardMatrix[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes)
    rewardMatrixShielding[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, shielding=True)
    rewardMatrixRewardShaping[i] = ql.doQLearning(amountOfEpisodes = amountEpisodes, rewardShaping=True)


#average out rewardMatrixes
averageGraph1 = rewardMatrix.mean(0)
averageGraph2 = rewardMatrixShielding.mean(0)
averageGraph3 = rewardMatrixRewardShaping.mean(0)

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



