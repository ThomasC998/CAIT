import gym
import numpy as np
import time

env = gym.make('Taxi-v3')
Q = np.zeros([env.observation_space.n, env.action_space.n])
G = 0
alpha = 0.618
print("starting training")
for episode in range(0,10000):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
            action = np.argmax(Q[state]) #1
            state2, reward, done, info = env.step(action) #2
            Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
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
        action = np.argmax(Q[state])
        state, reward, done, info = env.step(action) #2
        G += reward
        print(state)
        env.render()
    print('Episode {} Total Reward: {}'.format(episode,G))
    time.sleep(1)