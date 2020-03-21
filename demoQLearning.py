import gym
import numpy as np
import time

nbTrainingEpisodes = 400

env = gym.make('Taxi-v3')
Q = np.zeros([env.observation_space.n, env.action_space.n])
G = 0
alpha = 0.618
print("Starting training for {} episodes".format(nbTrainingEpisodes))
for episode in range(1,nbTrainingEpisodes+1):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
            action = np.argmax(Q[state]) #1
            state2, reward, done, info = env.step(action) #2
            Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
            G += reward
            state = state2   
    if episode % (nbTrainingEpisodes/10) == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))

print("training finished")
print("starting render")
time.sleep(5)


for episode in range(,30):
    done = False
    G = 0
    state = env.reset()
    env.render()
    time.sleep(2)
    while done != True:
        time.sleep(1)
        action = np.argmax(Q[state])
        state, reward, done, info = env.step(action) #2
        G += reward
        print(state)
        env.render()
    print('Episode {} Total Reward: {}'.format(episode,G))
    print()
    print()
    time.sleep(5)