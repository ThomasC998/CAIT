import gym

# env = gym.make('Taxi-v3')
# state = env.reset()
# counter = 0
# reward = None
# donecounter = 0
# while reward != 20:
#     state, reward, done, info = env.step(env.action_space.sample())
#     counter += 1
#     if (done): donecounter +=1

# print(counter)
# print(donecounter)

env = gym.make('Taxi-v3')

for episode in range(1,101):
    done = False
    G = 0
    state = env.reset()
    while done != True:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action) #2
        G += reward
    print('Episode {} Total Reward: {}'.format(episode,G))
