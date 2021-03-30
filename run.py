
import numpy as np
from agents.dqn import DQNAgent, Estimator, StateProcessor
import tensorflow as tf
from ai_safety_gridworlds.environments.vase_world import VaseWorld
import itertools
import os
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque, namedtuple
import datetime

#
# Test preprocessing and estimator
#


EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards", "episode_safety"])

doTraining = False
modelLocation = "8x8OneWallOrTwo"#"saveTest2"

epsilon_start = 0.01
epsilon_end = 0.01

if doTraining:
   epsilon_start = 1.0

print("Start training DQN Vase World.")
worldSize=(10,10)
env = VaseWorld(level=0, worldSize=worldSize)
actions_num = env.action_spec().maximum + 1
world_shape = env.observation_spec()['board'].shape
frames_state = 1
batch_size = 32

start_time = datetime.datetime.now()



num_episodes = 4000  # 5000
stats = EpisodeStats(episode_lengths=np.zeros(num_episodes),
                     episode_rewards=np.zeros(num_episodes),
                     episode_safety=np.zeros(num_episodes))

tf.compat.v1.reset_default_graph()
with tf.Session() as sess:
    agent = DQNAgent(sess,
                 world_shape,
                 int(actions_num),
                 env,
                 frames_state=frames_state,
                 experiment_dir=modelLocation,
                 replay_memory_size=20000,  # 10000
                 replay_memory_init_size=3000,  # 3000
                 update_target_estimator_every=1000,  # 500
                 discount_factor=0.99,
                 epsilon_start=epsilon_start,
                 epsilon_end=epsilon_end,
                     epsilon_decay_steps=250000,
                     batch_size=batch_size, worldSize=worldSize)
    
    for i_episode in range(num_episodes):
        if i_episode==3000:
            doTraining=False
        # Save the current checkpoint
        if doTraining:
            agent.save()
        #else:
         #   agent.total_t=200000
        #print(env.GAME_ART)
        ret = 0
        time_step = env.reset()  # for the description of timestep see ai_safety_gridworlds.environments.shared.rl.environment
        #print(time_step.observation)
        for t in itertools.count():
            #print(time_step.observation)
            #if i_episode > 3000:
            #    print(agent.get_state(time_step.observation))
            action = agent.act(time_step.observation)
            #if i_episode > 3000:
            #    print("Action: "+ str(action))
            time_step = env.step(action)
            #if i_episode<5000:
            if doTraining:
                loss = agent.learn(time_step, action)
            #loss = -1
            else:
                loss=None
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                        t, agent.total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            ret += time_step.reward
            #print()
            #print(action, time_step.reward)
            if time_step.last():
                break
        stats.episode_lengths[i_episode] = t
        stats.episode_rewards[i_episode] = ret
        stats.episode_safety[i_episode] = env.get_last_performance()
        #print("Safety: ")
        #print(env.get_overall_performance())
        if i_episode % 1 == 0:
            print("\nEpisode return: {}, and performance: {}.".format(ret, env.get_last_performance()))


elapsed = datetime.datetime.now() - start_time
print("Return: {}, elasped: {}.".format(ret, elapsed))
print("Training Finished")
print("Episode rewards:")

finalRewards = pd.Series(stats.episode_rewards)
movAv = pd.Series.rolling(finalRewards, window=100, center=False).mean()
print(movAv)
plt.plot(movAv)

finalSafety = pd.Series(stats.episode_safety)
movAvSafety = pd.Series.rolling(finalSafety, window=100, center=False).mean()
print(finalSafety)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.ylim(-250,0)
plt.show()

plt.ylabel("Safety")
plt.plot(movAvSafety)

plt.show()
