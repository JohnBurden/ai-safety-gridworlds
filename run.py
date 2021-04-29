
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
import math
#
# Test preprocessing and estimator
#


def evaluateAgent(agent, difficulty, env, evalEpisodes=10000):
   returns = []
   safeties = []
   confidences = []
   for i_episode in range(evalEpisodes):
        ret = 0
        time_step = env.reset(difficulty)
        for t in itertools.count():
            action = agent.act(time_step.observation)
            qVals = agent.q.predict(sess, np.expand_dims(agent.get_state(time_step.observation), 0))[0]
            qVals *= 7
            qVals = qVals + abs(np.min(qVals))
            e_qVals = np.exp(qVals - np.max(qVals))
            softmax = e_qVals/e_qVals.sum()
            entropy = 0
            for p in softmax:
               if not p==0:
                  entropy-=p*math.log(p,4)
            confidences.append(entropy)
           
            time_step = env.step(action)
            ret += time_step.reward
            if time_step.last():
                break
        returns.append(ret)
        safeties.append(env.get_last_performance())
   return np.mean(returns), np.mean(safeties), np.mean(confidences)







EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards", "episode_safety", "episode_confidence"])

doEvaluation = True
doTraining = False
modelLocation = "TrainEvalTest3"#"saveTest2"

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


wallSizes = [5,4,3,2,1]


num_episodes = 3000  # 5000
stats = EpisodeStats(episode_lengths=np.zeros(num_episodes),
                     episode_rewards=np.zeros(num_episodes),
                     episode_safety=np.zeros(num_episodes),
                     episode_confidence=np.zeros(num_episodes))

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
        # Save the current checkpoint
        if doTraining:
            agent.save()
        else:
           break
        ret = 0
        time_step = env.reset(np.random.choice([1,2,3,4,5]))  # for the description of timestep see ai_safety_gridworlds.environments.shared.rl.environment
        #print(time_step.observation)
        confidences = []
        for t in itertools.count():
            action = agent.act(time_step.observation)

            qVals = agent.q.predict(sess, np.expand_dims(agent.get_state(time_step.observation), 0))[0]
            qVals *= 7
            qVals = qVals + abs(np.min(qVals))
            e_qVals = np.exp(qVals - np.max(qVals))
            softmax = e_qVals/e_qVals.sum()
            entropy = 0
            for p in softmax:
               if not p==0:
                  entropy-=p*math.log(p,4)
            confidences.append(entropy)
            #print(qVals)
            time_step = env.step(action)
            if doTraining:
                loss = agent.learn(time_step, action)
            else:
                loss=None
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format( t, agent.total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            ret += time_step.reward
            if time_step.last():
                break
        stats.episode_lengths[i_episode] = t
        stats.episode_rewards[i_episode] = ret
        stats.episode_safety[i_episode] = env.get_last_performance()
        stats.episode_confidence[i_episode] = np.mean(confidences)
        if i_episode % 1 == 0:
            print("\nEpisode return: {}, and performance: {}.".format(ret, env.get_last_performance()))

    if doEvaluation:
      difficultyScores = []
      for wallSize in wallSizes:
         performance, safety, confidence = evaluateAgent(agent, wallSize, env)
         difficultyScores.append((performance, safety, confidence))

            
elapsed = datetime.datetime.now() - start_time
#print("Return: {}, elasped: {}.".format(ret, elapsed))
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

print(difficultyScores)

plt.show()

performances = []
safeties = []
meanConf = []

for e in difficultyScores:
   performances.append(e[0])
   safeties.append(e[1])
   meanConf.append(e[2])

plt.xlabel("Difficulty")
plt.ylabel("Return")
plt.plot([1,2,3,4,5], performances, marker='o')
plt.show()

plt.ylabel("Safety")
plt.plot([1,2,3,4,5], safeties, marker='o')
plt.show()

plt.ylabel("Mean Entropy")
plt.plot([1,2,3,4,5], meanConf, marker='o')
plt.show()
