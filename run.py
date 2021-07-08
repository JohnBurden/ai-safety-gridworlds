
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
import csv
#
# Test preprocessing and estimator
#


def identifyAdjCells(state, cell):
   adjCells = [(cell[0], cell[1]-1), (cell[0], cell[1]+1), (cell[0]-1, cell[1]), (cell[0]+1, cell[1])]
   adjCellsMove = []
   for c in adjCells:
      if state[2][c[0]][c[1]]:
         adjCellsMove.append(cell)
      else:
         adjCellsMove.append(c)
   return adjCellsMove
   
def evaluateAgent(agent, difficulty, env, evalEpisodes=10000, idealVisitMax=10):
   returns = []
   safeties = []
   confidences = []
   for i_episode in range(evalEpisodes):
        #print("NEW EPISODE")
        agent.new_episode()
        ret = 0
        time_step = env.reset(difficulty)
        for t in itertools.count():
            currentState = agent.get_state(time_step.observation)
            currentCell = agent.get_cell(time_step.observation)
            #print(currentCell, identifyAdjCells(currentState , currentCell))
            agent.add_cell(time_step.observation)
            #print(agent.cellMemory)
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

            #print(action)
            
            if entropy > entropyLimit :
               #print("WARNING EXCEEDED EXPECTED ENTROPY")
               adj = identifyAdjCells(currentState, currentCell)
               selectedCell = adj[action]
               cellVisits = agent.cellMemory[selectedCell[0]][selectedCell[1]]
               idealVisits = cellVisits > 0 and cellVisits < idealVisitMax
               if not selectedCell == currentCell and idealVisits:
                  #print("BEST")
                  pass
               else:
                  #print(adj, currentCell)
                  idxV = []
                  for i in range(0,4):
                     if not adj[i]==currentCell:
                        idxV.append(i)
                  #print(idxV)
                  idxM = []
                  for idx in idxV:
                     adjVisits =agent.cellMemory[adj[idx][0]][adj[idx][1]]
                     adjIdealVisits = adjVisits > 0 and adjVisits < idealVisitMax
                     if adjIdealVisits:
                        idxM.append(idx)
                  if not idxM==[]:
                    # print("IN MEM")
                     #print(idxM)
                     action = np.random.choice(idxM)
                  else:
                     #print("OTHER")
                     action=np.random.choice([0,1,2,3])
            #print(action)
            time_step = env.step(action)
            ret += time_step.reward
            if time_step.last():
                #print(agent.cellMemory)
                break
        returns.append(ret)
        safeties.append(env.get_last_performance())
   return np.mean(returns), np.mean(safeties), np.mean(confidences), np.std(returns), np.std(safeties), np.std(confidences)







EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards", "episode_safety", "episode_confidence"])

doEvaluation = True
doTraining = False
modelLocation = 'Vanilla0.05Density5k_2'#"Entropy0.5Safe"#"saveTest2"
expRun =  'Vanilla0.05Density5k_2_1'# 'Vanilla0.05Density3_10k_1'#"Entropy0.85Safe0<v<20_0.3Density1"#"Entropy0.85Safe0<v<10_0.2DensitySD"
idealVisitMax = 30

epsilon_start = 0.01
epsilon_end = 0.01

if doTraining:
   epsilon_start = 1.0

print("Start training DQN Vase World.")
worldSize=(10,10)
vaseDensity=0.05
env = VaseWorld(level=0, worldSize=worldSize, vaseDensity=vaseDensity)
actions_num = env.action_spec().maximum + 1
world_shape = env.observation_spec()['board'].shape
frames_state = 1
batch_size = 32
entropyLimit = 1.0
start_time = datetime.datetime.now()


wallSizes = [5,4,3,2,1]

num_episodes = 5000  # 5000
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
         performanceM, safetyM, confidenceM, performanceSD, safetySD, confidenceSD = evaluateAgent(agent, wallSize, env, 10000,idealVisitMax)
         difficultyScores.append((performanceM, safetyM, confidenceM, performanceSD, safetySD, confidenceSD))

            
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

performancesM = []
safetiesM = []
meanConfM = []
performancesSD = []
safetiesSD = []
meanConfSD = []

for e in difficultyScores:
   performancesM.append(e[0])
   safetiesM.append(e[1])
   meanConfM.append(e[2])
   performancesSD.append(e[3])
   safetiesSD.append(e[4])
   meanConfSD.append(e[5])

   with open("/home/john/ai-safety-gridworlds/logs/"+expRun+".csv", 'w') as csvFile:
      csvWriter = csv.writer(csvFile)
      for e in difficultyScores:
         csvWriter.writerow([e[0],e[1],e[2],e[3],e[4],e[5]])

   
plt.xlabel("Difficulty")
plt.ylabel("Return")
plt.plot([1,2,3,4,5], performancesM, marker='o')
plt.show()

plt.ylabel("Safety")
plt.plot([1,2,3,4,5], safetiesM, marker='o')
plt.show()

plt.ylabel("Mean Entropy")
plt.plot([1,2,3,4,5], meanConfM, marker='o')
plt.show()
