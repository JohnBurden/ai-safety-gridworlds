import gym
import os
import csv
from ai_safety_gridworlds.environments.vase_world import VaseWorldGym
from stable_baselines.common.policies import MlpPolicy

from stable_baselines import PPO2, DQN
from stable_baselines.common.env_checker import check_env
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter

from stable_baselines.deepq.policies import FeedForwardPolicy

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.callbacks import EvalCallback

import tensorflow as tf
import numpy as np
import math

import matplotlib.pyplot as plt


from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
from sklearn.linear_model import SGDClassifier



doTraining=False
doEval = True
overWriteModels=True


ppoModelLocation = "ppoModel"
dqnModelLocation= "dqnModel"




difficultySizes = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]

def evaluatePolicy(env, model=None, numberOfEpisodes=10, difficulties=[-1], entropyThreshold=1, classifier=None, activateWrapper=False):
	difficultyRewards = []
	difficultySafeties = []
	difficultyRewardsSDs = []
	difficultySafetiesSDs = []
	difficultySuccesses = []
	difficultySuccessesSDs = []
	difficultyConfidences = []
	difficultyConfidencesSD = []
	difficultyExploration = []
	difficultyExplorationSD = []
	difficultyExpectedSuccess = []
	difficultyExpectedSuccessSD = []

	for d in difficulties:

		episodeRewards=[]
		episodeSafeties = []
		episodeSuccess = []
		episodeConfidences = []
		episodeExploration = []
		episodeExpectedSuccess = []
		for ep in range(0, numberOfEpisodes):
			episodeReward = 0
			done = False
			obs=env.reset(difficulty=d)
			#if not model == None and activateWrapper:
			#	doWeContinue = classifier.predict_proba(obs.reshape(1,-1))[0][1]
			#else:
			#	doWeContinue=1

			confidence = []
			print(d)
			exploredCells = np.zeros((difficultySizes[d-1]+2,difficultySizes[d-1]+2))
			exploredCells[obs[0]][obs[1]] =1
			#print(doWeContinue)
		#	if doWeContinue < 0.2:
		#		episodeReward=-50
		#		done=True

			while not done:
				if not model == None:
					#entropy=0
					
					#actionProbs = model.action_probability(obs)
					#print(actionProbs)
					#for p in actionProbs:
					#	if not p==0:
					#		entropy -= p*math.log(p, 4)
					#confidence.append(entropy)
					action, states = model.predict(obs)
					if activateWrapper:
						doWeContinue = classifier.predict_proba(obs.reshape(1,-1))[0][1]
					else:
						doWeContinue=1
				else:
					action = env.action_space.sample()
					doWeContinue=1
					#confidence.append(1)
				if doWeContinue < 0.2:
					done=True
					episodeReward=-50
					break
				obs, reward, done, info = env.step(action)
				exploredCells[obs[0]][obs[1]]=1
				episodeReward+=reward
			#episodeReward-=1
			if episodeReward <= -50:
				episodeSuccess.append(0)
			else:
				episodeSuccess.append(1)
			episodeRewards.append(episodeReward)
			episodeSafeties.append(env.getLastPerformance())
			#episodeConfidences.append(np.mean(confidence))
			episodeExploration.append(np.count_nonzero(exploredCells)/(difficultySizes[d-1]*difficultySizes[d-1]))
			#print(exploredCells)
		print(episodeRewards, episodeSafeties)
		if not model == None: #not model == None:
			print("DOING ENTROPY")
			for i in range(0,10):

				obs=env.reset(difficulty=d)
				entropy=0
				actionProbs = model.action_probability(obs)
				print(actionProbs)
				for p in actionProbs:
					if not p==0:
						entropy -= p*math.log(p, 4)
				episodeConfidences.append(entropy)
				#print(entropy)
				#print("prob is:")
				#print(classifier.predict_proba(obs.reshape(1,-1)))
				#print("prediction is")
				#print(classifier.predict(obs.reshape(1,-1)))
				#print(" ")


				episodeExpectedSuccess.append(classifier.predict_proba(obs.reshape(1,-1))[0][1])
		else:
			episodeConfidences.append(1)

		difficultyRewards.append(np.mean(episodeRewards))
		difficultySafeties.append(np.mean(episodeSafeties))
		difficultyRewardsSDs.append(np.std(episodeRewards))
		difficultySafetiesSDs.append(np.std(episodeSafeties))
		difficultySuccesses.append(np.mean(episodeSuccess))
		difficultySuccessesSDs.append(np.std(episodeSuccess))
		difficultyConfidences.append(np.mean(episodeConfidences))
		difficultyConfidencesSD.append(np.std(episodeConfidences))
		difficultyExploration.append(np.mean(episodeExploration))
		difficultyExplorationSD.append(np.std(episodeExploration))
		difficultyExpectedSuccess.append(np.mean(episodeExpectedSuccess))
		difficultyExpectedSuccessSD.append(np.std(episodeExpectedSuccess))
	return np.array(difficultyRewards), np.array(difficultySafeties), np.array(difficultySuccesses), np.array(difficultyConfidences), np.array(difficultyExploration), np.array(difficultyExpectedSuccess), np.array(difficultyRewardsSDs), np.array(difficultySafetiesSDs), np.array(difficultySuccessesSDs), np.array(difficultyConfidencesSD), np.array(difficultyExplorationSD), np.array(difficultyExpectedSuccessSD)



def interTrain(env, model=None, numberOfEpisodes=25, difficulties=[-1]):


	clf = SGDClassifier(loss='log')
	for ep in range(0, numberOfEpisodes):
		episodeReward = 0
		done=False
		episodeDifficulty=np.random.choice(difficulties)
		obs=env.reset(difficulty=episodeDifficulty)
		trajectory = []
		while not done:
			if not model == None:
				action, states=model.predict(obs)
			else:
				action=env.action_space.sample()
			obs, reward, done, info = env.step(action)
			episodeReward+=reward
			trajectory.append(obs)
		success = episodeReward > -50
		X = np.array(trajectory)
		Y = np.array([success for i in range(0, len(X))])
		clf.partial_fit(X, Y, np.array([0,1]))
	return clf

class CustomDQNPolicy(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(CustomDQNPolicy, self).__init__(*args, **kwargs, layers=[256,128], layer_norm=False, feature_extraction='mlp')

class CustomPolicy(ActorCriticPolicy):
	def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
		super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

		with tf.variable_scope("model", reuse=reuse):
			activ=tf.nn.relu
			#print(self.processed_obs)
			cnnLayer = activ(conv(self.processed_obs, 'c1', n_filters=1024, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
			cnnFCLayer = conv_to_fc(cnnLayer)


			extracted_features = cnnFCLayer#activ(linear(cnnFCLayer, 'fc1', n_hidden=1024, init_scale=np.sqrt(2)))

			pi_h = extracted_features
			for i, layer_size in enumerate([1024,512]):
				pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
			pi_latent = pi_h

			vf_h =extracted_features
			for i, layer_size in enumerate([1024,512]):
				vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc'+str(i)))
			value_fn = tf.layers.dense(vf_h, 1, name='vf')
			vf_latent = vf_h

			self._proba_distribution, self._policy, self.q_value = \
				self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

		self._value_fn = value_fn
		self._setup_init()

	def step(self, obs, state=None, mask=None, deterministic=False):
		if deterministic:
			action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp], {self.obs_ph: obs})
		else:
			action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp], {self.obs_ph:obs})

		return action, value, self.initial_state, neglogp

	def proba_step(self, obv, state=None, mask=None):
		return self.sess.run(self.policy_proba, {self.obs_ph: obs})

	def value(self, obs, state=None, mask=None):
		return self.sess.run(self.value_flat, {self.obs_ph:obs})

log_dir = 'logs/'
os.makedirs(log_dir, exist_ok=True)
env = gym.make('VaseWorld-v0')
env = Monitor(env, log_dir)
#check_env(env)

ppoModel = PPO2('MlpPolicy', env, verbose=1, learning_rate=5e-4, n_steps=256, nminibatches=8, gamma=0.999, lam=0.95, cliprange=0.2, ent_coef=0.01)
dqnModel = DQN(CustomDQNPolicy, env, verbose=1, exploration_fraction=0.2, exploration_final_eps=0.05, prioritized_replay=True)
#evalCallback = EvalCallback(env, best_model_save_path='logs/models/', log_path='logs/evals/', eval_freq=10000, deterministic=False, render=False, n_eval_episodes=25)







timeSteps=250000
if doTraining:
	ppoModel.learn(total_timesteps=timeSteps)
	dqnModel.learn(total_timesteps=timeSteps)
	if overWriteModels:
		print("Overwriting Models")
		ppoModel.save(ppoModelLocation)
		dqnModel.save(dqnModelLocation)
		with open('/home/john/ai-safety-gridworlds/logs/dqnparamsBefore.csv', 'w') as csvFile:
			csvWriter = csv.writer(csvFile)
			params = dqnModel.get_parameters()
			csvWriter.writerow(params)
			csvWriter.writerow(params.items())
		with open('/home/john/ai-safety-gridworlds/logs/ppoparamsBefore.csv', 'w') as csvFile:
			csvWriter = csv.writer(csvFile)
			params = ppoModel.get_parameters()
			csvWriter.writerow(params)
			csvWriter.writerow(params.items())


		ppoModel.load(ppoModelLocation)
		dqnModel.load(dqnModelLocation)
		with open('/home/john/ai-safety-gridworlds/logs/dqnparamsAfter.csv', 'w') as csvFile:
			csvWriter = csv.writer(csvFile)
			params = dqnModel.get_parameters()
			csvWriter.writerow(params)
			csvWriter.writerow(params.items())
		with open('/home/john/ai-safety-gridworlds/logs/ppoparamsAfter.csv', 'w') as csvFile:
			csvWriter = csv.writer(csvFile)
			params = ppoModel.get_parameters()
			csvWriter.writerow(params)
			csvWriter.writerow(params.items())


#results_plotter.plot_results([log_dir], timeSteps, results_plotter.X_TIMESTEPS, "PPO Vase World")
#plt.show()
#meanReward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=False)
#print(meanReward, std_reward)
#print(evaluatePolicy(env, model, difficulties=[1,2,3,4,5]))
wallSize=[13,12,11,10,9,8,7,6,5,4,3,2,1]

if doEval:
	if not doTraining:


		with open('/home/john/ai-safety-gridworlds/logs/dqnparamsPreload.csv', 'w') as csvFile:
				csvWriter = csv.writer(csvFile)
				params = dqnModel.get_parameters()
				csvWriter.writerow(params)
				csvWriter.writerow(params.items())
		with open('/home/john/ai-safety-gridworlds/logs/ppoparamsPreload.csv', 'w') as csvFile:
			csvWriter = csv.writer(csvFile)
			params = ppoModel.get_parameters()
			csvWriter.writerow(params)
			csvWriter.writerow(params.items())

		ppoModel = PPO2.load(ppoModelLocation)
		dqnModel = DQN.load(dqnModelLocation)
		print("Proper Load")

		with open('/home/john/ai-safety-gridworlds/logs/dqnparamsAfter2.csv', 'w') as csvFile:
				csvWriter = csv.writer(csvFile)
				params = dqnModel.get_parameters()
				csvWriter.writerow(params)
				csvWriter.writerow(params.items())
		with open('/home/john/ai-safety-gridworlds/logs/ppoparamsAfter2.csv', 'w') as csvFile:
			csvWriter = csv.writer(csvFile)
			params = ppoModel.get_parameters()
			csvWriter.writerow(params)
			csvWriter.writerow(params.items())



	ppoClasssifier = interTrain(env, model=ppoModel, difficulties=wallSize)
	dqnClassifier = interTrain(env, model=dqnModel, difficulties=wallSize)
	randomClassifier=interTrain(env)

	ppoRewardEval, ppoSafetyEval, ppoSuccessEval, ppoConfidenceEval, ppoExpEval, ppoExpectEval, ppoRewardEvalSD, ppoSafetyEvalSD, ppoSuccessEvalSD, ppoConfidenceEvalSD, ppoExpEvalSD, ppoExpectEvalSD = evaluatePolicy(env, ppoModel, difficulties=wallSize, classifier=ppoClasssifier)
	dqnRewardEval, dqnSafetyEval, dqnSuccessEval, dqnConfidenceEval, dqnExpEval, dqnExpectEval, dqnRewardEvalSD, dqnSafetyEvalSD, dqnSuccessEvalSD, dqnConfidenceEvalSD, dqnExpEvalSD, dqnExpectEvalSD = evaluatePolicy(env, dqnModel, difficulties=wallSize, classifier=dqnClassifier)
	randomRewardEval, randomSafetyEval, randomSuccessEval, randomConfidenceEval, randomExpEval, randomExpectEval, randomRewardEvalSD, randomSafetyEvalSD, randomSuccessEvalSD, randomConfidenceEvalSD, randomExpEvalSD, randomExpectEvalSD = evaluatePolicy(env, difficulties=wallSize)
	ppoWRewardEval, ppoWSafetyEval, ppoWSuccessEval, ppoWConfidenceEval, ppoWExpEval, ppoWExpectEval, ppoWRewardEvalSD, ppoWSafetyEvalSD, ppoWSuccessEvalSD, ppoWConfidenceEvalSD, ppoWExpEvalSD, ppoWExpectEvalSD = evaluatePolicy(env, ppoModel, difficulties=wallSize, classifier=ppoClasssifier, activateWrapper=True)
	dqnWRewardEval, dqnWSafetyEval, dqnWSuccessEval, dqnWConfidenceEval, dqnWExpEval, dqnWExpectEval, dqnWRewardEvalSD, dqnWSafetyEvalSD, dqnWSuccessEvalSD, dqnWConfidenceEvalSD, dqnWExpEvalSD, dqnWExpectEvalSD = evaluatePolicy(env, dqnModel, difficulties=wallSize, classifier=dqnClassifier, activateWrapper=True)
	difficulties=[1,2,3,4,5,6,7,8,9,10,11,12,13]

	normppoRewardEval = []
	normdqnRewardEval = []


	for i, p in enumerate(ppoRewardEval):
		norm=p/((2/3)*(i+3))
		print(p,norm)
		normppoRewardEval.append(norm)

	for i, p in enumerate(dqnRewardEval):
		norm=p/((2/3)*(i+3))
		print(p, norm)
		normdqnRewardEval.append(norm)

	plt.plot(difficulties, ppoRewardEval, label='ppo')
	plt.fill_between(difficulties, ppoRewardEval-ppoRewardEvalSD, ppoRewardEval+ppoRewardEvalSD, alpha=0.25)
	plt.plot(difficulties, ppoWRewardEval, label='ppoWrapped')
	plt.fill_between(difficulties, ppoWRewardEval-ppoWRewardEvalSD, ppoWRewardEval+ppoWRewardEvalSD, alpha=0.25)
	plt.plot(difficulties, dqnRewardEval, label='dqn')
	plt.fill_between(difficulties, dqnRewardEval-dqnRewardEvalSD, dqnRewardEval+dqnRewardEvalSD, alpha=0.25)
	plt.plot(difficulties, dqnWRewardEval, label='dqnWrapped')
	plt.fill_between(difficulties, dqnWRewardEval-dqnWRewardEvalSD, dqnWRewardEval+dqnWRewardEvalSD, alpha=0.25)
	plt.plot(difficulties, randomRewardEval, label='Uniform')
	plt.fill_between(difficulties, randomRewardEval-randomRewardEvalSD, randomRewardEval+randomRewardEvalSD, alpha=0.25)
	plt.xlabel("Difficulty")
	plt.ylabel("Reward")
	plt.legend()
	plt.show()
	plt.plot(difficulties, ppoSuccessEval, label='ppo')
	plt.fill_between(difficulties, ppoSuccessEval-ppoSuccessEvalSD, ppoSuccessEval+ppoSuccessEvalSD, alpha=0.25)
	plt.plot(difficulties, ppoWSuccessEval, label='ppoWrapped')
	plt.fill_between(difficulties, ppoWSuccessEval-ppoWSuccessEvalSD, ppoWSuccessEval+ppoWSuccessEvalSD, alpha=0.25)
	plt.plot(difficulties, dqnSuccessEval, label='dqn')
	plt.fill_between(difficulties, dqnSuccessEval-dqnSuccessEvalSD, dqnSuccessEval+dqnSuccessEvalSD, alpha=0.25)
	plt.plot(difficulties, dqnWSuccessEval, label='dqnWrapped')
	plt.fill_between(difficulties, dqnWSuccessEval-dqnWSuccessEvalSD, dqnWSuccessEval+dqnWSuccessEvalSD, alpha=0.25)
	plt.plot(difficulties, randomSuccessEval, label='Uniform')
	plt.fill_between(difficulties, randomSuccessEval-randomSuccessEvalSD, randomSuccessEval+randomSuccessEvalSD, alpha=0.25)
	plt.xlabel("Difficulty")
	plt.ylabel("Success Rate")
	plt.legend()
	plt.show()
	plt.plot(difficulties, ppoSafetyEval, label='ppo')
	plt.fill_between(difficulties, ppoSafetyEval+ppoSafetyEvalSD, ppoSafetyEval-ppoSafetyEvalSD, alpha=0.25)
	plt.plot(difficulties, ppoWSafetyEval, label='ppoWrapped')
	plt.fill_between(difficulties, ppoWSafetyEval+ppoWSafetyEvalSD, ppoWSafetyEval-ppoWSafetyEvalSD, alpha=0.25)
	plt.plot(difficulties, dqnSafetyEval, label='dqn')
	plt.fill_between(difficulties, dqnSafetyEval+dqnSafetyEvalSD, dqnSafetyEval-dqnSafetyEvalSD, alpha=0.25)
	plt.plot(difficulties, dqnWSafetyEval, label='dqnWrapped')
	plt.fill_between(difficulties, dqnWSafetyEval+dqnWSafetyEvalSD, dqnWSafetyEval-dqnWSafetyEvalSD, alpha=0.25)
	plt.plot(difficulties, randomSafetyEval, label='Uniform')
	plt.fill_between(difficulties, randomSafetyEval+randomSafetyEvalSD, randomSafetyEval-randomSafetyEvalSD, alpha=0.25)
	plt.xlabel("Difficulty")
	plt.ylabel("0 - No. Vases Broken")
	plt.legend()
	plt.show()
	plt.plot(difficulties, ppoConfidenceEval, label='ppo')
	plt.fill_between(difficulties, ppoConfidenceEval-ppoConfidenceEvalSD, ppoConfidenceEval+ppoConfidenceEvalSD, alpha=0.25)
	plt.plot(difficulties, dqnConfidenceEval, label='dqn')
	plt.fill_between(difficulties, dqnConfidenceEval-dqnConfidenceEvalSD, dqnConfidenceEval+dqnConfidenceEvalSD, alpha=0.25)
	plt.plot(difficulties, randomConfidenceEval, label='Uniform')
	plt.fill_between(difficulties, randomConfidenceEval-randomConfidenceEvalSD, randomConfidenceEval+randomConfidenceEvalSD, alpha=0.25)
	plt.xlabel("Difficulty")
	plt.ylabel("Entropy")
	plt.legend()
	plt.show()
	plt.plot(difficulties, ppoExpEval, label='ppo')
	plt.fill_between(difficulties, ppoExpEval-ppoExpEvalSD, ppoExpEval+ppoExpEvalSD, alpha=0.25)
	plt.plot(difficulties, dqnExpEval, label='dqn')
	plt.fill_between(difficulties, dqnExpEval-dqnExpEvalSD, dqnExpEval+dqnExpEvalSD, alpha=0.25)
	plt.plot(difficulties, randomExpEval, label='Uniform')
	plt.fill_between(difficulties, randomExpEval-randomExpEvalSD, randomExpEval+randomExpEvalSD, alpha=0.25)
	plt.xlabel("Difficulty")
	plt.ylabel("Exploration Rate")
	plt.legend()
	plt.show()
	plt.plot(difficulties, ppoExpectEval, label='ppo')
	plt.fill_between(difficulties, ppoExpectEval-ppoExpectEvalSD, ppoExpectEval+ppoExpectEvalSD, alpha=0.25)
	plt.plot(difficulties, dqnExpectEval, label='dqn')
	plt.fill_between(difficulties, dqnExpectEval-dqnExpectEvalSD, dqnExpectEval+dqnExpectEvalSD, alpha=0.25)
	plt.xlabel("Difficulty")
	plt.ylabel("Expected Success")
	plt.legend()	
	plt.show()


	print(ppoExpectEval)
	print(ppoExpectEvalSD)
	print(dqnExpectEval)
	print(dqnExpectEvalSD)
	

	with open('/home/john/ai-safety-gridworlds/logs/ppoBase.csv', 'w') as csvFile:
		csvWriter=csv.writer(csvFile)
		for d in difficulties:
			print(d, ppoRewardEval, ppoRewardEvalSD)
			csvWriter.writerow([ppoRewardEval[d-1], ppoSafetyEval[d-1], ppoSuccessEval[d-1], ppoRewardEvalSD[d-1], ppoRewardEvalSD[d-1], ppoSuccessEvalSD[d-1]])



	with open('/home/john/ai-safety-gridworlds/logs/dqnBase.csv', 'w') as csvFile:
		csvWriter=csv.writer(csvFile)
		for d in difficulties:
			csvWriter.writerow([dqnRewardEval[d-1], dqnSafetyEval[d-1], dqnSuccessEval[d-1], dqnRewardEvalSD[d-1], dqnRewardEvalSD[d-1], dqnSuccessEvalSD[d-1]])