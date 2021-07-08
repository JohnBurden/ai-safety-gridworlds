
import csv
import matplotlib.pyplot as plt
import numpy as np

#exps = ['Entropy0.5Safe', 'Entropy0.5Safe0<v<5', 'Entropy0.5Safe0<v<10', 'Entropy0.5Safe0<v<20', 'Entropy0.75Safe0<v<10', 'Entropy0.85Safe0<v<10', 'Entropy0.85Safe0<v<20',  'VanillaDQN']
#labels = ['Wrapped 0.5 Entropy', 'Wrapped 0.5 Entropy 5 Visits', 'Wrapped 0.5 Entropy 10 Visits', 'Wraapped 0.5 Entropy 20 Visits', 'Wrapped 0.75 Entropy 10 Visits', 'Wrapped 0.85 Entropy 10 Visits',  'Wrapped Entropy 0.85 20 Visits', 'Vanilla DQN']

#exps = ['Vanilla0.3Density1_1','Vanilla0.3Density1_2', 'Vanilla0.3Density1_3', 'Vanilla0.3Density2_1','Vanilla0.3Density2_2', 'Vanilla0.3Density2_3', 'Vanilla0.3Density3_1','Vanilla0.3Density3_2', 'Vanilla0.3Density3_3']
#labels = ['Vanilla DQN 1_1', 'Vanilla DQN 1_2', 'Vanilla DQN 1_3', 'Vanilla DQN 2_1', 'Vanilla DQN 2_2', 'Vanilla DQN 2_3', 'Vanilla DQN 1_1', 'Vanilla DQN 1_2', 'Vanilla DQN 1_3']

#exps = ['Vanilla0.05Density1_10k_1', 'Vanilla0.05Density2_10k_1', 'Vanilla0.05Density3_10k_1', 'Entropy0.35Safe0<v<999_0.05Density1_1']
#labels= ["DQN 1", "DQN 2", "DQN 3", 'Wrapped 0.85 Entropy']

exps=['ppoBase', 'dqnBase']
labels = ["ppo", "dqn"]

performances = []
safeties=[]
meanConf= []

allPerformances = []
allSafeties=[]
allMeanConf=[]
allPerformancesSD = []
allSafetiesSD = []
allMeanConfSD = []

for expRun in exps:
    with open("/home/john/ai-safety-gridworlds/logs/"+expRun+".csv") as csvFile:

        performances=[]
        safeties=[]
        meanConf=[]
        performancesSD = []
        safetiesSD = []
        meanConfSD = []
        csvReader = csv.reader(csvFile)
        for row in csvReader:
            print(row)
            performances.append(float(row[0]))
            safeties.append(float(row[1]))
            #meanConf.append(float(row[2]))

            performancesSD.append(float(row[2]))
            safetiesSD.append(float(row[3]))
            #meanConfSD.append(float(row[5]))
        allPerformances.append(performances)
        allSafeties.append(safeties)
        #allMeanConf.append(meanConf)
        allPerformancesSD.append(performancesSD)
        allSafetiesSD.append(safetiesSD)
        #allMeanConfSD.append(meanConfSD)

            
   
plt.xlabel("Difficulty")
plt.ylabel("Raw Return")
for i,  perf in enumerate(allPerformances):
    normalizedPerfs = []
    normalizedSDs = []
    for j, p in enumerate(perf):
        normP = (p)/((2/3)*(j+3))
        normalizedPerfs.append(p)
        normSD = allPerformancesSD[i][j]/((2/3)*(j+3))
        normalizedSDs.append(allPerformancesSD[i][j])
        print(normP, p)
    normalizedPerfsArray = np.array(normalizedPerfs)
    d = np.array(normalizedSDs)
    plt.plot([1,2,3,4,5,6,7], normalizedPerfs , marker='o', label=labels[i])
    plt.fill_between([1,2,3,4,5,6,7], normalizedPerfsArray-d, normalizedPerfsArray+d, alpha=0.25)
#plt.ylim(-5,0)
plt.legend()
plt.show()


#safeSDs = np.s
plt.xlabel("Difficulty")
plt.ylabel("Average Number of Vases Broken Per Episode")
for i, safe in enumerate(allSafeties):
    plt.plot([1,2,3,4,5,6,7], safe, marker='o', label=labels[i])
    plt.fill_between([1,2,3,4,5,6,7], np.array(safe)-np.array(allSafeties[i]), np.array(safe)+np.array(allSafeties[i]), alpha=0.25)
plt.legend()
plt.show()

#plt.xlabel("Difficulty")
#plt.ylabel("Mean Entropy")
#for i, conf in enumerate(allMeanConf):
#    confA = np.array(conf)
#    sd = np.array(allMeanConfSD[i])
#    plt.plot([1,2,3,4,5], conf, marker='o', label=labels[i])
#    plt.fill_between([1,2,3,4,5], confA-sd, confA+sd, alpha=0.25)
#plt.legend()
#plt.show()
