
import random
import numpy as np
import multiprocessing
import pandas as pd
import pickle
import sys
from utils.env import *

def fitnness_function(repairing_sequence):
    inspected_list = []
    G, resil, node_resil_list = initial_situation(seed=seed)
    print(repairing_sequence)
    resilience = [resil]
    for action in repairing_sequence:
        _, resil, reward, done, node_resil_list = repairing_process(G, action=action,
                                                                    pre_resilience=resil,
                                                                    inspected_path=inspected_list)
        resilience.append(resil)
    return repairing_sequence, sum(resilience), resilience


def create_population(failure_pipes, Pop_size):
    population = []
    for i in range(0, Pop_size):
        population.append(random.sample(failure_pipes, len(failure_pipes)))
    return population


def rankPop(population):
    cores = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(processes=cores)
    fitness_results = pool.map(fitnness_function, population)
    pool.close()
    pool.join()
    fitness_array = np.array(fitness_results)
    fitness_results = fitness_array[np.argsort(fitness_array[:, 1])[::-1]][:, :2]
    resilience_lists = fitness_array[:, 2]
    return fitness_results, resilience_lists
    # return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(popRanked, columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def breed(parent1, parent2):
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    child = [0] * len(parent1)
    for i in range(startGene, endGene):
        child[i] = parent1[i]
    for element in parent2:
        if element not in child:
            child[child.index(0)] = element
        else:
            continue
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            pipe1 = individual[swapped]
            pipe2 = individual[swapWith]

            individual[swapped] = pipe2
            individual[swapWith] = pipe1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def load_pickle(opath):
    f = open(opath, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


if __name__ == '__main__':
    seed = 14

    def next_generation(currentGen):

        ranked, resilience_lists = rankPop(currentGen)
        selecion = selection(ranked, eliteSize=3)
        children = breedPopulation(selecion, eliteSize=5)
        nextgeneration = mutatePopulation(children, mutationRate=0.1)
        del children
        del selecion
        return ranked, nextgeneration, resilience_lists


    def save_file(path, file):
        with open(path, "wb") as f:
            pickle.dump(file, f)


    G = nx.read_gpickle('./data/road network graph2.gpickle')
    failure_pipes = list(G.edges)
    pop = create_population(failure_pipes, 10)

    resilience_progress = []
    resilience_action_progress = []
    for i in range(0, 150):
        ranked, pop, resilience_lists = next_generation(pop)
        print(ranked.shape)
        best_resil = ranked[0][1]
        best_action = ranked[0][0]
        resilience_progress.append(list(resilience_lists))
        resilience_action_progress.append(best_action)
        print('##this is the {} generation, the best resilience so far is {}'.format(i, best_resil))
    print("##the best ranked sequence so far is {}".format(ranked[0][0]))


    save_file('./results/random initial/GA resilience_seed {}.pkl'.format(seed), resilience_progress)
    save_file('./results/random initial/GA actions_seed {}.pkl'.format(seed), resilience_action_progress)
