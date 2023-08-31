import numpy
import numpy as np
import pandas as pd
import random
from numpy.random import randint
from numpy.random import rand
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import norm

# Finds the mean value of each sensor for every class
def FindMeanOfSensors(df):
    means = []
    for i in range(1, 6):
        filt = (df['class'] == i)
        temp = df.loc[filt].mean().values
        temp = temp[0:12]
        means.append(temp)
    return means

#Generates random initial population
def initial_population(n_pop, n_bit, n_crom,):
    pop = []
    for i in range(n_pop):
        temp = []
        for j in range(n_crom):
            number = random.getrandbits(n_bit)
            binary_string = format(number, '0b')
            temp.append(binary_string)
        pop.append(temp)
    return pop

#Converts binary to desimal
def FindDecimal(population, n_bit):
    decimal = []
    for i in range(len(population)):
        temp = int(population[i], 2)
        temp = temp / (pow(2, n_bit))
        temp = round(temp, 6)
        decimal.append(temp)
    return decimal

#evaluate population
def eval(population, MeanSensors, n_pop, n_bit, c):
    scores = []
    for i in range(n_pop):

        decimal = FindDecimal(population[i], n_bit)

        cosine = np.dot(decimal, MeanSensors[4]) / (norm(decimal) * norm(MeanSensors[4]))
        temp = 0
        for j in range(0, 4):
            temp = temp + (np.dot(decimal, MeanSensors[j]) / (norm(decimal) * norm(MeanSensors[j])))
        x = 1 - (1 / 4) * (temp)
        x = x * c
        score = (cosine + x) / (1 + c)
        scores.append(score)

    return scores

#Select parents with tournament selection
def Selection(population, scores, K=3):
    selection_ix = randint(len(population))
    for ix in randint(0, len(population), K - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]



#Uniform Crossover
def Uniform_crossover(p1, p2, r_cross, n_crom):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    cr1 = []
    cr2 = []
    chl1 = []
    chl2 = []
    # check for recombination
    if rand() < r_cross:
        for i in range(n_crom):
            cr1 = list(c1[i])
            cr2 = list(c1[i])
            P = np.random.rand(len(cr1))
            for j in range(len(P)-1):
                if P[j] < 0.3:
                    temp = cr1[j]
                    cr1[j] = cr2[j]
                    cr1[j] = temp
                x = ''.join(str(e) for e in cr1)
                y = ''.join(str(e) for e in cr1)
            chl1.append(x)
            chl2.append(y)
    else:
        chl1 = c1
        chl2 = c2

    return [chl1, chl2]

#mutation function
def mutation(K, r_mut):
    for i in range(len(K)):
        temp = list(K[i])
        for j in range(len(temp)):
            if rand() < r_mut:
               if temp[j] == '1':
                   temp[j] = '0'
               else:
                   temp[j] = '1'
            x = ''.join(str(e) for e in temp)
        K[i] = x

    return 0


def genetic_algorithm(n_bit, n_iter, n_pop, r_cross, r_mut, c, n_crom):
    #initial population
    population = initial_population(n_pop, n_bit, n_crom)

    best, best_eval = 0, 0

    for gen in range(n_iter):
        #evaluate population
        scores = eval(population, MeanSensors, n_pop, n_bit, c)
        #best, best_eval = 0, 0
        #find best score
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = population[i], scores[i]
                #print(">%d, new best f(%s) = %.3f" % (gen, population[i], scores[i]))
        #select parents
        selected = [Selection(population, scores) for _ in range(n_pop)]

        children = []

        for j in range(0, n_pop, 2):
            #get selected parents
            p1, p2 = selected[j], selected[j + 1]
            # crossover parents
            for k in Uniform_crossover(p1, p2, r_cross, n_crom):
                #mutation
                mutation(k, r_mut)

                children.append(k)
        # replace population
        population = children

    return [best, best_eval]




df = pd.read_csv("new_dataset.csv")


X = df.iloc[:, 0:12]

scaler = MinMaxScaler(feature_range=(0, 1))
df.iloc[:, 0:12] = scaler.fit_transform(X)
df.iloc[:, 0:12] = df.iloc[:, 0:12].round(6)


MeanSensors = FindMeanOfSensors(df)



n_iter = 100
n_pop = 200
n_crom = 12
n_bit = 20
c = 0.25
r_cross = 0.1
r_mut = 0.01

scores = []

for i in range(10):
    best, score = genetic_algorithm(n_bit, n_iter, n_pop, r_cross, r_mut, c, n_crom)
    print('Done:', i)
    print('f(%s) = %f' % (best, score))
    decimal = FindDecimal(best, n_bit)
    decimal = np.reshape(decimal, (1, -1))
    decimal = scaler.inverse_transform(decimal)
    Mean = np.reshape(MeanSensors[4], (1, -1))
    Mean = scaler.inverse_transform(Mean)
    print('Mean of sitting sensors')
    print(Mean)
    print('Best')
    print(decimal)
    scores.append(score)

print(score.mean())





