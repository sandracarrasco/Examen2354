# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import array
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from operator import attrgetter

#Maximizar=1 minimizar=-1

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#tipo individuo 
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
#operaciones
toolbox = base.Toolbox()
#Funcion que se llena el individuo

# Attribute generator  ## genera numeros al radom entre 0 y 1
toolbox.register("attr_bool", random.randint, 0, 1)
##generacion de los individuos y poblacion
# Structure initializers
# se  va a crear un individual que es un array de manera repetitiva con un tamaño de 100 con el atributo booleano0 1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
# la poblacion se repite respecto el individuo en una lista 
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#funcion objetivo

#---------------------*********---------------------------
#PROCEDIMIENTO DEL ALGORITMO GENETICO SIN LIBRERIA
def funcionfx(individual):
    decimal = int("".join(map(str,individual)),2)
    return decimal*decimal*decimal+decimal*decimal+decimal,

def cxdos(num1, num2):
 
    size = min(len(num1), len(num2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    num1[cxpoint1:cxpoint2], num2[cxpoint1:cxpoint2] \
        = num2[cxpoint1:cxpoint2], num1[cxpoint1:cxpoint2]

    return num1, num2

def mutacion(individual, indpb):
    
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])

    return individual,

def seleccionar(individuals, k, tournsize, fit_attr="fitness"):
    
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen

def selRandom(individuals, k):

    return [random.choice(individuals) for i in range(k)]


toolbox.register("evaluate", funcionfx)
toolbox.register("mate", cxdos)
toolbox.register("mutate", mutacion, indpb=0.05)
toolbox.register("select", seleccionar, tournsize=2)

def main():

    random.seed(20)
    
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof

if __name__ == "__main__":
    main()
