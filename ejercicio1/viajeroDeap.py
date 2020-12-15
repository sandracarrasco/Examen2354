# -*- coding: utf-8 -*-
"""


@author: SANDRA
"""

import random
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import algorithms



archivo = open("grafos.csv")
for linea in archivo:     
    datos = linea.split(";")    
    print(datos[0]+"\t" +datos[1]+"\t"+datos[2] +"\t" +datos[3])
      

Caminos=[[0,10,8,25],
    [10,0,4,29],
    [8,4,0,27],
    [25,29,27,0]]
cantidadCaminos= 4


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Generación de un camino aleatorio
toolbox.register("indices", random.sample, range(cantidadCaminos), cantidadCaminos) # aquí debemos registar una función que generar una muestra de individuo
#print(toolbox.indices())
# Generación de inviduos 
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
#generacion de la poblacion
toolbox.register("population", tools.initRepeat, list, toolbox.individual, 100) #


#Definimos la función objetivo

def funcobj(individual):
    
    # distancia entre el último elemento y el primero
    distancia = Caminos[individual[-1]][individual[0]]
    # distancia entre las ciudades
    for crom1, crom2 in zip(individual[0:-1], individual[1:]):
        distancia += Caminos[crom1][crom2]
    return distancia,

#con toolbox register registramos el cruce,la mutacion,la seleccion y la evaluacion

toolbox.register("mate", tools.cxOrdered)                       
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05) 
toolbox.register("select", tools.selTournament, tournsize=2)    
toolbox.register("evaluate", funcobj)



def main():
    random.seed(50) 
    pop = toolbox.population() # creamos la población inicial 
    hof = tools.HallOfFame(1) 
    #realizamos las estadisticas
    stats = tools.Statistics(lambda ind: ind.fitness.values) 
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    """realizamos el respectivo ajuste tenemos un 0.5 en el cruce ,un 0.2 en mutacion, 120 generaciones 
    """
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 120, stats=stats, halloffame=hof, verbose=True)
    return hof, log


hof, log = main()
print(log)

print("camino optimo: %s" %hof[0])

