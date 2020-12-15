
import numpy as np
from sklearn.neural_network import MLPClassifier

import pandas as pd

datos=pd.read_csv("datos.csv")
archivo = open("datos.csv")
for linea in archivo:     
    datos = linea.split(",")    
    print(datos[1]+"\t" +datos[2]+"\t"+datos[7] +"\t" +datos[10]+"\t"+datos[13])
    
datos=pd.read_csv("test.csv")
archivo1 = open("test.csv")
for linea in archivo1:     
    test = linea.split(",")    
    print(test[1]+"\t" +test[2]+"\t"+test[7] +"\t" +test[10]+"\t"+test[13])


def divisiondatos(data):
    newaxis = (int)(len(data)*0.8)
    data_train = data[:newaxis]
    data_test = data[newaxis:]
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    for i in data_train:
        x_train.append(i[:-1])
        y_train.append(i[-1:][0])
    
    for i in data_test:
        x_test.append(i[:-1])
        y_test.append(i[-1:][0])
   

    x_train =np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return x_train, y_train, x_test, y_test 
def proceso():
  
    archivo = pd.read_csv("datos.csv", header = None)
    
    data = np.array(archivo)
    
    preparar = []
        
    for i in data:
        if not("?" in i):
            i[1] = float (i[1])
            i[13] = float(i[13])
            preparar.append(i)
      
    classatrib = {"+": 1, "-": 0}
    vals = [1 for i in range(len(preparar[0]))]
    max  = [0 for i in range(len(preparar[0]))]
    
    for i in range(len(preparar)):
        for j in range(len(preparar[0])):
            n = preparar[i][j]
            if isinstance(n, str):
                if n not in classatrib:
                    classatrib[n] = vals[j]
                    vals[j]+=1 
                preparar[i][j] = classatrib[n]
            else:
                if(n > max[j]):
                    max[j] = n
  
    for i in range(len(vals)):
        if vals[i] > 1:
            vals[i]-=1
        else:
            vals[i] = max[i]
    
    
    
    for i in range(len(preparar)):
        for j in range(len(preparar[0])):
            if(preparar[i][j] > 2):
                preparar[i][j] /= vals[j]
    
    return preparar

    




data = proceso()
x_train, y_train, x_test, y_test  = divisiondatos(data)

   

mlp = MLPClassifier(activation = "relu" ,solver = "adam", hidden_layer_sizes=(100, 100, 50,1), max_iter=300)
mlp.fit(x_train,y_train)
print("que tan cerca al 1")
print(mlp.score(x_train,y_train))
test = mlp.predict(x_test)

nuevo = 0

mc = np.zeros((2,2))
for i in range(len(y_test)):
    if(test[i] == y_test[i]):
        nuevo += 1
        mc[test[i]][y_test[i]] += 1
    else:
        mc[y_test[i]][test[i]] += 1

nuevo = (nuevo/len(test))*100
print("Prediccion: ","{:.2f}".format(nuevo)+"%")



        
