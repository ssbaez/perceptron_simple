import numpy as np
import random
import time

class Perceptron:
    def __init__(self,data,labels,learning_rate,bias):
        self.data = data
        self.labels = labels
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights = np.random.rand(self.data.shape[1])

    def __str__(self):
        return '==>los datos de entrenamiento son: \n{}\n==>Y deseada es: \n{}\n==>Los pesos son: \n{}\n==>el learning rate es: \n{}\n==>el bias es: \n{}\n'.format(self.data,self.labels,self.weights,self.learning_rate,self.bias)
    
    def __chgWeights(self,error,inputs):
        self.weights += self.learning_rate * error * inputs
    
    def train_desc(self):
        end = False
        y = None
        epoca = 1
        
        while end == False:

            i = 0
            print('\nepoca {}'.format(epoca))
            errores = np.random.rand(self.data.shape[0])

            for inputs,label in zip(self.data,self.labels):

                summation = np.dot(inputs,self.weights) # se redunda aqui con el fin de ejemplificar el proceso
                print('==>sumatoria {}: {}'.format(i,round(summation,2)))
                y = self.predict(inputs,self.weights)
                error = label - y
                errores[i] = error

                if error != 0:
                    self.__printInfo(y,label,error,inputs)
                    print('\nSe recalculan los pesos:')
                    print('alfa: {}'.format(self.learning_rate))
                    print('weights ant: {}'.format(self.weights))
                    self.__chgWeights(error,inputs)
                    print('weights new: {}'.format(self.weights))
                    break
                elif errores.sum() == 0:
                    self.__printInfo(y,label,error,inputs)
                    end = True
                    break
                else:
                    self.__printInfo(y,label,error,inputs)
                i += 1

            epoca += 1
    
    def predict(self,inputs,weights):
        summation = np.dot(inputs,weights)
        return 1 if summation > 0 else -1
    
    def __printInfo(self,y,label,error,inputs):
        print('y: {}'.format(y))
        print('y deseada: {}'.format(label))
        print('error: {}'.format(error))
        print('inputs: {}'.format(inputs))

    def train(self):
        end = False
        y = None
        epoca = 1

        star_time = time.time()

        while end == False:
            i = 0
            errores = np.random.rand(self.data.shape[0])
            for inputs,label in zip(self.data,self.labels):
                y = self.predict(inputs,self.weights)
                error = label - y
                errores[i] = error
                if error != 0:
                    self.__chgWeights(error,inputs)
                    break
                elif errores.sum() == 0:
                    end = True
                    break
                i += 1

            epoca += 1
        end_time = time.time()
        print('{} epocas'.format(epoca))
        print('tiempo de ejecucion: {:f}'.format(end_time - star_time))

# ZEN of python, explicit is better than implicit
ALFA = random.uniform(0,1)
BIAS = 1

# datos de entrenamiento AND
x = np.array([[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
y = np.array([-1,-1,-1,1])

# datos de entrenamiento OR
#x = np.array([[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
#y = np.array([-1,1,1,1])
    
y = Perceptron(x,y,ALFA,BIAS)
#print(y)
y.train_desc()
