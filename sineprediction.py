import numpy as np

import matplotlib.pyplot as plt



def getPredictionValue(xVal,weights):
    orderOfPoly=len(weights)
    xV= [xVal**i for i in range(orderOfPoly)]
    return np.dot(xV,weights)

     
x = np.linspace(0,5,50)
y =  np.sin(x) + np.random.randn(50)*0.05 

     
w = [-2.00448672, -1.01326228, -0.20832782,  0.55992311, -0.09885708]

yPred = [getPredictionValue(x,w) for x in x]

plt.scatter(x,yPred,color='red')
plt.plot(x,yPred)

plt.plot(x,y)                              
plt.show()

lossArr = []
yPreditionArr = []
weightVec = np.random.randn(5)
LearningRate = 1e-8
epochs = 1000000
derivativeIncrement = 0.0000000001


n = float(len(x))



def squaredErrorloss(y,x,w):
  yPredicted = [getPredictionValue(x,w) for x in x]
  loss = sum((y-yPredicted)**2) / len(w)
  return loss,yPredicted


def takeDerivative(f,x,y,w,currentIndex,derivativeInc):
    d0,ypredicted = f(y,x,w)
    w[currentIndex] += derivativeInc
    d1,ypredicted = f(y,x,w)
    return ((d1-d0)/derivativeInc),ypredicted


for i in range(epochs): 

    D_m,ypredicted = takeDerivative(squaredErrorloss,x,y,weightVec,1,derivativeIncrement)
    D_m1,ypredicted = takeDerivative(squaredErrorloss,x,y,weightVec,2,derivativeIncrement)
    D_m2,ypredicted = takeDerivative(squaredErrorloss,x,y,weightVec,3,derivativeIncrement)
    D_m3,ypredicted = takeDerivative(squaredErrorloss,x,y,weightVec,4,derivativeIncrement)
    D_c,ypredicted =  takeDerivative(squaredErrorloss,x,y,weightVec,0,derivativeIncrement)
    weightVec[1] -=  LearningRate * D_m  
    weightVec[2] -=  LearningRate * D_m1
    weightVec[3] -= LearningRate * D_m2
    weightVec[4] -= LearningRate * D_m3
    weightVec[0] -=  LearningRate * D_c  
    loss,_ = squaredErrorloss(y,x,weightVec)
    lossArr.append(loss)
    if(i%1000 == 0):
     print("Loss at epoch {} is {}".format(i,loss))
print (weightVec)

plt.scatter(x,ypredicted,color='green')
plt.plot(x,ypredicted)

plt.plot(x,y)                              
plt.show()








