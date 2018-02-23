import numpy as np
import matplotlib.pyplot as mp
from scipy import optimize


class Neural_Network(object):

	def __init__(self):
	
		#define hyperparameters
		
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3
		
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
		
		
		
	def forward(self, X):
		#propagate input through network
		self.z2 = np.dot(X, self.W1)
		
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat
		
		
	def sigmoid(self, z):
		
		return 1/(1+np.exp(-z))
		
	def sigmoidPrime(self, z):
	
		return np.exp(-z)/((1+np.exp(-z))**2)
		
	def costFunction(self, X, y):
	
		self.yHat = self.forward(X)
		
		return sum((y-self.yHat)**2)/2
		

	def costFunctionPrime(self, X, y):
		
		self.yHat = self.forward(X)
		
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		
		dJdW2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		
		dJdW1 = np.dot(X.T, delta2)
		
		return dJdW1, dJdW2
		
		#Help functions
		
	def getParams(self):
	
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		
		return params
		
	def setParams(self, params):
	
		W1_start = 0 
		W1_end = self.hiddenLayerSize*self.inputLayerSize
		
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize,self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize,self.outputLayerSize))

	def computeGradients(self, X, y):
	
		dJdW1, dJdW2 = self.costFunctionPrime(X, y)
		
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
		
def computeNumericalGradients(N, X, y):

	paramsInitial = N.getParams()
	
	numgrad = np.zeros(paramsInitial.shape)
	perturb = np.zeros(paramsInitial.shape)
	
	e = 1e-4
	
	for p in range(len(paramsInitial)):
		#set perturb vector
		
		perturb[p] = e
		
		N.setParams(paramsInitial + perturb)
		loss2 = N.costFunction(X,y)
		
		
		N.setParams(paramsInitial - perturb)
		loss1 = N.costFunction(X,y)
		
		#Numgrad
		
		numgrad[p] = (loss2 - loss1)/(2*e)
		
		#Returning to original values
		perturb[p] = 0
	
	N.setParams(paramsInitial)
		
	return numgrad
				
class trainer(object):

	def __init__(self, N):
		#make local reference to NN
		self.N = N
		
	def costFunctionWrapper(self, params, X, y):
		self.N.setParams(params)
		cost = self.N.costFunction(X, y)
		grad = self.N.computeGradients(X, y)
		return cost, grad
		
	def callBackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.X, self.y))
		
	options = {'maxIter' : 200, 'disp' : 'True'}
	
	
	def train(self, X, y):
		
		#Internal variables for callback
		self.X = X
		self.y = y
		
		#empty list to store costs
		
		self.J = []
		
		params0 = self.N.getParams()
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = 'BFGS', args=(X, y), options = {'maxIter' : 300, 'disp' : 'True'}, callback = self.callBackF)
		
		self.N.setParams(_res.x)
		self.optimizationResults = _res



NN = Neural_Network()

X = np.array([[3,5], 
[1,2],
[10,2]])

y = np.array([[75],
[82],
[93]])

y.astype('float')
X.astype('float')

yMax = np.amax(y)
xMax = np.amax(X)

X = np.true_divide(X, xMax)
y = np.true_divide(y, yMax)


##Chapter 5
# numgrad = computeNumericalGradients(NN, X, y)
# grad = NN.computeGradients(X, y)
# 
# 
# myError = np.linalg.norm(grad - numgrad)/np.linalg.norm(grad + numgrad)
# 
# print myError
##Chapter 5 END

T = trainer(NN)

T.train(X, y)

mp.plot(T.J)
mp.grid(1)
mp.ylabel('Cost')
mp.show()


# scalar = 3
# iterations = 1500
# 
# myCost = np.zeros(iterations)
# myValues = np.zeros([iterations, 3])
# 
# myTestData = np.ones([iterations, 3])
# myTestData[:,0] *= y[0]
# myTestData[:,1] *= y[1]
# myTestData[:,2] *= y[2]
# myTestData *= yMax
# 
# for i in range(0,iterations):
# 
# 	myCost[i] = NN.costFunction(X,y)
# 	yHat = NN.forward(X)
# 	myValues[i,0] = yHat[0]
# 	myValues[i,1] = yHat[1]
# 	myValues[i,2] = yHat[2]
# 	
# 	dJdW1, dJdW2 = NN.costFunctionPrime(X, y)
# 	NN.W1 = NN.W1 - scalar*dJdW1
# 	NN.W2 = NN.W2 - scalar*dJdW2
# 
print 'Optimized value: ', NN.forward(X).T*93
print 'Actual value: ', y.T*93
# 
# 
# #Plotting data
# mp.figure(1)
# mp.subplot(211)
# mp.title('Error')
# mp.plot(myCost, linewidth=2)
# mp.axis([0, iterations, 0, myCost[iterations/10]])
# mp.grid(1)
# mp.subplot(212)
# mp.title('Fitted values')
# 
# 
# fitted1, = mp.plot(myValues[:,0]*yMax, 'r--', label='Fitted values')
# fitted2, = mp.plot(myValues[:,1]*yMax, 'r--')
# fitted3, = mp.plot(myValues[:,2]*yMax, 'r--')
# 
# 
# testdata1, = mp.plot(myTestData[:,0], 'b-', label='Target values')
# testdata2, = mp.plot(myTestData[:,1], 'b-')
# testdata3, = mp.plot(myTestData[:,2], 'b-')
# 
# mp.axis([0, iterations, np.amin(myTestData)*0.9, np.amax(myTestData)*1.15])
# mp.legend([fitted1, testdata1],['Fitted data', 'Target values'] )
# mp.grid(1)
# mp.show()
