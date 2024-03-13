import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import ast

class SVM:
    def __init__(self, kernel='linear', data=None, C = 1.0, sigma=1, degree=3, tol=1e-5, slack=True):
        self.data = data
        self.classA, self.classB, self.inputs, self.targets = data.get_data() if data is not None else data
        self.kernelstr = kernel
        self.kernel = self._choose_kernel(kernel)
        self.C = C if slack else None
        self.sigma = sigma
        self.degree = degree
        self.tol = tol
        self.slack = slack
        self.sv = []
        
        if data is not None:
            self.N = len(self.inputs)
            self.alpha = np.zeros(self.N)
            self.sep_alpha = np.zeros(self.N)
            self.b = 0
            self.B = [(0, self.C) for _ in range(self.N)]
            self.XC = {'type':'eq', 'fun':self._zerofun}
            self.P = self._generate_P()
        else:
            self.N = 0
            self.alpha = None
            self.sep_alpha = None
            self.b = None
            self.B = None
            self.XC = None
            self.P = None
        
    def _choose_kernel(self, kernel):
        if kernel == 'linear':
            return lambda x, y: np.dot(x, y)
        elif kernel == 'polynomial':
            return lambda x, y: (np.dot(x, y) + 1) ** self.degree
        elif kernel == 'RBF':
            return lambda x, y: math.exp(-np.linalg.norm(x - y)**2 / (2 * (self.sigma ** 2)))
        else:
            raise ValueError('Choice of kernel is not valid.')
        
    def _zerofun(self, alpha):
        return np.dot(alpha, self.targets)
    
    def _objective(self, alpha):
        return 0.5 * np.sum(np.outer(alpha, alpha) * self.P) - np.sum(alpha)
    
    def _generate_P(self):
        P = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                P[i, j] = self.targets[i] * self.targets[j] * self.kernel(self.inputs[i], self.inputs[j])
        return P
    
    def _calculate_b(self):
        ts = [self.sv[i][1] for i in range(len(self.sv))]
        for i in range(len(self.sv)):
            for s in self.sv:
                self.b += np.sum([s[2] * s[1] * self.kernel(s[0], self.sv[i][0])])
            self.b -= ts[i]
        if len(self.sv) > 0:
            self.b /= len(self.sv)
    
    def _indicator(self, x, y):
        return np.sum([s[2] * s[1] * self.kernel(s[0], np.array([x, y])) for s in self.sv]) - self.b
    
    def _extract_nonzero(self):
        for i in range(self.N):
            if self.tol <= self.alpha[i]:
                self.sep_alpha[i] = self.alpha[i]

    def _determine_sv(self):
        for i in range(self.N):
            if self.sep_alpha[i] > 0:
                self.sv.append((self.inputs[i], self.targets[i], self.sep_alpha[i]))

    def update(self, data):
        self.data = data
        self.classA, self.classB, self.inputs, self.targets = data.get_data() if data is not None else data
        self.sv = []
        self.N = len(self.inputs)
        self.alpha = np.zeros(self.N)
        self.sep_alpha = np.zeros(self.N)
        self.b = 0
        self.B = [(0, self.C) for _ in range(self.N)]
        self.XC = {'type':'eq', 'fun':self._zerofun}
        self.P = self._generate_P()
    
    def optimize(self):
        print('Initiating optimization...', end=' ')
        if self.inputs is None:
            print('No data to optimize.')
            return False
        
        ret = minimize(self._objective, np.zeros(self.N), bounds=self.B, constraints=self.XC)
        if not ret['success']:
            print('No solution found.')
            return False
        
        self.alpha = ret.x
        self._extract_nonzero()
        self._determine_sv()
        self._calculate_b()

        print('Optimization successful.')
        return True
    
    def set_kernel(self, kernel, kernel_params=None):
        self.kernelstr = kernel
        
        if kernel_params is not None:
            self.sigma = kernel_params['sigma'] if kernel_params.get('sigma') is not None else self.sigma
            self.degree = kernel_params['degree'] if kernel_params.get('degree') is not None else self.degree

        self.kernel = self._choose_kernel(kernel)
        self.update(self.data)

    def set_slack(self, slack=None, C=1.0):
        if slack is not None:
            self.slack = slack

        if self.slack is False:
            self.C = None
        else:
            self.C = C
        self.update(self.data)

    def plot_data(self):
        plt.plot([p[0] for p in self.classA], [p[1] for p in self.classA], 'b.', label='Class A')
        plt.plot([p[0] for p in self.classB], [p[1] for p in self.classB], 'r.', label='Class B')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()
    
    def plot_boundary(self):
        print('Generating decision boundary...', end=' ')
        plt.plot([p[0] for p in self.classA], [p[1] for p in self.classA], 'b.', label='Class A')
        plt.plot([p[0] for p in self.classB], [p[1] for p in self.classB], 'r.', label='Class B')
        plt.axis('equal')
        plt.autoscale(False)
        xgrid = np.linspace(-10, 10, 500)
        ygrid = np.linspace(-10, 10, 500)
        grid = np.array([[self._indicator(x, y) for x in xgrid] for y in ygrid])
        plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
        plt.grid()
        plt.legend()
        print('Done.')
        plt.show()

    def __repr__(self) -> str:
        return f'SVM properties:\nkernel={self.kernelstr}\nC={self.C}\nsigma={self.sigma}\ndegree={self.degree}\ntol={self.tol}\nslack={self.slack}'
    

if __name__ == '__main__':
    from datapy import Data
    data = Data(shape='double_arc', N=100, radius=1, thickness=1, seperation=-0.5)
    svm = SVM(data=data, kernel='polynomial', slack=False)

    print(svm)
    if svm.optimize():
        svm.plot_boundary()

    svm.set_slack(slack=True, C=10.0)
    print(svm)
    if svm.optimize():
        svm.plot_boundary()
