import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)

class Data:
    def __init__(self, classA=None, classB=None, shape=None, p={'A': 10, 'B': 20}, sepx={'A': 1.5, 'B': 0.0}, sepy={'A': 0.5, 'B': 0.5}, **kwargs):

        match shape:
            case 'double_arc':
                self.classA, self.classB = double_arc(**kwargs)
            case 'double_circle':
                self.classA, self.classB = double_circle(**kwargs)
            case None:
                self.classA = np.concatenate((np.random.randn(p['A'], 2) * 0.2 + [sepx['A'], sepy['A']], np.random.randn(p['A'], 2) * 0.2 + [-sepx['A'], sepy['A']])) if classA is None else classA
                self.classB = np.random.randn(p['B'], 2) * 0.2 + [sepx['B'], -sepy['B']] if classB is None else classB
        
        self.inputs = np.concatenate((self.classA, self.classB))
        self.targets = np.concatenate((np.ones(self.classA.shape[0]), -np.ones(self.classB.shape[0])))
        permute = list(range(self.inputs.shape[0]))
        random.shuffle(permute)
        self.inputs = self.inputs[permute, :]
        self.targets = self.targets[permute]
        self.writeToFile()

    def _regenerate(self):
        inputs = np.concatenate((self.classA, self.classB))
        targets = np.concatenate((np.ones(self.classA.shape[0]), -np.ones(self.classB.shape[0])))
        permute = list(range(inputs.shape[0]))
        random.shuffle(permute)
        self.inputs = inputs[permute, :]
        self.targets = targets[permute]
        self.writeToFile()

    def shift(self, classX=None, shift=0, axis=2):
        if axis == 0 or axis == 1:
            if classX == 'A':
                self.classA[:, axis] += shift
            elif classX == 'B':
                self.classB[:, axis] += shift
            else:
                self.classA[:, axis] += shift
                self.classB[:, axis] += shift
        else:
            if classX == 'A':
                self.classA += shift
            elif classX == 'B':
                self.classB += shift
            else:
                self.classA += shift
                self.classB += shift
        self._regenerate()

    def scale_variance(self, classX=None, factor=1):
        if classX == 'A':
            meanA = np.mean(self.classA, axis=0)
            self.classA = meanA + factor * (self.classA - meanA)
        elif classX == 'B':
            meanB = np.mean(self.classB, axis=0)
            self.classB = meanB + factor * (self.classB - meanB)
        else:
            meanA = np.mean(self.classA, axis=0)
            meanB = np.mean(self.classB, axis=0)
            self.classA = meanA + factor * (self.classA - meanA)
            self.classB = meanB + factor * (self.classB - meanB)
        self._regenerate()

    def scale(self, classX=None, scale=1, axis=2):
        if axis == 0 or axis == 1:
            if classX == 'A':
                self.classA[:, axis] *= scale
            elif classX == 'B':
                self.classB[:, axis] *= scale
            else:
                self.classA[:, axis] *= scale
                self.classB[:, axis] *= scale
        else:
            if classX == 'A':
                self.classA *= scale
            elif classX == 'B':
                self.classB *= scale
            else:
                self.classA *= scale
                self.classB *= scale
        self._regenerate()

    def rotate(self, classX=None, angle=0):
        theta = np.radians(angle)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        if classX == 'A':
            self.classA = np.dot(self.classA, rot)
        elif classX == 'B':
            self.classB = np.dot(self.classB, rot)
        else:
            self.classA = np.dot(self.classA, rot)
            self.classB = np.dot(self.classB, rot)
        self._regenerate()

    def mirror(self, classX=None, axis=2):
        if axis == 0:
            self.scale(classX=classX, scale=-1, axis=axis+1)
        elif axis == 1:
            self.scale(classX=classX, scale=-1, axis=axis-1)
        else:
            self.scale(classX=classX, scale=-1)
        self._regenerate()

    def randomize(self):
        shift = np.random.randint(-2, 2)
        axis = np.random.randint(3)
        classX = np.random.choice(['A', 'B', None])

        self.shift(classX=classX, shift=shift, axis=axis)

        factor = np.random.randint(1, 5) / np.random.choice(1, 5) * np.random.choice([-1, 1])
        axis = np.random.randint(3)
        classX = np.random.choice(['A', 'B', None])
        self.scale_variance(classX=classX, factor=factor)
        
        classX = np.random.choice(['A', 'B', None])
        angle = np.random.rand() * 360
        self.rotate(classX=classX, angle=angle)

        axis = np.random.randint(3)
        classX = np.random.choice(['A', 'B', None])
        self.mirror(axis=axis)

    def get_classA(self):
        return self.classA
    
    def get_classB(self):
        return self.classB

    def get_inputs(self):
        return self.inputs
    
    def get_targets(self):
        return self.targets
    
    def get_data(self):
        return (self.classA, self.classB, self.inputs, self.targets)

    def writeToFile(self):
        with open ('dataFile.txt', 'w') as f:
            f.write(str(self.classA.tolist()) + '\n')
            f.write(str(self.classB.tolist()) + '\n')
            f.write(str(self.inputs.tolist()) + '\n')
            f.write(str(self.targets.tolist()) + '\n')

    def plot(self, figure=None):
        plt.figure(np.random.randint(1, 1000)) if figure is None else plt.figure(figure)
        plt.plot([p[0] for p in self.classA], [p[1] for p in self.classA], 'b.', label='Class A')
        plt.plot([p[0] for p in self.classB], [p[1] for p in self.classB], 'r.', label='Class B')
        plt.axis('equal')
        plt.legend()
        plt.grid()

def double_arc(N=500, radius=1, thickness=1, seperation=0):
    classA = []
    classB = []

    origin = np.zeros(2)
    classA_center = origin - np.array([radius / 2, 0])
    classB_center = origin + np.array([radius / 2, 0])

    for _ in range(N):
        theta = np.random.rand() * 2 * np.pi

        r = np.random.rand() * thickness + radius - thickness / 2

        if theta < np.pi:
            x = r * np.cos(theta) + classA_center[0]
            y = r * np.sin(theta) + classA_center[1]
            classA.append([x, y])
        else:
            x = r * np.cos(theta) + classB_center[0]
            y = r * np.sin(theta) + classB_center[1]
            classB.append([x, y])

    classA = np.array(classA)
    classB = np.array(classB)

    classA[:, 1] += seperation
    classB[:, 1] -= seperation
    
    return classA, classB


def double_circle(N=500, radius=1, thickness=0.1, seperation=0.5):
    classA = []
    classB = []

    for _ in range(int(N / 2)):
        thetaA = np.random.rand() * 2 * np.pi
        rA = np.random.rand() * radius
        xA = rA * np.cos(thetaA)
        yA = rA * np.sin(thetaA)

        thetaB = np.random.rand() * 2 * np.pi
        rB = np.random.rand() * (radius + thickness + seperation) + seperation + radius
        xB = rB * np.cos(thetaB)
        yB = rB * np.sin(thetaB)

        classA.append([xA, yA])
        classB.append([xB, yB])

    classA = np.array(classA)
    classB = np.array(classB)

    return classA, classB

if __name__ == '__main__':

    data = Data('double_arc', 500, 1, 0.1, -0.2)
    data.plot()

    plt.show()





    
