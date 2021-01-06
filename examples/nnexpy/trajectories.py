class KnotDescriptor(object):
    def __init__(self, crossing, component, *args, **kwargs):
        self.crossing = crossing
        self.component = component
        self.trajectory = kwargs.get('trajectory', [])

    @classmethod
    def fromTemplate(self, template, *args, **kwargs):
        from .trajectories_utils import trefoil3D, trefoil4D, circular3D, circular4D, figureEight3D, figureEight4D
        if template == 'trefoil3D':
            return self.fromTrajectory(trefoil3D(*args, **kwargs))
        if template == "trefoil4D":
            return self.fromTrajectory(trefoil4D(*args, **kwargs))
        if template == "circular3D":
            return self.fromTrajectory(circular3D(*args, **kwargs))
        if template == "circular4D":
            return self.fromTrajectory(trefoil4D(*args, **kwargs))
        if template == "figureEight3D":
            return self.fromTrajectory(figureEight3D(*args, **kwargs))
        if template == "figureEight4D":
            return self.fromTrajectory(figureEight4D(*args, **kwargs))

    @classmethod
    def fromTrajectory(self, trajectory):
        from .trajectories_utils import intersect, vectorProduct, ABIsUp
        crossing = []
        component = []
        self.trajectory = trajectory
        for i in range(len(trajectory) - 1):
            for j in range(i+1, len(trajectory) - 1):
                if intersect(trajectory[i + 1], trajectory[i], trajectory[j + 1], trajectory[j]) and abs(i - j) > 2 and not (i == 0 and j == len(trajectory) - 2 or i == len(trajectory) - 2 and j == 0):
                    if vectorProduct(trajectory[i + 1], trajectory[i], trajectory[j + 1], trajectory[j]) > 0:
                        if not ABIsUp(trajectory[i + 1], trajectory[i], trajectory[j + 1], trajectory[j]):
                            crossing.append(('r', i, j, 'u'))
                        else:
                            crossing.append(('r', i, j, 'a'))
                    elif vectorProduct(trajectory[i + 1], trajectory[i], trajectory[j + 1], trajectory[j]) < 0:
                        if not ABIsUp(trajectory[i + 1], trajectory[i], trajectory[j + 1], trajectory[j]):
                            crossing.append(('l', i, j, 'u'))
                        else:
                            crossing.append(('l', i, j, 'a'))
        componentList = []
        for x in crossing:
            if (x[3] == 'a'):
                componentList.append(x[2])
            else:
                componentList.append(x[1])
        componentList.sort()
        for i in range(len(componentList) - 1):
            component.append((componentList[i] + 1, componentList[i+1]))
        if (len(component) > 1):
            component.append((componentList[-1] + 1, componentList[0]))
        else:
            component = [(0, len(trajectory) - 1)]

        return self(crossing, component, trajectory=trajectory)

    def findCrossingComponents(self, crossing):
        if not crossing in self.crossing:
            raise ValueError(
                "crossing must be one of the crossings of the knot")
        c1 = len(self.component) - 1
        c2 = 0
        c3 = 0
        for i, x in enumerate(self.component):
            if (crossing[1] == x[1] or crossing[2] == x[1]):
                c2 = i
                if i == len(self.component) - 1:
                    c3 = 0
                else:
                    c3 = i + 1
            if crossing[1] >= x[0] and crossing[1] < x[1] or crossing[2] >= x[0] and crossing[2] < x[1]:
                c1 = i

        return c1, c2, c3

    def determinant(self):
        import numpy as np
        import math as m
        if (len(self.crossing) == 0 or len(self.component) == 1):
            self.det = 1
            return 1
        matrix = np.zeros(
            (len(self.crossing), len(self.component)), dtype=int)
        for i, x in enumerate(self.crossing):
            c1, c2, c3 = KnotDescriptor.findCrossingComponents(self, x)
            matrix[i][c1] = 2
            matrix[i][c2] = -1
            matrix[i][c3] = -1
        print(matrix)
        matrix = np.delete(matrix, 0, 0)
        matrix = np.delete(matrix, 0, 1)

        res = int(np.round(abs(np.linalg.det(matrix))))
        self.det = res
        return res

    def plot2D(self):
        if len(self.trajectory) == 0:
            raise ValueError("no trajectory")
        import matplotlib.pyplot as plt
        plt.scatter([e[0] for e in self.trajectory], [e[1]
                                                      for e in self.trajectory], s=4, c=range(len(self.trajectory)), cmap="hsv")
        Lcrossing = []
        Rcrossing = []
        for y in self.crossing:
            if y[0] == 'r':
                Rcrossing.append(self.trajectory[y[1]])
            else:
                Lcrossing.append(self.trajectory[y[1]])
        plt.scatter([e[0] for e in Rcrossing], [e[1]
                                                for e in Rcrossing], s=200)
        plt.scatter([e[0] for e in Lcrossing], [e[1]
                                                for e in Lcrossing], s=200)

        plt.show()

    def plot3D(self):
        if len(self.trajectory) == 0:
            raise ValueError("no trajectory")
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        ax = plt.axes(projection="3d")
        ax.scatter3D([e[0] for e in self.trajectory], [e[1]
                                                       for e in self.trajectory], [e[2] for e in self.trajectory], s=4, c=range(len(self.trajectory)), cmap="hsv")
        Lcrossing = []
        Rcrossing = []
        for y in self.crossing:
            if y[0] == 'r':
                Rcrossing.append(self.trajectory[y[1]])
            else:
                Lcrossing.append(self.trajectory[y[1]])
        ax.scatter3D([e[0] for e in Rcrossing], [e[1]
                                                 for e in Rcrossing], [e[2] for e in Rcrossing], s=200)
        ax.scatter3D([e[0] for e in Lcrossing], [e[1]
                                                 for e in Lcrossing], [e[2] for e in Lcrossing], s=200)

        plt.show()
