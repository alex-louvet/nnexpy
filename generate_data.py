class DataDescriptor(object):
    """docstring for DataDescriptor."""

    def __init__(self, *args, **kwargs):
        import random as r
        import math as m
        bounds = kwargs.get('bounds',
                            Bounds({
                                'xmin': -1,
                                'xmax': 1,
                                'ymin': -1,
                                'ymax': 1
                            }))
        centerList = kwargs.get('centerList', None)
        radiusList = kwargs.get('radiusList', None)
        random = kwargs.get('random', None)
        nHoles = kwargs.get('nHoles', None)
        maxRadius = min(bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin)

        if random:
            r.seed(random)
        if not centerList and not nHoles and not radiusList:
            raise ValueError(
                'Either centerList, nHoles, or radiusList is required')
        if centerList and nHoles and nHoles != len(centerList):
            raise ValueError(
                "nHoles must be equal to the length of centerList")
        elif radiusList and nHoles and nHoles != len(radiusList):
            raise ValueError(
                "nHoles must be equal to the length of radiusList")
        elif centerList and radiusList and len(radiusList) != len(centerList):
            raise ValueError(
                "the length of radiusList must be equal to the length of centerList"
            )

        if not nHoles:
            if radiusList:
                nHoles = len(radiusList)
            elif centerList:
                nHoles = len(centerList)

        if not radiusList:
            radiusList = []
            for _ in range(nHoles):
                radiusList.append(r.random() * maxRadius / (nHoles))
        if not centerList:
            centerList = []
            for _ in range(nHoles):
                test = True
                while test:
                    test = False
                    temp = DataPoint({
                        'x':
                        (bounds.xmax - bounds.xmin) * r.random() + bounds.xmin,
                        'y': (bounds.ymax - bounds.ymin) * r.random() + bounds.ymin
                    })
                    if temp.x < bounds.xmin + radiusList[len(centerList)] or temp.y < bounds.ymin + radiusList[len(centerList)] or temp.x > bounds.xmax - radiusList[len(centerList)] or temp.y > bounds.ymax - radiusList[len(centerList)]:
                        test = True
                    else:
                        for i, point in enumerate(centerList):
                            if point.distanceTo(temp) <= radiusList[i] + radiusList[len(centerList)]:
                                test = True
                    if not test:
                        centerList.append(temp)

        self.bounds = bounds
        self.centerList = centerList
        self.radiusList = radiusList

    def plot(self):
        import math as m
        import matplotlib.pyplot as plt
        for k, center in enumerate(self.centerList):
            X = [center.x] + [center.x + self.radiusList[k] *
                              m.cos(i / 100 * 2.0 * m.pi) for i in range(100)]
            Y = [center.y] + [center.y + self.radiusList[k] *
                              m.sin(i / 100 * 2.0 * m.pi) for i in range(100)]
            plt.scatter(X, Y, marker='.')
            plt.plot()
        plt.axis('equal')
        plt.show()

    def generateData(self, *args, **kwargs):
        import math as m
        import random as r
        import numpy as np
        classNumber = kwargs.get('classNumber', 2)
        pointsNumber = kwargs.get('pointsNumber', 1000)
        random = kwargs.get('random', None)
        if random:
            r.seed(random)
        points = []
        if classNumber < 2:
            raise ValueError(
                'classNumber must be at least 2')

        for classIndex in range(1, classNumber):
            pointDistribution = np.array([radius for (i, radius) in enumerate(
                self.radiusList) if i % (classNumber - 1) == classIndex - 1])
            overallSquaredSum = np.sum(pointDistribution**2)
            pointDistribution = np.round(
                pointDistribution**2 * pointsNumber / overallSquaredSum)
            for i in range(len(pointDistribution)):
                center = self.centerList[i *
                                         (classNumber - 1) + (classIndex - 1)]
                radius = self.radiusList[i *
                                         (classNumber - 1) + (classIndex - 1)]
                for _ in range(int(pointDistribution[i])):
                    position = r.random()
                    radiusProp = r.random()
                    points.append(DataPoint({'x': center.x + radiusProp * radius * m.cos(position * 2.0 * m.pi),
                                             'y': center.y + radiusProp * radius * m.sin(position * 2.0 * m.pi), 'cluster': classIndex}))
        for _ in range(pointsNumber):
            test = False
            while not test:
                test = True
                temp = DataPoint({
                    'x':
                    (self.bounds.xmax - self.bounds.xmin) *
                    r.random() + self.bounds.xmin,
                    'y': (self.bounds.ymax - self.bounds.ymin) * r.random() + self.bounds.ymin,
                    'cluster': 0
                })
                for holeIndex in range(len(self.centerList)):
                    if temp.distanceTo(self.centerList[holeIndex]) <= self.radiusList[holeIndex]:
                        test = False
                if test:
                    points.append(temp)
        return DataInstance({'classNumber': classNumber, 'pointsNumber': pointsNumber, 'points': points})


class DataInstance(object):
    """docstring for DataInstance."""

    def __init__(self, arg):
        self.classNumber = arg['classNumber']
        self.pointsNumber = arg['pointsNumber']
        self.points = arg['points']

    def plot(self):
        import matplotlib.pyplot as plt
        for k in range(self.classNumber):
            X = [point.x for point in self.points if point.cluster == k]
            Y = [point.y for point in self.points if point.cluster == k]
            plt.scatter(X, Y, marker='.')
            plt.plot()
        plt.axis('equal')
        plt.show()


class DataPoint(object):
    """docstring for DataPoint."""

    def __init__(self, arg):
        self.x = arg['x']
        self.y = arg['y']
        if 'cluster' in arg:
            self.cluster = arg['cluster']

    def distanceTo(self, dataPoint):
        from math import sqrt
        return sqrt((dataPoint.x - self.x)**2 +
                    (dataPoint.y - self.y)**2)


class Bounds(object):
    """docstring for Bounds."""

    def __init__(self, arg):
        self.xmin = arg['xmin']
        self.xmax = arg['xmax']
        self.ymin = arg['ymin']
        self.ymax = arg['ymax']