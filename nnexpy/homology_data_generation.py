class DataDescriptor(object):
    def __init__(self, *args, **kwargs):
        import random as r
        import math as m
        if "dimension" in kwargs and "bounds" in kwargs:
            if (kwargs.get('bounds').dimension != kwargs.get('dimension')):
                raise ValueError(
                    'bounds and data must live in the same dimension')
            else:
                self.bounds = kwargs.get('bounds')
                self.dimension = kwargs.get('dimension')
        elif "dimension" in kwargs:
            self.dimension = kwargs.get('dimension')
            self.bounds = Bounds(dimension=self.dimension)
        elif "bounds" in kwargs:
            self.bounds = kwargs.get('bounds')
            self.dimension = self.bounds.dimension
        else:
            self.dimension = 2
            self.bounds = Bounds(dimension=2)

        centerList = kwargs.get('centerList', None)
        if centerList:
            for x in centerList:
                if len(x.coordinates) != self.dimension:
                    raise ValueError(
                        'all center must live in the data dimension')
        radiusList = kwargs.get('radiusList', None)
        maxStrata = kwargs.get('maxStrata', 1)
        minStrata = kwargs.get('minStrata', 1)
        random = kwargs.get('random', None)
        nHoles = kwargs.get('nHoles', None)
        maxRadius = min([x['max'] - x['min']
                         for x in self.bounds.boundsCoordinates])

        self.holeDimension = kwargs.get(
            'holeDimension', [self.dimension] * nHoles)
        self.orientation = kwargs.get('orientation', None)

        if random:
            r.seed(random)
        if not centerList and not nHoles and not radiusList:
            raise ValueError(
                'Either centerList, nHoles, or radiusList is required')
        if centerList and nHoles and nHoles != len(centerList):
            raise ValueError(
                "nHoles must be equal to the length of centerList")
        if nHoles != len(self.holeDimension):
            raise ValueError(
                "hole dimensions must be given for no or all holes")
        for x in self.holeDimension:
            if x > self.dimension:
                raise ValueError(
                    'a sphere dimension can not be greater than the dimension of the space')

        if not nHoles:
            if radiusList:
                nHoles = len(radiusList)
            elif centerList:
                nHoles = len(centerList)

        if not radiusList:
            radiusList = []
            for k in range(nHoles):
                temp = []
                start = 0
                if self.holeDimension[k] < 0:
                    temp.append(0)
                    temp.append(r.random() * maxRadius / (2 * (nHoles)))
                    start = 1
                    self.holeDimension[k] = -1 * self.holeDimension[k]
                for _ in range(start, r.randint(minStrata, maxStrata)):
                    temp.append(r.random() * maxRadius / (2 * (nHoles)))
                    temp.append(r.random() * maxRadius / (2 * nHoles))
                temp.sort()
                radiusList.append([(temp[2 * i], temp[2 * i + 1])
                                   for i in range(len(temp) // 2)])
        if not centerList:
            centerList = []
            for _ in range(nHoles):
                test = True
                while test:
                    test = False
                    temp = []
                    for i in range(self.dimension):
                        temp.append(
                            (self.bounds.boundsCoordinates[i]['max'] - self.bounds.boundsCoordinates[i]['min']) * r.random() + self.bounds.boundsCoordinates[i]['min'])
                    temp = DataPoint(coordinates=temp,
                                     dimension=self.dimension)
                    radius = radiusList[len(centerList)][-1][1]
                    for i in range(len(temp.coordinates)):
                        bounds = self.bounds.boundsCoordinates[i]
                        if temp.coordinates[i] < bounds['min'] + radius or temp.coordinates[i] > bounds['max'] - radius:
                            test = True
                        else:
                            for i, point in enumerate(centerList):
                                if point.distanceTo(temp) <= radiusList[i][-1][1] + radiusList[len(centerList)][-1][1]:
                                    test = True
                    if not test:
                        centerList.append(temp)

        self.centerList = centerList
        self.radiusList = radiusList

        if self.orientation:
            if len(self.orientation) != len(self.centerList):
                raise(ValueError('Invalid length for orientation got {} expected {}'.format(
                    len(self.orientation), len(self.centerList))))
            for i, x in enumerate(self.holeDimension):
                if len(self.orientation[i]) != self.dimension - abs(x):
                    raise ValueError('Invalid orientation got {} expected length {}'.format(
                        self.orientation[i], self.dimension - abs(x)))
                for y in self.orientation[i]:
                    if y >= self.dimension:
                        raise ValueError(
                            'Orientation value represents unchanged coordinates and must be smaller than data dimension ({}), got {}'.format(self.dimension, y))

        else:
            orientation = []
            for _ in range(len(self.centerList)):
                holeOrientation = []
                for _ in range(self.dimension - abs(self.holeDimension[i])):
                    random = r.randint(0, len(temp) - 1)
                    a = temp.pop(random)
                    holeOrientation.append(a)
                orientation.append(holeOrientation)
            self.orientation = orientation

    def generateTestData(self, *args, **kwargs):
        import random as r
        import numpy as np
        nPoints = kwargs.get('nPoints', 1000)
        points = []
        for _ in range(nPoints):
            temp = []
            for i in range(self.dimension):
                temp.append(
                    (self.bounds.boundsCoordinates[i]['max'] - self.bounds.boundsCoordinates[i]['min']) * r.random() + self.bounds.boundsCoordinates[i]['min'])
            points.append(DataPoint(
                coordinates=temp,
                cluster=0
            ))
        return DataInstance({'dimension': self.dimension, 'classNumber': 1, 'nPoints': nPoints, 'points': points})

    def plot(self):
        import math as m
        import matplotlib.pyplot as plt
        import random as r
        if self.dimension > 3:
            raise ValueError(
                'No display possible for dimension higher than 3')
        elif self.dimension == 2:
            plt.clf()
            for k, center in enumerate(self.centerList):
                for radius in self.radiusList[k]:
                    X = [center.coordinates[0] + (r.random() * (radius[0] - radius[1]) + radius[1]) *
                         m.cos(i / 100 * 2.0 * m.pi) for i in range(10000)]
                    Y = [center.coordinates[1] + (r.random() * (radius[0] - radius[1]) + radius[1]) *
                         m.sin(i / 100 * 2.0 * m.pi) for i in range(10000)]
                    plt.scatter(X, Y, marker='.')
                    plt.plot()
            plt.axis('equal')
            plt.show()
        else:
            from mpl_toolkits import mplot3d
            import numpy as np
            import random as r
            ax = plt.axes(projection="3d")
            for k, center in enumerate(self.centerList):
                X, Y, Z = [], [], []
                for radius in self.radiusList[k]:
                    for _ in range(1000):
                        radiusProp = r.random()
                        radiusLength = (
                            radiusProp * radius[0] + (1 - radiusProp) * radius[1])
                        temp = []
                        for _ in range(len(center.coordinates)):
                            a = r.random()
                            temp.append(2 * a - 1)
                        randomVector = np.array(temp)
                        randomVector = (
                            radiusLength / np.sqrt(np.sum(randomVector ** 2))) * randomVector
                        point = center.coordinates + randomVector
                        X.append(point[0])
                        Y.append(point[1])
                        Z.append(point[2])
                    ax.scatter3D(X, Y, Z, marker='.', s=[1] * len(X))
            plt.show()

    def generateData(self, *args, **kwargs):
        import math as m
        import random as r
        import numpy as np
        classNumber = kwargs.get('classNumber', 2)
        nPoints = kwargs.get('nPoints', 1000)
        random = kwargs.get('random', None)
        if random:
            r.seed(random)
        points = []
        if classNumber < 2:
            raise ValueError(
                'classNumber must be at least 2')

        for classIndex in range(1, classNumber):
            pointDistribution = []
            for (i, radius) in enumerate(self.radiusList):
                if i % (classNumber - 1) == classIndex - 1:
                    pointDistribution.append(
                        sum([x[1]**self.dimension - x[0]**self.dimension for x in radius]))
            pointDistribution = np.array(pointDistribution)
            overallSquaredSum = np.sum(pointDistribution)
            pointDistribution = np.round(
                pointDistribution * nPoints / overallSquaredSum)
            for i in range(len(pointDistribution)):
                center = np.array(self.centerList[i *
                                                  (classNumber - 1) + (classIndex - 1)].coordinates)
                temp = [i for i in range(self.dimension)]
                dimHole = [j in self.orientation[i]
                           for j in range(self.dimension)]

                holeDimension = self.holeDimension[i]

                radius = self.radiusList[i *
                                         (classNumber - 1) + (classIndex - 1)]
                for _ in range(int(pointDistribution[i])):
                    stratum = r.choice(radius)
                    radiusProp = r.random()**(1/abs(holeDimension))
                    radiusLength = (
                        radiusProp * stratum[1] + (1 - radiusProp) * stratum[0])
                    temp = []
                    for i in range(len(center)):
                        if not dimHole[i]:
                            a = r.random()
                            temp.append(2 * a - 1)
                        else:
                            temp.append(0)
                    randomVector = np.array(temp)
                    randomVector = (
                        radiusLength / np.sqrt(np.sum(randomVector ** 2))) * randomVector
                    points.append(
                        DataPoint(coordinates=center + randomVector, cluster=classIndex))

        for _ in range(nPoints):
            test = False
            while not test:
                test = True
                temp = []
                for i in range(self.dimension):
                    temp.append(
                        (self.bounds.boundsCoordinates[i]['max'] - self.bounds.boundsCoordinates[i]['min']) * r.random() + self.bounds.boundsCoordinates[i]['min'])
                temp = DataPoint(coordinates=temp,
                                 dimension=self.dimension)
                for holeIndex in range(len(self.centerList)):
                    for stratum in self.radiusList[holeIndex]:
                        if temp.distanceTo(self.centerList[holeIndex]) <= stratum[1] and temp.distanceTo(self.centerList[holeIndex]) >= stratum[0]:
                            test = False
                if test:
                    points.append(temp)
        return DataInstance({'classNumber': classNumber, 'nPoints': nPoints, 'points': points, 'dimension': self.dimension})


class DataInstance(object):
    def __init__(self, arg):
        self.classNumber = arg['classNumber']
        self.nPoints = arg['nPoints']
        self.points = arg['points']
        self.dimension = arg['dimension']

    def plot(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        import random as r
        from .homology_utils import selectRandomSublist
        nPoints = kwargs.get('nPoints', self.nPoints)
        noBack = kwargs.get('noBack', False)
        if noBack:
            begin = 1
        else:
            begin = 0
        if self.dimension > 3:
            raise ValueError(
                'No display possible for dimension higher than 3')
        elif self.dimension == 2:
            plt.close()
            for k in range(begin, self.classNumber):
                pointList = selectRandomSublist(
                    [point for point in self.points if point.cluster == k], nPoints)
                X = [point.coordinates[0]
                     for point in pointList]
                Y = [point.coordinates[1]
                     for point in pointList]
                plt.scatter(X, Y, marker='.', s=[0.3] * len(X))
                plt.plot()
            plt.axis('equal')
            plt.show()
        else:
            from mpl_toolkits import mplot3d
            import numpy as np
            ax = plt.axes(projection="3d")
            for k in range(begin, self.classNumber):
                pointList = selectRandomSublist(
                    [point for point in self.points if point.cluster == k], nPoints)
                X = [point.coordinates[0]
                     for point in pointList]
                Y = [point.coordinates[1]
                     for point in pointList]
                Z = [point.coordinates[2]
                     for point in pointList]
                ax.scatter3D(X, Y, Z, marker='.', s=[1] * len(X))
            plt.show()

    def numpyify(self):
        import numpy as np
        return np.array([np.array(elt.coordinates) for elt in self.points]), np.array([elt.cluster for elt in self.points])

    def predict(self, model, *args, **kwargs):
        import numpy as np
        data, label = self.numpyify()
        label = model.predict_classes(
            data, **kwargs)
        points = []
        for i in range(len(data)):
            points.append(DataPoint(
                coordinates=data[i],
                cluster=label[i]
            ))
        return DataInstance({'dimension': self.dimension, 'classNumber': np.max(label) + 1, 'nPoints': len(label), 'points': points, 'orientation': self.orientation})

    def predictAndEvaluate(self, model, *args, **kwargs):
        data, label = self.numpyify()
        label_evaluation = model.predict_classes(
            data, **kwargs)
        res = 0
        for i in range(len(label)):
            if label[i] == label_evaluation[i]:
                res += 1
        return res / len(label)

    def computeSimplex(self, *args, **kwargs):
        import gudhi
        import random as r
        import numpy as np
        nPoints = kwargs.get('nPoints', self.nPoints)
        targetCluster = kwargs.get('targetCluster', [1])
        pointListTemp = []
        for point in self.points:
            if point.cluster in targetCluster:
                pointListTemp.append(np.array(point.coordinates))

        pointList = []
        for point in pointListTemp:
            random = r.random()
            if random <= nPoints/len(pointListTemp):
                pointList.append(point)
        if len(pointList) == 0:
            return []
        point_complex = gudhi.AlphaComplex(points=pointList)
        simplex_tree = point_complex.create_simplex_tree()
        return simplex_tree

    def computeBottleNeckDistance(self, instance, *args, **kwargs):
        import gudhi
        import numpy as np
        min_persistence = kwargs.get('min_persistence', 0)
        nPoints = kwargs.get('nPoints', self.nPoints)
        targetCluster = kwargs.get('targetCluster', [1])
        persistence = instance.computeSimplex(
            nPoints=nPoints, targetCluster=targetCluster)
        if persistence == []:
            return -1
        persistence.persistence(min_persistence=min_persistence)
        compareTo = self.computeSimplex(
            nPoints=nPoints, targetCluster=targetCluster)
        if compareTo == []:
            return -1
        compareTo.persistence(min_persistence=min_persistence)
        maxDistance = 0
        for dim in range(self.dimension):
            persistence_intervals = np.sqrt(
                persistence.persistence_intervals_in_dimension(dim))
            compareToIntervals = np.sqrt(
                compareTo.persistence_intervals_in_dimension(dim))
            bottleneck_distance = gudhi.bottleneck_distance(
                persistence_intervals, compareToIntervals)
            maxDistance = max(maxDistance, bottleneck_distance)
        return maxDistance

    def bettiNumbers(self, *args, **kwargs):
        from gudhi import RipsComplex
        import random as r
        import numpy as np
        nPoints = kwargs.get('nPoints', self.nPoints)
        targetCluster = kwargs.get('targetCluster', [1])
        maxEdge = kwargs.get('maxEdge', 10)
        maxDim = kwargs.get('maxDim', self.dimension)
        fromValue = kwargs.get('fromValue', 0.05)
        toValue = kwargs.get('toValue', 0.05)
        pointListTemp = []
        for point in self.points:
            if point.cluster in targetCluster:
                pointListTemp.append(np.array(point.coordinates))

        pointList = []
        for point in pointListTemp:
            random = r.random()
            if random <= nPoints/len(pointListTemp):
                pointList.append(point)
        point_complex = RipsComplex(
            max_edge_length=maxEdge, points=pointList)
        simplex_tree = point_complex.create_simplex_tree(
            max_dimension=maxDim)
        persistence = simplex_tree.persistence()
        return simplex_tree.persistent_betti_numbers(from_value=fromValue, to_value=toValue)

    def newBettiNumbers(self, *args, **kwargs):
        from networkx import Graph, connected_components, number_connected_components
        import numpy as np
        import random as r
        from .homology_utils import findPointStructDimension
        targetCluster = kwargs.get('targetCluster', [1])
        threshold = kwargs.get('threshold', 0.05)
        nPoints = kwargs.get('nPoints', self.nPoints)
        errorRate = kwargs.get('errorRate', 0.005)
        plot = kwargs.get('plot', False)
        # Build graph
        G = Graph()
        pointListTemp = []
        for point in self.points:
            if point.cluster in targetCluster:
                pointListTemp.append(np.array(point.coordinates))

        pointList = []
        for point in pointListTemp:
            random = r.random()
            if random <= nPoints/len(pointListTemp):
                pointList.append(point)

        n = len(pointList)//100
        nodes = [i for i in range(len(pointList))]
        edges = []
        for i in range(len(pointList)):
            if i % 100 == 0:
                print(str(i//100) + " / " + str(n))
            for j in range(i+1, len(pointList)):
                dist = np.sqrt(
                    np.sum((pointList[i] - pointList[j])**2))
                if dist <= threshold:
                    edges.append((i, j))
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        conComp = connected_components(G)
        conCompNumber = number_connected_components(G)
        centers = []
        compDim = []
        if plot:
            points = []
        ball = [False for _ in range(conCompNumber)]
        for e, z in enumerate(conComp):
            x = list(z)
            errorCount = 0
            if len(x) > 1 and errorCount < errorRate*nPoints:
                origin = pointList[x[0]]
                temp = [0 for _ in range(len(origin))]
                for k in range(len(x)):
                    temp += pointList[x[k]]

                temp /= len(x)
                compDim.append(findPointStructDimension(
                    [pointList[e] for e in x]))
                centers.append(temp)
                for pt in x:
                    dist = np.sqrt(
                        np.sum((pointList[pt] - centers[-1])**2))
                    if dist <= threshold:
                        ball[e] = True
            else:
                errorCount += 1
            if plot:
                for point in x:
                    points.append(DataPoint(coordinates=pointList[point],
                                            dimension=self.dimension, cluster=e + 1))
                print(e + 1, len(x))

        if plot:
            temp = DataInstance({'dimension': self.dimension, 'points': points, 'nPoints': len(
                points), 'classNumber': conCompNumber + 1})
            temp.plot(noBack=True)

        betti = [0 for _ in range(self.dimension)]
        for i in range(len(centers)):
            if not ball[i]:
                betti[compDim[i]] += 1
            else:
                betti[0] += 1
        return betti


class DataPoint(object):
    def __init__(self, *args, **kwargs):
        self.cluster = kwargs.get('cluster', 0)
        if "dimension" in kwargs:
            if "coordinates" in kwargs:
                if len(kwargs.get('coordinates')) != kwargs.get('dimension'):
                    raise ValueError(
                        'dimension of the data point must be the number of it\'s coordinates')
                else:
                    self.dimension = kwargs.get('dimension')
                    self.coordinates = tuple(kwargs.get("coordinates"))
            else:
                import random as r
                self.dimension = kwargs.get("dimension")
                temp = [0] * self.dimension
                for i in range(self.dimension):
                    temp[i] = r.random()
                self.coordinates = tuple(temp)

        elif "coordinates" in kwargs:
            self.coordinates = tuple(kwargs.get("coordinates"))
            self.dimension = len(kwargs.get("coordinates"))
        else:
            import random as r
            self.dimension = kwargs.get("dimension", 2)
            self.coordinates = kwargs.get(
                "coordinates", (r.random(), r.random()))

    def distanceTo(self, dataPoint):
        from math import sqrt
        if len(self.coordinates) != len(dataPoint.coordinates):
            raise ValueError(
                'distance between 2 points can only be computed if they live in the same dimension')
        res = 0
        for i in range(len(self.coordinates)):
            res += (self.coordinates[i] - dataPoint.coordinates[i])**2
        return sqrt(res)


class Bounds(object):
    def __init__(self, *args, **kwargs):
        if "dimension" in kwargs:
            if "boundsCoordinates" in kwargs:
                for x in kwargs.get('boundsCoordinates'):
                    if x['min'] >= x['max']:
                        raise ValueError(
                            'Dimension max must be strictly bigger than dimension\'s min')
                if len(kwargs.get('boundsCoordinates')) != kwargs.get('dimension'):
                    raise ValueError(
                        'Dimension of bounds and announced dimension must be the same')
                else:
                    self.dimension = kwargs.get('dimension')
                    self.boundsCoordinates = kwargs.get("boundsCoordinates")
            else:
                self.dimension = kwargs.get("dimension")
                self.boundsCoordinates = [
                    {'min': 0, 'max': 1}] * kwargs.get("dimension")
        elif "boundsCoordinates" in kwargs:
            for x in kwargs.get('boundsCoordinates'):
                if x['min'] >= x['max']:
                    raise ValueError(
                        'Dimension max must be strictly bigger than dimension\'s min')
            self.boundsCoordinates = kwargs.get("boundsCoordinates")
            self.dimension = len(kwargs.get("boundsCoordinates"))
        else:
            self.dimension = kwargs.get("dimension", 2)
            self.boundsCoordinates = kwargs.get(
                "boundsCoordinates", [{'min': 0, 'max': 1}, {'min': 0, 'max': 1}])
