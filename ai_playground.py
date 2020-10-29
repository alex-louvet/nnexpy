import shutil
import sys
import os
import pickle
from sklearn.utils import shuffle
from generate_data_dimension import DataDescriptor, DataPoint, DataInstance
import numpy as np
import subprocess
import json

randomSeed = 468643654

dataDescriptorList = [
    DataDescriptor(dimension=3, nHoles=3, holeDimension=[2, 2, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=4, holeDimension=[2, 2, 2, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=4, holeDimension=[2, 2, 3, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=5, holeDimension=[2, 2, 2, 3, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=6, holeDimension=[2, 2, 2, 3, 3, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=5, holeDimension=[2, 2, 3, 3, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=4, holeDimension=[2, 3, 3, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=3, holeDimension=[2, 3, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=4, holeDimension=[2, 2, 2, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=4, holeDimension=[2, -2, -3, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=5, holeDimension=[2, 2, -2, 3, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=6, holeDimension=[2, 2, -2, -3, 3, 3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=5, holeDimension=[2, 2, 3, 3, -3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
    DataDescriptor(dimension=3, nHoles=4, holeDimension=[2, 3, 3, -3
                                                         ], random=randomSeed, minStrata=1, maxStrata=3),
]

iterNum = 1
epoch_number = 1000
startIter = 0
if (len(sys.argv) > 1):
    iterNum = int(sys.argv[1])

if (len(sys.argv) > 2):
    epoch_number = int(sys.argv[2])

if (len(sys.argv) > 3):
    startIter = int(sys.argv[3])

for j, x in enumerate(dataDescriptorList):

    myRootPath = "./models/instance_" + str(j) + "/"

    if not os.path.exists(myRootPath):
        os.makedirs(myRootPath)
    else:
        for root, dirs, files in os.walk(myRootPath):
            for file in files:
                os.remove(os.path.join(root, file))

    dataDescriptor = x
    instance = dataDescriptor.generateData(
        pointsNumber=10000, random=randomSeed)

    with open(myRootPath + 'data_descriptor.pkl', 'wb') as output:
        pickle.dump(dataDescriptor.centerList, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataDescriptor.radiusList, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataDescriptor.bounds, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataDescriptor.holeDimension,
                    output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(instance.orientation, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(randomSeed, output, pickle.HIGHEST_PROTOCOL)

    data, label = instance.numpyify()
    data, label = shuffle(data, label, random_state=0)

    file = open(myRootPath + 'information.txt', 'w')
    file.write('Iteration Number: ' + str(iterNum) + "\n")
    file.write('Epoch Number: ' + str(epoch_number) + "\n")
    file.close()

    input_shape = (len(dataDescriptor.centerList[0].coordinates),)
    for i in range(startIter, startIter + iterNum):
        mypath = myRootPath + 'training_' + str(i) + '/'
        print('\n\n\n################################################# ' +
              str(i) + ' ##########################################\n\n\n')
        if not os.path.exists(mypath):
            os.makedirs(mypath)
        else:
            for root, dirs, files in os.walk(mypath):
                for file in files:
                    os.remove(os.path.join(root, file))

        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            with open('/tmp/data_file.pkl', 'wb') as dump:
                pickle.dump(data, dump, pickle.HIGHEST_PROTOCOL)

            with open('/tmp/label_file.pkl', 'wb') as dump:
                pickle.dump(label, dump, pickle.HIGHEST_PROTOCOL)

            subprocess.call(['python3', 'rasScript.py', str(i), str(input_shape[0]),
                             str(mypath), str(epoch_number), '/tmp/data_file.pkl', '/tmp/label_file.pkl'])

    subprocess.call(['python3', 'bettiScript.py', str(j)])
