import numpy as np
from ..common import calcSpherical
from concurrent.futures import ThreadPoolExecutor


class ClusterCoordinates:
    """
    This class takes care of cluster coordinates
    """
    def __init__(self) -> None:
        # these are the sensor IDs of the pxd modules/panels from the root file, they are
        # use to identify on which panels a cluster event happened
        self.panelIDs = np.array([ 8480,  8512,  8736,  8768,  8992,  9024,  9248,  9280,
                                   9504,  9536,  9760,  9792, 10016, 10048, 10272, 10304,
                                  16672, 16704, 16928, 16960, 17184, 17216, 17440, 17472,
                                  17696, 17728, 17952, 17984, 18208, 18240, 18464, 18496,
                                  18720, 18752, 18976, 19008, 19232, 19264, 19488, 19520])

        # every line in this corresponds to one entry in the array above, this is used
        # to put the projected uv plane in the right position
        self.panelShifts = np.array([[ 1.3985    ,  0.2652658 ,  3.68255],
                                     [ 1.3985    ,  0.2652658 , -0.88255],
                                     [ 0.80146531,  1.17631236,  3.68255],
                                     [ 0.80146531,  1.17631236, -0.88255],
                                     [-0.2652658 ,  1.3985    ,  3.68255],
                                     [-0.2652658 ,  1.3985    , -0.88255],
                                     [-1.17631236,  0.80146531,  3.68255],
                                     [-1.17631236,  0.80146531, -0.88255],

                                     [-1.3985    , -0.2652658 ,  3.68255],
                                     [-1.3985    , -0.2652658 , -0.88255],
                                     [-0.80146531, -1.17631236,  3.68255],
                                     [-0.80146531, -1.17631236, -0.88255],
                                     [ 0.2652658 , -1.3985    ,  3.68255],
                                     [ 0.2652658 , -1.3985    , -0.88255],
                                     [ 1.2652658 , -0.80146531,  3.68255],
                                     [ 1.2652658 , -0.80146531, -0.88255],

                                     [ 2.2015    ,  0.2652658 ,  5.01305],
                                     [ 2.2015    ,  0.2652658 , -1.21305],
                                     [ 1.77559093,  1.32758398,  5.01305],
                                     [ 1.77559093,  1.32758398, -1.21305],
                                     [ 0.87126021,  2.039055  ,  5.01305],
                                     [ 0.87126021,  2.039055  , -1.21305],
                                     [-0.2652658 ,  2.2015    ,  5.01305],
                                     [-0.2652658 ,  2.2015    , -1.21305],

                                     [-1.32758398,  1.77559093,  5.01305],
                                     [-1.32758398,  1.77559093, -1.21305],
                                     [-2.039055  ,  0.87126021,  5.01305],
                                     [-2.039055  ,  0.87126021, -1.21305],
                                     [-2.2015    , -0.2652658 ,  5.01305],
                                     [-2.2015    , -0.2652658 , -1.21305],
                                     [-1.77559093, -1.32758398,  5.01305],
                                     [-1.77559093, -1.32758398, -1.21305],

                                     [-0.87126021, -2.039055  ,  5.01305],
                                     [-0.87126021, -2.039055  , -1.21305],
                                     [ 0.2652658 , -2.2015    ,  5.01305],
                                     [ 0.2652658 , -2.2015    , -1.21305],
                                     [ 1.32758398, -1.77559093,  5.01305],
                                     [ 1.32758398, -1.77559093, -1.21305],
                                     [ 2.039055  , -0.87126021,  5.01305],
                                     [ 2.039055  , -0.87126021, -1.21305]])

        # every entry here corresponds to the entries in the array above, these are
        # used for rotating the projected uv plane
        self.panelRotations = np.array([ 90,  90, 225, 225, 180, 180, 135, 135,
                                        270, 270, 405, 405, 360, 360, 495, 495,
                                         90,  90,  60,  60,  30,  30, 180, 180,
                                        150, 150, 120, 120, 270, 270,  60,  60,
                                        390, 390, 360, 360, 330, 330, 300, 300])

        # the layer and ladder arrays, for finding them from sensor id
        self.panelLayer  = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2])
        self.panelLadder = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21])

        # all transpormaations are stored in a dict, with the sensor id as a keyword
        self.transformation = {}
        self.layersLadders = {}
        for i in range(len(self.panelIDs)):
            self.transformation[self.panelIDs[i]] = [self.panelShifts[i], self.panelRotations[i]]
            self.layersLadders[self.panelIDs[i]] = [self.panelLayer[i], self.panelLadder[i]]

    def get(self, uPositions: np.ndarray, vPositions: np.ndarray, sensorIDs: np.ndarray) -> dict:
        """
        converting the uv coordinates, together with sensor ids, into xyz coordinates
        """
        #setting up index chunks for multi threading
        #indexChunks = np.array_split(range(len(sensorIDs)), 4)

        xPosition, yPosition, zPosition = np.zeros_like(uPositions, dtype=float), np.zeros_like(uPositions, dtype=float), np.zeros_like(uPositions, dtype=float)

        for sensorID in self.panelIDs:
            indices = np.where(sensorIDs == sensorID)

            # grabbing the shift vector and rotation angle
            shift, angle = self.transformation[sensorID]

            # setting up rotation matrix
            theta = np.deg2rad(angle)
            rotMatrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

            points = np.column_stack((uPositions[indices], np.zeros_like(uPositions[indices]), vPositions[indices]))
            rotated = np.matmul(points, rotMatrix) + shift
            xPosition[indices], yPosition[indices], zPosition[indices] = rotated[:,0], rotated[:,1], rotated[:,2]

        return {'xPosition': xPosition, 'yPosition': yPosition, 'zPosition': zPosition}

    def layers(self, sensorIDs: np.ndarray) -> dict:
        """
        looks up the corresponding layers and ladders for every cluster
        """
        layersLadders = {}
        length = len(sensorIDs)
        layers = np.empty(length, dtype=int)
        ladders = np.empty(length, dtype=int)

        for i, id in enumerate(sensorIDs):
            layers[i], ladders[i] = self.layersLadders[id]

        return {'layer': np.array(layers, dtype=int),
               'ladder': np.array(ladders, dtype=int)}

    def sphericals(self, xPosition: np.ndarray, yPosition: np.ndarray, zPosition: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        this calculates spherical coordinates from xyz coordinates
        """
        return calcSpherical(xPosition, yPosition, zPosition)
