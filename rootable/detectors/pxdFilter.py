import numpy as np
from uproot import TTree


class FindUnselectedClusters:
    """
    the purpose of this class is to reconstruct the different information
    of pxd clusters that lay outside of ROIs. instead of adding a flag to
    entries if they are outside or inside of a ROI or adding additional
    branches (god I hate that concept) containing all the info, they do
    this random shit, where there all digitis and I have to guess together
    what it is what, especially that the internal structure is completly
    different for selected and unselected digits. on top of that, the naming
    convention sucks and I picked something that is more self-explanatory.
    """
    def __init__(self) -> None:
        """
        Initialize the FindUnselectedClusters class.

        Parameters:
        - panelIDs (list[int]): List of sensor panel IDs.

        Attributes:
        - uBounds (tuple): The min/max bounds for u position.
        - vBounds (list[tuple]): The min/max bounds for v position for each sensor.
        - uvMapping (dict): Mapping of sensor IDs to their u/v bounds.
        - keyWords (list[str]): Keywords to extract from the event tree.
        """
        # these are the sensor IDs of the pxd modules/panels from the root file, they are
        # use to identify on which panels a cluster event happened
        self.panelIDs = np.array([ 8480,  8512,  8736,  8768,  8992,  9024,  9248,  9280,
                              9504,  9536,  9760,  9792, 10016, 10048, 10272, 10304,
                             16672, 16704, 16928, 16960, 17184, 17216, 17440, 17472,
                             17696, 17728, 17952, 17984, 18208, 18240, 18464, 18496,
                             18720, 18752, 18976, 19008, 19232, 19264, 19488, 19520])

        # u/v position min/max for layer 1 & 2, they are needed for reconstructing roi unselected cluster locations
        # it's the upper and lower bound of a u/v postion on a per sensor id
        self.uFit = {8480: np.poly1d([ 0.005    , -0.6228546]),
                     8512: np.poly1d([ 0.005     , -0.62285449]),
                     8736: np.poly1d([ 0.005    , -0.6228546]),
                     8768: np.poly1d([ 0.005     , -0.62285449]),
                     8992: np.poly1d([ 0.005    , -0.6228546]),
                     9024: np.poly1d([ 0.005     , -0.62285449]),
                     9248: np.poly1d([ 0.005    , -0.6228546]),
                     9280: np.poly1d([ 0.005     , -0.62285449]),
                     9504: np.poly1d([ 0.005    , -0.6228546]),
                     9536: np.poly1d([ 0.005     , -0.62285449]),
                     9760: np.poly1d([ 0.005    , -0.6228546]),
                     9792: np.poly1d([ 0.005     , -0.62285449]),
                     10016: np.poly1d([ 0.005    , -0.6228546]),
                     10048: np.poly1d([ 0.005     , -0.62285449]),
                     10272: np.poly1d([ 0.005    , -0.6228546]),
                     10304: np.poly1d([ 0.005     , -0.62285449]),
                     16672: np.poly1d([ 0.005     , -0.62285456]),
                     16704: np.poly1d([ 0.005     , -0.62285445]),
                     16928: np.poly1d([ 0.005     , -0.62285456]),
                     16960: np.poly1d([ 0.005     , -0.62285446]),
                     17184: np.poly1d([ 0.005     , -0.62285456]),
                     17216: np.poly1d([ 0.005     , -0.62285446]),
                     17440: np.poly1d([ 0.005     , -0.62285456]),
                     17472: np.poly1d([ 0.005     , -0.62285446]),
                     17696: np.poly1d([ 0.005     , -0.62285456]),
                     17728: np.poly1d([ 0.005     , -0.62285446]),
                     17952: np.poly1d([ 0.005     , -0.62285456]),
                     17984: np.poly1d([ 0.005     , -0.62285446]),
                     18208: np.poly1d([ 0.005     , -0.62285456]),
                     18240: np.poly1d([ 0.005     , -0.62285446]),
                     18464: np.poly1d([ 0.005     , -0.62285456]),
                     18496: np.poly1d([ 0.005     , -0.62285446]),
                     18720: np.poly1d([ 0.005     , -0.62285456]),
                     18752: np.poly1d([ 0.005     , -0.62285446]),
                     18976: np.poly1d([ 0.005     , -0.62285456]),
                     19008: np.poly1d([ 0.005     , -0.62285446]),
                     19232: np.poly1d([ 0.005     , -0.62285456]),
                     19264: np.poly1d([ 0.005     , -0.62285446]),
                     19488: np.poly1d([ 0.005     , -0.62285456]),
                     19520: np.poly1d([ 0.005     , -0.62285445])}
        self.vFit = {8480: np.poly1d([ 0.00587037, -2.29395374]),
                     8512: np.poly1d([ 0.00587037, -2.20862039]),
                     8736: np.poly1d([ 0.00587037, -2.29395374]),
                     8768: np.poly1d([ 0.00587037, -2.20862039]),
                     8992: np.poly1d([ 0.00587037, -2.29395375]),
                     9024: np.poly1d([ 0.00587037, -2.20862039]),
                     9248: np.poly1d([ 0.00587037, -2.29395375]),
                     9280: np.poly1d([ 0.00587037, -2.20862039]),
                     9504: np.poly1d([ 0.00587037, -2.29395375]),
                     9536: np.poly1d([ 0.00587037, -2.20862039]),
                     9760: np.poly1d([ 0.00587037, -2.29395375]),
                     9792: np.poly1d([ 0.00587037, -2.2086204 ]),
                     10016: np.poly1d([ 0.00587037, -2.29395375]),
                     10048: np.poly1d([ 0.00587037, -2.20862039]),
                     10272: np.poly1d([ 0.00587037, -2.29395375]),
                     10304: np.poly1d([ 0.00587037, -2.20862039]),
                     16672: np.poly1d([ 1.44676145e-06,  7.00144541e-03, -3.09694398e+00]),
                     16704: np.poly1d([-1.44676141e-06,  9.22077745e-03, -3.12427848e+00]),
                     16928: np.poly1d([ 1.44676147e-06,  7.00144538e-03, -3.09694398e+00]),
                     16960: np.poly1d([-1.44676141e-06,  9.22077745e-03, -3.12427848e+00]),
                     17184: np.poly1d([ 1.44676151e-06,  7.00144535e-03, -3.09694397e+00]),
                     17216: np.poly1d([-1.44676138e-06,  9.22077742e-03, -3.12427847e+00]),
                     17440: np.poly1d([ 1.44676148e-06,  7.00144538e-03, -3.09694398e+00]),
                     17472: np.poly1d([-1.44676141e-06,  9.22077744e-03, -3.12427848e+00]),
                     17696: np.poly1d([ 1.44676154e-06,  7.00144533e-03, -3.09694397e+00]),
                     17728: np.poly1d([-1.44676144e-06,  9.22077747e-03, -3.12427849e+00]),
                     17952: np.poly1d([ 1.44676148e-06,  7.00144539e-03, -3.09694398e+00]),
                     17984: np.poly1d([-1.44676143e-06,  9.22077746e-03, -3.12427848e+00]),
                     18208: np.poly1d([ 1.44676142e-06,  7.00144543e-03, -3.09694399e+00]),
                     18240: np.poly1d([-1.44676147e-06,  9.22077748e-03, -3.12427848e+00]),
                     18464: np.poly1d([ 1.44676148e-06,  7.00144539e-03, -3.09694398e+00]),
                     18496: np.poly1d([-1.44676139e-06,  9.22077742e-03, -3.12427847e+00]),
                     18720: np.poly1d([ 1.44676152e-06,  7.00144535e-03, -3.09694397e+00]),
                     18752: np.poly1d([-1.44676141e-06,  9.22077744e-03, -3.12427848e+00]),
                     18976: np.poly1d([ 1.44676153e-06,  7.00144534e-03, -3.09694397e+00]),
                     19008: np.poly1d([-1.44676139e-06,  9.22077743e-03, -3.12427848e+00]),
                     19232: np.poly1d([ 1.44676152e-06,  7.00144537e-03, -3.09694398e+00]),
                     19264: np.poly1d([-1.44676145e-06,  9.22077748e-03, -3.12427849e+00]),
                     19488: np.poly1d([ 1.44676150e-06,  7.00144538e-03, -3.09694398e+00]),
                     19520: np.poly1d([-1.44676143e-06,  9.22077746e-03, -3.12427848e+00])}

        # Keywords for extracting data from the event tree
        self.keyWords = [
            'pxd_unfiltered_digits/pxd_unfiltered_digits.m_uCellID',
            'pxd_unfiltered_digits/pxd_unfiltered_digits.m_vCellID',
            'pxd_unfiltered_digits/pxd_unfiltered_digits.m_charge',
            'pxd_unfiltered_digits/pxd_unfiltered_digits.m_sensorID'
        ]

    def _pixelToUV(self, uvIndex: tuple[int], sensorID: int) -> float:
        """
        Convert pixel indices to u/v positions based on sensor ID.

        Parameters:
        - uvIndex (tuple[int]): The u/v pixel index.
        - sensorID (int): The sensor ID.

        Returns:
        - tuple[float]: The u/v positions.
        """
        uMapped = self.uFit[sensorID](uvIndex[0])
        vMapped = self.vFit[sensorID](uvIndex[1])
        # Calculate and return the u/v positions for the given pixel index
        return uMapped, vMapped

    def getClusters(self, eventTree: TTree, fileName: str) -> dict:
        """
        Wrapper method to get cluster data.

        Parameters:
        - eventTree (TTree): The input event tree containing digit information.

        Returns:
        - dict: A dictionary containing processed cluster data.
        """
        return self._process(eventTree, 'clusters', fileName)

    def getDigits(self, eventTree: TTree) -> dict:
        """
        Wrapper method to get digit data.

        Parameters:
        - eventTree (TTree): The input event tree containing digit information.

        Returns:
        - dict: A dictionary containing processed digit data.
        """
        return self._process(eventTree, 'digits')

    @staticmethod
    def fillMCData(sampleData: dict, sampleSize: int) -> dict:
        """
        this method generates bullshit mc data for unselected clusters
        I use it for filling out empty slots in the mc data, maybe I
        will figure out how to find the proper data inside the root files
        """
        fakeMCData = {}
        for key in sampleData:
            fakeMCData[key] = np.zeros((sampleSize, *sampleData[key].shape[1:]), dtype=type(sampleData[key][0]))

        return fakeMCData

    def _process(self, eventTree: TTree, processType: str = 'clusters', fileName: str = None) -> dict:
        """
        Common method to process either clusters or digits based on the given processType.

        Parameters:
        - eventTree (TTree): The input event tree containing digit information.
        - processType (str): The type of processing to perform ('clusters' or 'digits').

        Returns:
        - dict: A dictionary containing processed data.
        """
        # Extract arrays from the event tree
        filteredDigits = eventTree.arrays(self.keyWords, library='np')
        uCellIDs = filteredDigits['pxd_unfiltered_digits/pxd_unfiltered_digits.m_uCellID']
        vCellIDs = filteredDigits['pxd_unfiltered_digits/pxd_unfiltered_digits.m_vCellID']
        cellCharges = filteredDigits['pxd_unfiltered_digits/pxd_unfiltered_digits.m_charge']
        clusterSensorIDs = filteredDigits['pxd_unfiltered_digits/pxd_unfiltered_digits.m_sensorID']

        # Initialize variables
        eventNumbers, selected = [], []
        uPositions, vPositions = [], []
        uSizes, vSizes, clsSizes = [], [], []
        uCells, vCells, cCharges = [], [], []
        seedCharges, clsCharges = [], []
        sensorIDs = []
        detector, fileNames = [], []

        # Loop through each cell charge to populate matrices and process data
        for i in range(len(cellCharges)):
            # Initialize and populate the matrix
            matrixLadder = np.zeros((250, 768))
            matrixLadder[uCellIDs[i], vCellIDs[i]] = cellCharges[i]
            sensorID = clusterSensorIDs[i]
            xx, yy = np.nonzero(matrixLadder)
            knownPixels = {id: {'xx': [], 'yy': []} for id in self.panelIDs}

            for x, y, id in zip(xx, yy, sensorID):
                if x in knownPixels[id]['xx'] or y in knownPixels[id]['yy']:
                    continue
                else:
                    knownPixels[id]['xx'].extend(xx)
                    knownPixels[id]['yy'].extend(yy)

                eventNumbers.append(i)
                sensorIDs.append(id)
                uPosition, vPosition = self._pixelToUV((x, y), id)
                uPositions.append(uPosition)
                vPositions.append(vPosition)

                xLower = np.clip(x-4, a_min=0, a_max=768)
                xUpper = np.clip(x+5, a_min=0, a_max=768)

                yLower = np.clip(y-4, a_min=0, a_max=768)
                yUpper = np.clip(y+5, a_min=0, a_max=768)

                matrix = matrixLadder[xLower:xUpper,yLower:yUpper]
                if matrix.shape != (9,9):
                    padding = tuple(np.array([9, 9]) - np.array(matrix.shape))
                    matrix = np.pad(matrix, ((0, padding[0]), (0, padding[1])), mode="constant", constant_values=0)

                xxArgMax, yyArgMax = np.unravel_index(matrix.argmax(), (9,9))
                if xxArgMax != 4 or yyArgMax != 0:
                    xShift = 4 - xxArgMax
                    yShift = 4 - yyArgMax

                    matrix = np.roll(matrix, (xShift, yShift), axis=(0,1))

                vv, uu = np.nonzero(matrix)
                cCharges.append(matrix[vv, uu].astype(int))
                vCells.append(vv + x)
                uCells.append(uu + y)

                seedCharges.append(matrix[4,4].astype(int))
                clsCharges.append(np.sum(matrix).astype(int))
                clsSizes.append(np.count_nonzero(matrix))
                vSizes.append(len(vv))
                uSizes.append(len(uu))
                selected.append(False)
                detector.append('pxd')
                fileNames.append(fileName)

        # Return the appropriate data based on the processType
        if processType == 'clusters':
            return {
                'clsCharge': np.array(clsCharges).astype(int),
                'seedCharge': np.array(seedCharges).astype(int),
                'clsSize': np.array(clsSizes).astype(int),
                'uSize': np.array(uSizes).astype(int),
                'vSize': np.array(vSizes).astype(int),
                'uPosition': np.array(uPositions),
                'vPosition': np.array(vPositions),
                'sensorID': np.array(sensorIDs).astype(int),
                'eventNumber': np.array(eventNumbers).astype(int),
                'roiSelected': np.array(selected),
                'detector': np.array(detector),
                'fileName': np.array(fileNames)
            }

        return {
            'uCellIDs': np.array(uCells, dtype=object),
            'vCellIDs': np.array(vCells, dtype=object),
            'cellCharges': np.array(cCharges, dtype=object)
        }
