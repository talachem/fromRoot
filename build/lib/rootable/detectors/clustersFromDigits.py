from collections import defaultdict
import numpy as np
from numpy.typing import ArrayLike
from uproot import TTree
from ..common import extractMatrix


class ClustersFromDigits:
    """
    A class intended for reconstructing pxd cluster parameters from digit information
    """
    def __init__(self) -> None:
        # these are the sensor IDs of the pxd modules/panels from the root file, they are
        # use to identify on which panels a cluster event happened
        self.panelIDs = np.array([ 8480,  8512,  8736,  8768,  8992,  9024,  9248,  9280,
                                   9504,  9536,  9760,  9792, 10016, 10048, 10272, 10304,
                                  16672, 16704, 16928, 16960, 17184, 17216, 17440, 17472,
                                  17696, 17728, 17952, 17984, 18208, 18240, 18464, 18496,
                                  18720, 18752, 18976, 19008, 19232, 19264, 19488, 19520])

        # this maps sensor ids to the ladder, which is interpreted as image pixel depth
        self.panelIDtoDepth = {id: int(index) for index, id in enumerate(self.panelIDs)}

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

        self.digitsInKeys = {  'sensorID': 'PXDDigits/PXDDigits.m_sensorID',
                                'uCellID': 'PXDDigits/PXDDigits.m_uCellID',
                                'vCellID': 'PXDDigits/PXDDigits.m_vCellID',
                             'cellCharge': 'PXDDigits/PXDDigits.m_charge'}

        self.digitsOutKeys = {  'sensorID': 'PXDDigitsOUT/PXDDigitsOUT.m_sensorID',
                                 'uCellID': 'PXDDigitsOUT/PXDDigitsOUT.m_uCellID',
                                 'vCellID': 'PXDDigitsOUT/PXDDigitsOUT.m_vCellID',
                              'cellCharge': 'PXDDigitsOUT/PXDDigitsOUT.m_charge'}

    def branches(self, *, includeUnselected: bool = False) -> list:
        if includeUnselected is True:
            return list((self.digitsInKeys | self.digitsOutKeys).values())
        return list(self.digitsInKeys.values())

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

    def get(self, eventTree: TTree, inOut: str = 'inROI') -> dict:
        """
        Wrapper method to get cluster data.

        Parameters:
        - eventTree (TTree): The input event tree containing digit information.

        Returns:
        - dict: A dictionary containing processed cluster data.
        """
        uCellIDs, vCellIDs, cellCharges, sensorIDs = self._selectKeys(eventTree, inOut=inOut)
        return self._process(uCellIDs, vCellIDs, cellCharges, sensorIDs)

    def _selectKeys(self, eventTree: TTree, inOut: str = 'inROI') -> tuple:
        """
        grabbing the relavent arrays from the event tree
        I seperated this out to shorten the already lengthy '_process' method
        """
        if inOut == 'inROI':
            digits = eventTree.arrays(self.digitsInKeys.values(), library='np')
            uCellIDs = digits[self.digitsInKeys['uCellID']]
            vCellIDs = digits[self.digitsInKeys['vCellID']]
            cellCharges = digits[self.digitsInKeys['cellCharge']]
            sensorIDs = digits[self.digitsInKeys['sensorID']]
        else:
            digits = eventTree.arrays(self.digitsOutKeys.values(), library='np')
            uCellIDs = digits[self.digitsOutKeys['uCellID']]
            vCellIDs = digits[self.digitsOutKeys['vCellID']]
            cellCharges = digits[self.digitsOutKeys['cellCharge']]
            sensorIDs = digits[self.digitsOutKeys['sensorID']]

        return uCellIDs, vCellIDs, cellCharges, sensorIDs

    def _process(self, uCellIDsAllEvents: ArrayLike, vCellIDsAllEvents: ArrayLike, cellChargesAllEvents: ArrayLike, sensorIDsAllEvents: ArrayLike) -> dict:
        """
        Common method to process either clusters or digits based on the given processType.

        Parameters:
        - eventTree (TTree): The input event tree containing digit information.
        - processType (str): The type of processing to perform ('clusters' or 'digits').

        Returns:
        - dict: A dictionary containing processed data.
        """
        # Initialize variables
        eventNumbers = []
        uPositions, vPositions = [], []
        uSizes, vSizes, clsSizes = [], [], []
        uCells, vCells, cCharges = [], [], []
        seedCharges, clsCharges = [], []
        sensorIDs = []

        matrixLadder = np.zeros((40, 250, 768))
        # Loop through each cell charge to populate matrices and process data
        for i in range(len(sensorIDsAllEvents)):
            sensorID = sensorIDsAllEvents[i]
            if len(sensorID) == 0:
                continue
            # Initialize and populate the matrix
            uuIDs, vvIDs = uCellIDsAllEvents[i], vCellIDsAllEvents[i]
            depth = [self.panelIDtoDepth[id] for id in sensorID]
            matrixLadder[depth, uuIDs, vvIDs] = cellChargesAllEvents[i]

            # checking if no pixels overlap
            assert len(vvIDs) == len(sensorID), f"event: {i}, vvIDs: {len(vvIDs)}, sensorID: {len(sensorID)}"
            assert len(uuIDs) == len(sensorID), f"event: {i}, uuIDs: {len(uuIDs)}, sensorID: {len(sensorID)}"
            #print(i, len(sensorID), len(cellCharges))

            # here I store all pixels, that have already been visited
            knownPixels = {id: set() for id in self.panelIDs}

            for j, (x, y, id) in enumerate(zip(uuIDs, vvIDs, sensorID)):
                if (x, y) in knownPixels[id]:
                    continue

                eventNumbers.append(i)
                sensorIDs.append(id)

                matrix, globalUPositions, globalVPositions, seedUGlobal, seedVGlobal = extractMatrix(matrixLadder[self.panelIDtoDepth[id]], x, y, eventNumber = (i,j))

                cCharges.append(matrix[np.nonzero(matrix)].astype(int))
                vCells.append(globalUPositions)
                uCells.append(globalVPositions)

                seedChargePos = np.unravel_index(matrix.argmax(), matrix.shape)
                seedCharges.append(matrix[seedChargePos[0],seedChargePos[1]].astype(int))
                clsCharges.append(np.sum(matrix).astype(int))
                clsSizes.append(np.count_nonzero(matrix))
                uSizes.append(len(np.nonzero(matrix.sum(0))))
                vSizes.append(len(np.nonzero(matrix.sum(1))))

                # Update knownPixels with the global coordinates of the non-zero pixels
                knownPixels[id].update(zip(globalUPositions, globalVPositions))

                # Append these global (u, v) positions to their respective lists
                uPosition, vPosition = self._pixelToUV((seedUGlobal, seedVGlobal), id)
                uPositions.append(uPosition)
                vPositions.append(vPosition)

            matrixLadder[depth, uuIDs, vvIDs] = 0

        return {
            'eventNumber': np.array(eventNumbers).astype(int),
            'clsCharge': np.array(clsCharges).astype(int),
            'seedCharge': np.array(seedCharges).astype(int),
            'clsSize': np.array(clsSizes).astype(int),
            'uSize': np.array(uSizes).astype(int),
            'vSize': np.array(vSizes).astype(int),
            'uPosition': np.array(uPositions),
            'vPosition': np.array(vPositions),
            'sensorID': np.array(sensorIDs).astype(int),
            'uCellIDs': np.array(uCells, dtype=object),
            'vCellIDs': np.array(vCells, dtype=object),
            'cellCharges': np.array(cCharges, dtype=object)
        }
