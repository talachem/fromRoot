import numpy as np
from numpy.typing import ArrayLike
from uproot import TTree
from ..common import fillMCList, extractMatrix


class MCtoClusters:
    """
    This class establishes the relationship between clusters and monte carlo data
    """
    def __init__(self) -> None:
        # behind these keys are the monte carlo info on the simulated data
        self.mcKeys = {              'pdg': 'MCParticles/MCParticles.m_pdg',
                                    'mass': 'MCParticles/MCParticles.m_mass',
                                  'energy': 'MCParticles/MCParticles.m_energy',
                               'momentumX': 'MCParticles/MCParticles.m_momentum_x',
                               'momentumY': 'MCParticles/MCParticles.m_momentum_y',
                               'momentumZ': 'MCParticles/MCParticles.m_momentum_z',
                             'validVertex': 'MCParticles/MCParticles.m_validVertex',
                          'productionTime': 'MCParticles/MCParticles.m_productionTime',
                       'productionVertexX': 'MCParticles/MCParticles.m_productionVertex_x',
                       'productionVertexY': 'MCParticles/MCParticles.m_productionVertex_y',
                       'productionVertexZ': 'MCParticles/MCParticles.m_productionVertex_z',
                               'decayTime': 'MCParticles/MCParticles.m_decayTime',
                            'decayVertexX': 'MCParticles/MCParticles.m_decayVertex_x',
                            'decayVertexY': 'MCParticles/MCParticles.m_decayVertex_y',
                            'decayVertexZ': 'MCParticles/MCParticles.m_decayVertex_z'}

        # these two establish the relation ship to an from clusters and monte carlo
        # there more entries than in the cluster data, but there still mc data missing
        # for some cluster files
        self.mcClusterRelations = {'from': 'PXDClustersToMCParticles/m_elements/m_elements.m_from',
                                     'to': 'PXDClustersToMCParticles/m_elements/m_elements.m_to'}

    def branches(self, *, includeUnselected: bool = False) -> list:
        return list((self.mcKeys | self.mcClusterRelations).values())

    def get(self, eventTree: TTree) -> dict:
        """
        this loads the monte carlo from the root file
        """
        # the monte carlo data, they are longer than the cluster data
        mcData = eventTree.arrays(self.mcKeys.values(), library='np')
        pdg = mcData[self.mcKeys['pdg']]
        momentumX = mcData[self.mcKeys['momentumX']]
        momentumY = mcData[self.mcKeys['momentumY']]
        momentumZ = mcData[self.mcKeys['momentumZ']]

        # this loads the relation ships to and from clusters and mc data
        # this is the same level of retardedness as with the cluster digits
        clusterToMC = eventTree.arrays(self.mcClusterRelations['to'], library='np')[self.mcClusterRelations['to']]
        mcToCluster = eventTree.arrays(self.mcClusterRelations['from'], library='np')[self.mcClusterRelations['from']]

        # it need the cluster charge as a jagged/ragged array, maybe I could simply
        # use the event numbers, but I am too tired to fix this shitty file format
        clsCharge = eventTree.arrays('PXDClusters/PXDClusters.m_clsCharge', library='np')['PXDClusters/PXDClusters.m_clsCharge']

        # reorganizing MC data
        n = len(clusterToMC)
        momentumXList = np.zeros(n, dtype=object)
        momentumYList = np.zeros(n, dtype=object)
        momentumZList = np.zeros(n, dtype=object)
        pdgList = np.zeros(n, dtype=object)
        clusterNumbersList = np.zeros(n, dtype=object)
        for i in range(n):
            # _fillMCList fills in the missing spots, because there are not mc data for
            # every cluster, even though there are more entries in this branch than
            # in the cluster branch... as I said, the root format is retarded
            fullClusterReferences = fillMCList(mcToCluster[i], clusterToMC[i], len(clsCharge[i]))
            clusterNumbersList[i] = fullClusterReferences
            pdgs, xmom, ymom, zmom = self._getMCData(fullClusterReferences, pdg[i], momentumX[i], momentumY[i], momentumZ[i])
            momentumXList[i] = xmom
            momentumYList[i] = ymom
            momentumZList[i] = zmom
            pdgList[i] = pdgs

        return {
            'momentumX': np.hstack(momentumXList).astype(float),
            'momentumY': np.hstack(momentumYList).astype(float),
            'momentumZ': np.hstack(momentumZList).astype(float),
                  'pdg': np.hstack(pdgList).astype(int),
            'clsNumber': np.hstack(clusterNumbersList).astype(int)
            }

    @staticmethod
    def _getMCData(toClusters: ArrayLike, pdgs: ArrayLike, xMom: ArrayLike, yMom: ArrayLike, zMom: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        after filling and reorganizing MC data arrays one can finally collect the
        actual MC data, where there's data missing I will with zeros
        """
        n = len(toClusters)
        pxList = np.zeros(n)
        pyList = np.zeros(n)
        pzList = np.zeros(n)
        pdgList = np.zeros(n, dtype=int)

        for i, references in enumerate(toClusters):
            if references == -1:
                continue  # Arrays were initialized to zero
            else:
                pxList[i] = xMom[references]
                pyList[i] = yMom[references]
                pzList[i] = zMom[references]
                pdgList[i] = pdgs[references]

        return pdgList, pxList, pyList, pzList


class MCtoDigits:
    """
    This class establishes the relationship between digits and monte carlo data
    """
    def __init__(self) -> None:
        # these are the sensor IDs of the pxd modules/panels from the root file, they are
        # use to identify on which panels a cluster event happened
        self.panelIDs = np.array([ 8480,  8512,  8736,  8768,  8992,  9024,  9248,  9280,
                                   9504,  9536,  9760,  9792, 10016, 10048, 10272, 10304,
                                  16672, 16704, 16928, 16960, 17184, 17216, 17440, 17472,
                                  17696, 17728, 17952, 17984, 18208, 18240, 18464, 18496,
                                  18720, 18752, 18976, 19008, 19232, 19264, 19488, 19520])

        # behind these keys are the monte carlo info on the simulated data
        self.mcKeys = {              'pdg': 'MCParticles/MCParticles.m_pdg',
                                    'mass': 'MCParticles/MCParticles.m_mass',
                                  'energy': 'MCParticles/MCParticles.m_energy',
                               'momentumX': 'MCParticles/MCParticles.m_momentum_x',
                               'momentumY': 'MCParticles/MCParticles.m_momentum_y',
                               'momentumZ': 'MCParticles/MCParticles.m_momentum_z',
                             'validVertex': 'MCParticles/MCParticles.m_validVertex',
                          'productionTime': 'MCParticles/MCParticles.m_productionTime',
                       'productionVertexX': 'MCParticles/MCParticles.m_productionVertex_x',
                       'productionVertexY': 'MCParticles/MCParticles.m_productionVertex_y',
                       'productionVertexZ': 'MCParticles/MCParticles.m_productionVertex_z',
                               'decayTime': 'MCParticles/MCParticles.m_decayTime',
                            'decayVertexX': 'MCParticles/MCParticles.m_decayVertex_x',
                            'decayVertexY': 'MCParticles/MCParticles.m_decayVertex_y',
                            'decayVertexZ': 'MCParticles/MCParticles.m_decayVertex_z'}

        # if a file is missing cluster data, I need to establish mc-cluster relation
        # through the digits... who came up with this insanity?
        self.mcDigitsInRelations = {'from': 'PXDDigitsToMCParticles/m_elements/m_elements.m_from',
                                      'to': 'PXDDigitsToMCParticles/m_elements/m_elements.m_to'}

        self.mcDigitsOutRelations = {'from': 'PXDDigitsOUTToMCParticles/m_elements/m_elements.m_from',
                                       'to': 'PXDDigitsOUTToMCParticles/m_elements/m_elements.m_to'}

        # in order to establish the relation ship between mc and digits and ultematly
        # between mc and clusters I need this additional branch...
        # as I said many times, root is the most retarded file format on earth
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
            return list((self.mcKeys | self.mcDigitsInRelations | self.mcDigitsOutRelations).values())
        return list((self.mcKeys | self.mcDigitsInRelations).values())

    def get(self, eventTree: TTree, inOut: str = 'inROI') -> dict:
        pdg, momentumX, momentumY, momentumZ, fromDigits, toDigits, uCellIDs, vCellIDs, cellCharges, clusterSensorIDs = self._selectKeys(eventTree, inOut=inOut)
        return self._process(pdg, momentumX, momentumY, momentumZ, fromDigits, toDigits, uCellIDs, vCellIDs, cellCharges, clusterSensorIDs)

    def _selectKeys(self, eventTree: TTree, inOut: str = 'inROI') -> tuple:
        mcData = eventTree.arrays(self.mcKeys.values(), library='np')
        pdg = mcData[self.mcKeys['pdg']]
        momentumX = mcData[self.mcKeys['momentumX']]
        momentumY = mcData[self.mcKeys['momentumY']]
        momentumZ = mcData[self.mcKeys['momentumZ']]
        if inOut == 'inROI':
            relations = eventTree.arrays(self.mcDigitsInRelations.values(), library='np')
            fromDigits = relations[self.mcDigitsInRelations['from']]
            # I need to convert data type of each entry... fucking root
            toDigits = np.array([np.array([item[0] for item in relations[self.mcDigitsInRelations['to']][i]]) for i in range(len(relations[self.mcDigitsInRelations['to']]))], dtype=object)

            digits = eventTree.arrays(self.digitsInKeys.values(), library='np')
            uCellIDs = digits[self.digitsInKeys['uCellID']]
            vCellIDs = digits[self.digitsInKeys['vCellID']]
            cellCharges = digits[self.digitsInKeys['cellCharge']]
            clusterSensorIDs = digits[self.digitsInKeys['sensorID']]
        else:
            relations = eventTree.arrays(self.mcDigitsOutRelations.values(), library='np')
            fromDigits = relations[self.mcDigitsOutRelations['from']]
            toDigits = np.array([np.array([item[0] for item in relations[self.mcDigitsOutRelations['to']][i]]) for i in range(len(relations[self.mcDigitsOutRelations['to']]))], dtype=object)

            digits = eventTree.arrays(self.digitsOutKeys.values(), library='np')
            uCellIDs = digits[self.digitsOutKeys['uCellID']]
            vCellIDs = digits[self.digitsOutKeys['vCellID']]
            cellCharges = digits[self.digitsOutKeys['cellCharge']]
            clusterSensorIDs = digits[self.digitsOutKeys['sensorID']]

        return pdg, momentumX, momentumY, momentumZ, fromDigits, toDigits, uCellIDs, vCellIDs, cellCharges, clusterSensorIDs

    def _process(self, pdg: ArrayLike, momentumX: ArrayLike, momentumY: ArrayLike, momentumZ: ArrayLike, fromDigits: ArrayLike, toDigits: ArrayLike, uCellIDs: ArrayLike, vCellIDs: ArrayLike, cellCharges: ArrayLike, clusterSensorIDs: ArrayLike) -> dict:
        # Loop through each cell charge to populate matrices and process data
        pdgCodes = []
        momentumXList = []
        momentumYList = []
        momentumZList = []
        clsNumbers = []

        for i in range(len(clusterSensorIDs)):
            # Initialize and populate the matrix
            matrixLadder = np.zeros((250, 768))
            matrixLadder[uCellIDs[i], vCellIDs[i]] = cellCharges[i]
            sensorID = clusterSensorIDs[i]
            xx, yy = uCellIDs[i], vCellIDs[i]

            # retrieving mc data and references
            mcDigits = fillMCList(fromDigits[i], toDigits[i], len(sensorID))
            pdgs = pdg[i]
            momentaX = momentumX[i]
            momentaY = momentumY[i]
            momentaZ = momentumZ[i]

            assert len(mcDigits) == len(sensorID), f'event {i}, mcDigits: {len(mcDigits)} and sensorID: {len(sensorID)}'

            # here I store all pixels, that have already been visited
            knownPixels = {id: set() for id in self.panelIDs}

            for x, y, id, relation in zip(xx, yy, sensorID, mcDigits):
                if (x, y) in knownPixels[id]:
                    continue

                matrix, xLower, yLower = extractMatrix(matrixLadder, x, y)
                vv, uu = np.nonzero(matrix)

                # Convert local indices of non-zero pixels to global indices
                globalNonzeroX = vv + xLower
                globalNonzeroY = uu + yLower

                # Update knownPixels with the global coordinates of the non-zero pixels
                knownPixels[id].update(zip(globalNonzeroX, globalNonzeroY))

                if relation == -1:
                    pdgCodes.append(0)
                    momentumXList.append(0.0)
                    momentumYList.append(0.0)
                    momentumZList.append(0.0)
                else:
                    pdgCodes.append(pdgs[relation])
                    momentumXList.append(momentaX[relation])
                    momentumYList.append(momentaY[relation])
                    momentumZList.append(momentaZ[relation])
                clsNumbers.append(relation)

        return {
            'pdg': np.array(pdgCodes).astype(int),
            'momentumX': np.array(momentumXList),
            'momentumY': np.array(momentumYList),
            'momentumZ': np.array(momentumZList),
            'clsNumber': np.array(clsNumbers).astype(int)
        }
