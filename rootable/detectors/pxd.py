import numpy as np
from numpy.typing import ArrayLike
from uproot import TTree
from ..common import FancyDict
from .clusterCoordinates import ClusterCoordinates
from .mcToClusters import MCtoClusters, MCtoDigits
from .clustersFromDigits import ClustersFromDigits
from .generateMatrices import GenerateMatrices
import warnings


class PXD(FancyDict):
    def __init__(self, data: dict | None = None) -> None:
        self.name = 'pxd'

        # list of pxd panels
        self.panels = [[[-0.89 ,  0.36 ,  0.36 , -0.89 , -0.89 ], [ 1.4  ,  1.4  ,  1.4  ,  1.4  ,  1.4  ], [-3.12, -3.12, 5.92, 5.92, -3.12]],      # 00
                       [[ 1.25 ,  0.365,  0.365,  1.25 ,  1.25 ], [ 0.72 ,  1.615,  1.615,  0.72 ,  0.72 ], [-3.12, -3.12, 5.92, 5.92, -3.12]],      # 01
                       [[ 1.4  ,   1.4 ,  1.4  ,  1.4  ,  1.4  ], [-0.36 ,  0.89 ,  0.89 , -0.36 , -0.36 ], [-3.12, -3.12, 5.92, 5.92, -3.12]],      # 02
                       [[ 0.72 ,  1.615,  1.615,  0.72 ,  0.72 ], [-1.25 , -0.365, -0.365, -1.25 , -1.25 ], [-3.12, -3.12, 5.92, 5.92, -3.12]],      # 03
                       [[ 0.89 , -0.36 , -0.36 ,  0.89 ,  0.89 ], [-1.4  , -1.4  , -1.4  , -1.4  , -1.4  ], [-3.12, -3.12, 5.92, 5.92, -3.12]],      # 04
                       [[-1.25 , -0.365, -0.365, -1.25 , -1.25 ], [-0.72 , -1.615, -1.615, -0.72 , -0.72 ], [-3.12, -3.12, 5.92, 5.92, -3.12]],      # 05
                       [[-1.4  , -1.4  , -1.4  , -1.4  , -1.4  ], [ 0.36 , -0.89 , -0.89 ,  0.36 ,  0.36 ], [-3.12, -3.12, 5.92, 5.92, -3.12]],      # 06
                       [[-0.72 , -1.615, -1.615, -0.72 , -0.72 ], [ 1.25 ,  0.365,  0.365,  1.25 ,  1.25 ], [-3.12, -3.12, 5.92, 5.92, -3.12]],      # 07
                       [[-0.89 ,  0.36 ,  0.36 , -0.89 , -0.89 ], [ 2.2  ,  2.2  ,  2.2  ,  2.2  ,  2.2  ], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 08
                       [[ 0.345,  1.4  ,  1.4  ,  0.345,  0.345], [ 2.35 ,  1.725,  1.725,  2.35 ,  2.35 ], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 09
                       [[ 1.48 ,  2.1  ,  2.1  ,  1.48 ,  1.48 ], [ 1.85 ,  0.78 ,  0.78 ,  1.85 ,  1.85 ], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 10
                       [[ 2.2  ,  2.2  ,  2.2  ,  2.2  ,  2.2  ], [ 0.89 , -0.36 , -0.36 ,  0.89 ,  0.89 ], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 11
                       [[ 2.35 ,  1.725,  1.725,  2.35 ,  2.35 ], [-0.345, -1.4  , -1.4  , -0.345, -0.345], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 12
                       [[ 1.85 ,  0.78 ,  0.78 ,  1.85 ,  1.85 ], [-1.48 , -2.1  , -2.1  , -1.48 , -1.48 ], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 13
                       [[ 0.89 , -0.36 , -0.36 ,  0.89 ,  0.89 ], [-2.2  , -2.2  , -2.2  , -2.2  , -2.2  ], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 14
                       [[-0.345, -1.4  , -1.4  , -0.345, -0.345], [-2.35 , -1.725, -1.725, -2.35 , -2.35 ], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 15
                       [[-1.48 , -2.1  , -2.1  , -1.48 , -1.48 ], [-1.85 , -0.78 , -0.78 , -1.85 , -1.85 ], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 16
                       [[-2.2  , -2.2  , -2.2  , -2.2  , -2.2  ], [-0.89 ,  0.36 ,  0.36 , -0.89 , -0.89 ], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 17
                       [[-2.35 , -1.725, -1.725, -2.35 , -2.35 ], [ 0.345,  1.4  ,  1.4  ,  0.345,  0.345], [-4.28, -4.28, 8.08, 8.08, -4.28]],      # 18
                       [[-1.85 , -0.78 , -0.78 , -1.85 , -1.85 ], [ 1.48 ,  2.1  ,  2.1  ,  1.48 ,  1.48 ], [-4.28, -4.28, 8.08, 8.08, -4.28]]]      # 19

        # these are the branch names for cluster info in the root file
        self.clusterKeys = { 'clsCharge': 'PXDClusters/PXDClusters.m_clsCharge',
                            'seedCharge': 'PXDClusters/PXDClusters.m_seedCharge',
                               'clsSize': 'PXDClusters/PXDClusters.m_clsSize',
                                 'uSize': 'PXDClusters/PXDClusters.m_uSize',
                                 'vSize': 'PXDClusters/PXDClusters.m_vSize',
                                'uStart': 'PXDClusters/PXDClusters.m_uStart',
                                'vStart': 'PXDClusters/PXDClusters.m_vStart',
                             'uPosition': 'PXDClusters/PXDClusters.m_uPosition',
                             'vPosition': 'PXDClusters/PXDClusters.m_vPosition',
                              'sensorID': 'PXDClusters/PXDClusters.m_sensorID'}

        # these are the branch names for cluster digits in the root file
        self.digitKeys = {   'uCellIDs': 'PXDDigits/PXDDigits.m_uCellID',
                             'vCellIDs': 'PXDDigits/PXDDigits.m_vCellID',
                          'cellCharges': 'PXDDigits/PXDDigits.m_charge'}

        # this establishes the relationship between clusters and digits
        # because for some reaseon the branch for digits has a different
        # size/shape than the cluster branch
        self.clusterToDigis = 'PXDClustersToPXDDigits/m_elements/m_elements.m_to'


        # parameter for checking if coordinates have been loaded
        self.gotClusters = False
        self.gotCoordinates = False
        self.gotSphericals = False
        self.gotLayers = False
        self.gotDigits = False
        self.gotMatrices = False
        self.gotMCData = False

        # setting up classes for loading, reorganizing and calculating
        # specific pieces of data
        self.clusterCoordinates = ClusterCoordinates()
        self.generateMatrices = GenerateMatrices()
        self.clustersFromDigits = ClustersFromDigits()
        self.mcToClusters = MCtoClusters()
        self.mcToDigits = MCtoDigits()

        # this dict stores the data
        self.data = data if data is not None else {}
        self.length = 0

    def branches(self, *, includeUnselected: bool = False) -> dict:
        branches = {  'clusters': list(self.clusterKeys.values()),
                        'digits': self.clustersFromDigits.branches(includeUnselected=includeUnselected),
                    'monteCarlo': self.mcToClusters.branches(includeUnselected=includeUnselected)}
        branches['monteCarlo'].extend(self.mcToDigits.branches(includeUnselected=includeUnselected))
        return branches

    def getClusters(self, eventTree: TTree, fileName: str = None, includeUnselected: bool = False) -> None:
        """
        this uses the array from __init__ to load different branches into the data dict
        """
        #if self.gotClusters:
        #    return

        eventKeys = set(eventTree.keys())
        missing_branches = set(self.clusterKeys.values()) - eventKeys
        if missing_branches:
            clusters = self.clustersFromDigits.get(eventTree, 'inROI')
            for key in self.clusterKeys.keys():
                self.set(key, clusters[key])
            self.set('eventNumber', clusters['eventNumber'])
        else:
            for key, branch in self.clusterKeys.items():
                data = self._getData(eventTree, branch)
                self.set(key, data)
            clusters = eventTree.arrays('PXDClusters/PXDClusters.m_clsCharge', library='np')['PXDClusters/PXDClusters.m_clsCharge']
            self._getEventNumbers(clusters)

        length = len(self.data['clsCharge']) - self.length
        self.length = len(self.data['clsCharge'])
        self.set('roiSelected', np.array([True] * length))
        self.set('fileName', np.array([fileName] * length))

        missing_branches = set(self.clustersFromDigits.digitsOutKeys.values()) - eventKeys
        if includeUnselected and not missing_branches:
            clusters = self.clustersFromDigits.get(eventTree, 'outROI')
            clusters_ = {key: clusters[key] for key in self.clusterKeys.keys()}
            length = len(clusters_[list(clusters_.keys())[0]])
            self.length += length
            clusters_['roiSelected'] = np.array([False] * length)
            clusters_['fileName'] = np.array([fileName] * length)
            clusters_['eventNumber'] = clusters['eventNumber']
            self.extend(clusters_)

        self.gotClusters = True

    def _getEventNumbers(self, clusters: np.ndarray, offset: int = 0) -> None:
        """
        this generates event numbers from the structure of pxd clusters
        """
        eventNumbers = []
        for i in range(len(clusters)):
            eventNumbers.append(np.array([i]*len(clusters[i])) + offset)
        self.set('eventNumber', np.concatenate(eventNumbers))

    def _getData(self, eventTree: TTree, keyword: str, library: str = 'np') -> np.ndarray:
        """
        a private method for converting branches into something useful, namely
        into numpy arrays, if the keyward library is set to np.
        keyword: str = the full branch name
        library: str = can be 'np' (numpy), 'pd' (pandas) or 'ak' (akward)
                       see uproot documentation for more info
        """
        try:
            data = eventTree.arrays(keyword, library=library)[keyword]
            return np.hstack(data)
        except:
            return KeyError

    def getDigits(self, eventTree: TTree, includeUnselected: bool = False) -> None:
        """
        reorganizes digits, so that they fit to the clusters
        this is still pretty slow, because of the underlaying data structure
        """
        #if self.gotDigits:
        #    return

        eventKeys = set(eventTree.keys())
        digitKeys = set(self.digitKeys.values())
        digitKeys.add(self.clusterToDigis)
        missing_branches = digitKeys - eventKeys

        if missing_branches:
            digits = self.clustersFromDigits.get(eventTree, 'inROI')
            for key in self.digitKeys.keys():
                self.set(key, digits[key])
        else:
            digits = eventTree.arrays(self.digitKeys.values(), library='np')
            uCellIDs = digits[self.digitKeys['uCellIDs']]
            vCellIDs = digits[self.digitKeys['vCellIDs']]
            cellCharges = digits[self.digitKeys['cellCharges']]

            # this establishes the relation between digits and clusters, it's still
            # shocking to me, that this is necessary, why aren't digits stored in the
            # same way as clusters, than one wouldn't need to jump through hoops just
            # to have the data in a usable und sensible manner
            # root is such a retarded file format
            clusterDigits = eventTree.arrays(self.clusterToDigis, library='np')[self.clusterToDigis]

            uCellIDsTemp = []
            vCellIDsTemp = []
            cellChargesTemp = []
            for event in range(len(clusterDigits)):
                for cls in clusterDigits[event]:
                    uCellIDsTemp.append(uCellIDs[event][cls])
                    vCellIDsTemp.append(vCellIDs[event][cls])
                    cellChargesTemp.append(cellCharges[event][cls])

            self.set('uCellIDs', np.array(uCellIDsTemp, dtype=object))
            self.set('vCellIDs', np.array(vCellIDsTemp, dtype=object))
            self.set('cellCharges', np.array(cellChargesTemp, dtype=object))

        missing_branches = set(self.clustersFromDigits.digitsOutKeys.values()) - eventKeys
        if includeUnselected and not missing_branches:
            digits = self.clustersFromDigits.get(eventTree, 'outROI')
            digits = {key: digits[key] for key in self.digitKeys.keys()}
            self.extend(digits)

        self.gotDigits = True

    def getMatrices(self, eventTree: TTree = None, matrixSize: tuple = (9, 9), includeUnselected: bool = False) -> None:
        """
        Loads the digit branches into arrays and converts them into adc matrices
        """
        #if self.gotMatrices:
        #    return

        popDigits = False
        if self.gotDigits is False and eventTree:
            self.getDigits(eventTree=eventTree, includeUnselected=includeUnselected)
            popDigits = True

        cellCharges = self.data['cellCharges']
        uCellIDs = self.data['uCellIDs']
        vCellIDs = self.data['vCellIDs']

        matrices = self.generateMatrices.get(cellCharges, uCellIDs, vCellIDs, matrixSize=matrixSize)

        # Combine the results from all chunks
        self.set('matrix', matrices['matrix'])

        if popDigits is True:
            self.data.pop('uCellIDs')
            self.data.pop('vCellIDs')
            self.data.pop('cellCharges')
            self.gotDigits = False
        self.gotMatrices = True

    def getCoordinates(self, eventTree: TTree = None) -> None:
        """
        converting the uv coordinates, together with sensor ids, into xyz coordinates
        """
        #if self.gotCoordinates:
        #    return

        if eventTree:
            self.getClusters(eventTree)
        coordinates = self.clusterCoordinates.get(self['uPosition'], self['vPosition'], self['sensorID'])
        for key, data in coordinates.items():
            self.set(key, data)
        self.gotCoordinates = True

    def getLayers(self, eventTree: TTree = None) -> None:
        if eventTree:
            self.getClusters(eventTree)
        layers = self.clusterCoordinates.layers(self['sensorID'])
        for key, data in layers.items():
            self.set(key, data)

        self.gotLayers = True

    def getMCData(self, eventTree: TTree, includeUnselected: bool = False) -> None:
        """
        this loads the monte carlo from the root file
        """
        #if self.gotMCData:
        #    return

        eventKeys = set(eventTree.keys())
        missing_branches = set(self.clusterKeys.values()) - eventKeys
        if missing_branches:
            mcData = self.mcToDigits.get(eventTree, 'inROI')
        else:
            mcData = self.mcToClusters.get(eventTree)

        for key, data in mcData.items():
            self.set(key, data)

        if includeUnselected:
            mcData = self.mcToDigits.get(eventTree, 'outROI')
            self.extend(mcData)

        self.gotMCData = True
