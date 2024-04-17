import numpy as np
from numpy.typing import ArrayLike
import uproot as ur
from typing import Any, Iterable
import os, warnings
from .detectors import PXD
from .common import FancyDict


class Rootable:
    """
    this class uses uproot to load pxd data from root files and converts them into
    native python data structures.
    it can load the cluster information, uses the digits to generate the adc matrices,
    coordinates, layer and ladders and finally also monte carlo data.
    """
    def __init__(self, data: dict = None) -> None:
        self.pxd = PXD()

        # the root event tree
        self.eventTrees = [None]

        # import flags
        self.gotClusters = False
        self.gotDigits = False
        self.gotMatrices = False
        self.gotCoordinates = False
        self.gotLayers = False
        self.gotSphericals = False
        self.gotMCData = False

    def where(self, *conditions: str) -> dict:
        return self.pxd.where(*conditions)

    @property
    def data(self) -> dict:
        return {'pxd': self.pxd.data}

    def __repr__(self) -> str:
        return repr({'pxd': self.pxd.data})

    def __iter__(self) -> Iterable:
        return iter(self.pxd.data)

    def __len__(self) -> int:
        return len(self.pxd.data)

    def __getitem__(self, index: str | int | ArrayLike) -> FancyDict:
        """
        this makes the class subscriptable, one can retrieve one coloumn by using
        strings as keywords, or get a row by using integer indices or arrays
        """
        if index == 'pxd':
            return FancyDict(self.data['pxd'])
        elif isinstance(index, str):
            return self.data['pxd'][index]
        return FancyDict({key: value[index] for key, value in self.data['pxd'].items()})

    @property
    def numEvents(self) -> int:
        """
        this makes only sense if you use one file
        """
        return len(np.unique(self.pxd['eventNumber']))

    @property
    def numClusters(self) -> int:
        return len(self.pxd['clsCharge'])

    @property
    def particles(self) -> list:
        return np.unique(self.pxd['pdg'])

    def keys(self) -> dict:
        return {'pxd': self.pxd.keys()}

    def items(self) -> dict:
        return {'pxd': self.pxd.items()}

    def values(self) -> dict:
        return {'pxd': self.pxd.values()}

    def get(self, key: str, *args) -> np.ndarray:
        return self.pxd.get(key, *args)

    def pop(self, key: str) -> None:
        return self.pxd.pop(key)

    def stack(self, *columns, toKey: str, pop: bool = True) -> None:
       self.pxd.stack(*columns, toKey=toKey)

    def open(self, *fileNames: str, includeUnselected: bool = False) -> None:
        """
        Reads the file off of the hard drive; it automatically creates event numbers.
        """
        self.eventTrees = []
        self.fileNames = []
        branches = self.pxd.branches(includeUnselected=includeUnselected)

        self.multiplyFiles = True if len(fileNames) > 1 else False
        self.includeUnselected = includeUnselected
        for fileName in fileNames:
            file, _, treeName = fileName.partition(':')
            if not file.endswith('.root'):
                file += '.root'
            if not treeName:
                treeName = 'tree'
            # Setting the file name
            fileBaseName, _ = os.path.splitext(os.path.basename(fileName))
            self.fileNames.append(fileBaseName)
            # Attempting to open the file and tree
            try:
                eventTree = ur.open(f'{file}:{treeName}')
                self.eventTrees.append(eventTree)
                eventKeys = set(eventTree.keys())
                for branch_type, branch_list in branches.items():
                    missing_branches = set(branch_list) - eventKeys
                    if branch_type == 'clusters' and missing_branches:
                        warnings.warn(f"clusters from  '{file}' will be reconstructed from digits,\n this means there might be inaccuricies")
                    elif missing_branches:
                        warnings.warn(f"Missing branches for {branch_type} in '{file}': {missing_branches}")
            except FileNotFoundError:
                raise FileNotFoundError(f"File {file} not found.")

    def getClusters(self) -> None:
        if self.gotClusters:
            warnings.warn('already loaded clusters parameters')
        else:
            for eventTree, fileName in zip(self.eventTrees, self.fileNames):
                self.pxd.getClusters(eventTree, fileName, self.includeUnselected)
            self.gotClusters = True

    def getDigits(self) -> None:
        if self.gotDigits:
            warnings.warn('already loaded cluster digits')
        else:
            for eventTree in self.eventTrees:
                self.pxd.getDigits(eventTree, self.includeUnselected)
            self.gotDigits = True

    def getMatrices(self, matrixSize: tuple = (9, 9)) -> None:
        if self.gotMatrices:
            warnings.warn('already loaded matrices')
        if self.gotDigits:
            self.pxd.getMatrices(eventTree=None, matrixSize=matrixSize, includeUnselected=self.includeUnselected)
        else:
            for eventTree in self.eventTrees:
                self.pxd.getMatrices(eventTree=eventTree, matrixSize=matrixSize, includeUnselected=self.includeUnselected)
        self.gotMatrices = True

    def getCoordinates(self) -> None:
        if self.gotCoordinates:
            warnings.warn('already loaded clusters coordinates')
        if self.gotClusters:
            self.pxd.getCoordinates(None)
        else:
            for eventTree in self.eventTrees:
                self.pxd.getCoordinates(eventTree)
        self.gotCoordinates = True

    def getSphericals(self) -> None:
        if self.gotSphericals:
            warnings.warn('already loaded spherical coordinates')
        if self.gotClusters:
            self.pxd.getSphericals(None)
        else:
            for eventTree in self.eventTrees:
                self.pxd.getSphericals(eventTree)
        self.gotSphericals = True

    def getLayers(self) -> None:
        if self.gotLayers:
            warnings.warn('already loaded clusters layers/ladders')
        if self.gotClusters:
            self.pxd.getLayers(None)
        else:
            for eventTree in self.eventTrees:
                self.pxd.getLayers(eventTree)
        self.gotLayers = True

    def getMCData(self) -> None:
        if self.gotMCData:
            warnings.warn('already loaded clusters mc data')
        for eventTree in self.eventTrees:
            self.pxd.getMCData(eventTree, self.includeUnselected)
        self.gotMCData = True

    def asStructuredArray(self) -> np.ndarray:
        """
        this converts the data dict of this class into a structured numpy array
        """
        # Create a list to hold the dtype specifications
        dtype = []

        # Iterate through the dictionary keys and values
        for key, value in self.pxd.items():
            # Determine the data type of the first value in the list
            sampleValue = value[0]

            if isinstance(sampleValue, np.ndarray):
                # If the value is an array, use its shape and dtype
                shapes = [val.shape for val in value]
                if not all(shape == shapes[0] for shape in shapes):
                       fieldDtype = object
                else:
                    fieldDtype = (sampleValue.dtype, sampleValue.shape)
            else:
                # Otherwise, use the type of the value itself
                fieldDtype = type(sampleValue)

            # Append the key and data type to the dtype list
            dtype.append((key, fieldDtype))

        # Convert the dictionary to a list of tuples
        keys = list(self.pxd.keys())
        dataList = [tuple(self.pxd[key][i] for key in keys) for i in range(len(self.pxd[keys[0]]))]

        # Create the structured array
        structuredArray = np.array(dataList, dtype=dtype)

        return structuredArray

    def asDict(self) -> dict:
        return {'pxd': self.pxd.data}
