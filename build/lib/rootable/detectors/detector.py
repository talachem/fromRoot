from abc import ABC, abstractmethod


class Detector(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def panels(self):
        pass

    @abstractmethod
    def getCoordinates(self):
        pass

    @abstractmethod
    def getClusters(self):
        pass

    @abstractmethod
    def getMCData(self):
        pass

    @abstractmethod
    def getLayers(self):
        pass
