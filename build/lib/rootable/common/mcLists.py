from numpy.typing import ArrayLike



def findMissing(lst: list, length: int) -> list:
    """
    a private method for finding missing elements in mc data arrays
    """
    return sorted(set(range(0, length)) - set(lst))


def fillMCList(fromClusters: ArrayLike, toClusters: ArrayLike, length: int) -> list:
    """
    a private method for filling MC data arrays where clusters don't have
    any information
    """
    missingIndex = findMissing(fromClusters, length)
    testList = [-1] * length
    fillIndex = 0
    for i in range(len(testList)):
        if i in missingIndex:
            testList[i] = -1
        else:
            try:
                testList[i] = int(toClusters[fillIndex])
            except TypeError:
                testList[i] = int(toClusters[fillIndex][0])
            fillIndex += 1
    return testList
