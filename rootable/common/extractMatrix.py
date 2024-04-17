import sys
import numpy as np


def extractMatrix(matrixLadder: np.ndarray, uCellID: int, vCellID: int, eventNumber, matrixSize: tuple = (9,9)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #Extracts a 9x9 sub-matrix centered around the seed charge from a specific detector array.
    # Initial clipping to get the 9x9 matrix around (x, y)

    assert matrixSize[0] % 2 != 0, 'matrix size in u is not uneven -> seed pixel not centered'
    assert matrixSize[1] % 2 != 0, 'matrix size in v is not uneven -> seed pixel not centered'

    assert matrixSize[0] < 250, 'matrix size in u is larger than the ladder'
    assert matrixSize[1] < 768, 'matrix size in v is larger than the ladder'

    uCenter, vCenter = matrixSize[0] // 2, matrixSize[1] // 2

    uLower = np.clip(uCellID - uCenter, a_min=0, a_max=matrixLadder.shape[0] - matrixSize[0])
    uUpper = np.clip(uCellID + uCenter + 1, a_min=0, a_max=matrixLadder.shape[0])
    vLower = np.clip(vCellID - vCenter, a_min=0, a_max=matrixLadder.shape[1] - matrixSize[1])
    vUpper = np.clip(vCellID + vCenter + 1, a_min=0, a_max=matrixLadder.shape[1])

    matrix = matrixLadder[uLower:uUpper, vLower:vUpper]

    # Find the position of the seed charge (maximum charge) in the initial matrix
    seedChargePos = np.unravel_index(matrix.argmax(), matrix.shape)

    # Check if max charge pixel is not at the center and re-center if necessary
    if not seedChargePos == (uCenter, vCenter):
        # Calculate how much to roll in each direction to center the max charge
        uOffset, vOffset = seedChargePos[0] - uCenter, seedChargePos[1] - vCenter

        # Adjust the x and y lower bounds to center the max charge
        uLower = np.clip(uCellID + uOffset - uCenter, a_min=0, a_max=matrixLadder.shape[0] - matrixSize[0])
        vLower = np.clip(vCellID + vOffset - vCenter, a_min=0, a_max=matrixLadder.shape[1] - matrixSize[1])

        # Recalculate the matrix now centered around the max charge
        matrix = matrixLadder[uLower:uLower+matrixSize[0], vLower:vLower+matrixSize[1]]

    border = np.concatenate((matrix[0], matrix[-1], matrix[:,0], matrix[:,-1]))
    borderCounts = np.count_nonzero(border)
    bounderyConditions = (uLower <= 0), (uUpper >= matrixLadder.shape[0]), (vLower <= 0), (vUpper >= matrixLadder.shape[1])
    if borderCounts > 0 and not any(bounderyConditions):
        uSize = matrixSize[0] + 2
        vSize = matrixSize[1] + 2
        return extractMatrix(matrixLadder, uCellID, vCellID, eventNumber=eventNumber, matrixSize=(uSize,vSize))

    # Find non-zero pixels in the centered submatrix
    nonZeroPositions = np.nonzero(matrix)

    # Calculate the global coordinates for each non-zero pixel
    globalUPositions = nonZeroPositions[0] + uLower
    globalVPositions = nonZeroPositions[1] + vLower

    seedChargePos = np.unravel_index(matrix.argmax(), matrix.shape)
    seedUGlobal = seedChargePos[0] + uLower
    seedVGlobal = seedChargePos[1] + vLower

    return matrix, globalUPositions, globalVPositions, seedUGlobal, seedVGlobal
