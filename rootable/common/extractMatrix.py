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


def genLadder(uCells, vCells, charges, size=(250,768)):
    ladder = np.zeros(size, dtype=int)
    ladder[uCells, vCells] = charges
    return ladder


def genCluster(uCells, vCells, charges, size=(250,786)):
    """
    this function uses recursion to find clusters. first it generates the whole ladder
    and then it traveses, using recursion, the ladder to find all connected pixels
    the output format is a bit stupid and I apologize for that.
    the first index counts the cluster, the first/second coloumn are u/v cells
    and the last coloumn are the charges
    """
    ladder = genLadder(uCells, vCells, charges, size)
    # Initialize list to keep track of the clusters
    clusters = []
    # Set to keep track of visited cells to avoid infinite loops
    visited = set()

    def travel(uCell, vCell, current_cluster):
        # Check bounds
        if uCell < 0 or uCell >= ladder.shape[0] or vCell < 0 or vCell >= ladder.shape[1]:
            return
        # Check if the cell is empty or already visited
        if ladder[uCell, vCell] == 0 or (uCell, vCell) in visited:
            return
        # Mark the cell as visited
        visited.add((uCell, vCell))
        # Add the cell to the cluster
        current_cluster.append([uCell, vCell, ladder[uCell, vCell]])
        # Explore all four directions
        travel(uCell, vCell + 1, current_cluster)
        travel(uCell, vCell - 1, current_cluster)
        travel(uCell + 1, vCell, current_cluster)
        travel(uCell - 1, vCell, current_cluster)

    # Iterate over the given non-zero indices
    for u, v in zip(uCells, vCells):
        if ladder[u, v] != 0 and (u, v) not in visited:
            current_cluster = []
            travel(u, v, current_cluster)
            if current_cluster:
                clusters.append(current_cluster)

    return clusters


def genMatrices(clusters, size=(9,9)):
    num_clusters = len(clusters)
    matrices = np.zeros((num_clusters, *size), dtype=int)

    for i, cluster in enumerate(clusters):
        current = np.array(cluster)
        maxIndex = np.argmax(current[:, 2])

        # Coordinates of the cell with the maximum charge
        max_u, max_v = current[maxIndex, 0], current[maxIndex, 1]

        # Calculate offsets to center the maximum charge cell
        offset_u = int(size[0]/2) - max_u
        offset_v = int(size[1]/2) - max_v

        # Apply offsets and filter cells that fall within the 9x9 matrix bounds
        new_coords = current[:, :2] + [offset_u, offset_v]
        valid_coords = (new_coords[:, 0] >= 0) & (new_coords[:, 0] < size[0]) & \
                       (new_coords[:, 1] >= 0) & (new_coords[:, 1] < size[1])

        new_coords = new_coords[valid_coords]
        charges = current[valid_coords, 2]

        matrices[i, new_coords[:, 0], new_coords[:, 1]] = charges

    return matrices
