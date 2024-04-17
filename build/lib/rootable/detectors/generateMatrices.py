import numpy as np
from numpy.typing import ArrayLike
from concurrent.futures import ThreadPoolExecutor
from time import time


class GenerateMatrices:
    def __init__(self) -> None:
        pass

    def get(self, cellCharges: np.ndarray, uCellIDs: np.ndarray, vCellIDs: np.ndarray, matrixSize: tuple = (9, 9), order: str = 'uv') -> dict:
        assert order == 'uv' or order == 'vu', f"{order} is not a proper order, 'uv' or 'vu' are the only options"

        lengthes = np.vectorize(len)(cellCharges)
        uniqueLengthes = np.unique(lengthes)
        plotRange = np.array(matrixSize) // 2
        matrices = np.zeros((len(cellCharges), *matrixSize), dtype=int)

        for length in uniqueLengthes:
            indices = np.where(lengthes == length)[0]
            uCells = np.vstack(uCellIDs[indices])
            vCells = np.vstack(vCellIDs[indices])
            charges = np.vstack(cellCharges[indices])

            maxIndices = charges.argmax(1)
            uMax = uCells[np.arange(len(indices)), maxIndices]
            vMax = vCells[np.arange(len(indices)), maxIndices]

            uPos = uCells + plotRange[0] - np.repeat(uMax, length).reshape(-1, length)
            vPos = vCells + plotRange[1] - np.repeat(vMax, length).reshape(-1, length)

            valid_indices = ((uPos >= 0) & (uPos < matrixSize[0]) & (vPos >= 0) & (vPos < matrixSize[1])).flatten()

            # Flatten uPos, vPos, and charges with valid indices
            uPosFlat = uPos.flatten()[valid_indices]
            vPosFlat = vPos.flatten()[valid_indices]
            chargesFlat = charges.flatten()[valid_indices]

            # Calculate flat indices for assignment
            clusterIndicesFlat = np.repeat(indices, length)[valid_indices]
            if order == 'uv':
                flatIndices = np.ravel_multi_index((clusterIndicesFlat, uPosFlat, vPosFlat), dims=matrices.shape)
            else:
                flatIndices = np.ravel_multi_index((clusterIndicesFlat, vPosFlat, uPosFlat), dims=matrices.shape)

            # Assign values
            matrices.ravel()[flatIndices] = chargesFlat

        return {'matrix': matrices}
