import numpy as np


def calcSpherical(xPosition: np.ndarray, yPosition: np.ndarray, zPosition: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    a function for converting xyz to spherical coordinates
    """
    xSquare = np.square(xPosition)
    ySquare = np.square(yPosition)
    zSquare = np.square(zPosition)

    # Avoid division by zero by replacing zeros with a small number
    r = np.sqrt(xSquare + ySquare + zSquare)
    rSafe = np.where(r == 0, 1e-10, r)

    theta = np.arccos(zPosition / rSafe)
    phi = np.arctan2(yPosition, xPosition)

    return r, theta, phi
