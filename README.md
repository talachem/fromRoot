# What is this tool?

Let's all be frank, root sucks and the root file format is horrible.
It's among humanities worst pieces of software. With this small tool I hope to fix
the damage that was done, at least a little, by converting root files into a
native Python formats. Thus one will be able to build an user friendly analysis
library on top of this small library.

It's using [Numpy](https://numpy.org) and a library called [Uproot](https://github.com/scikit-hep/uproot5)
to read and process these damn root files. I cannot heap enough praise on how
great Uproot actually is. It's a godsend. So far it is specialist for one task,
which is to extract PXD data from Belle 2 data files. I will have to work on it
to make it actually viable for more use cases.

This tool is still in early development, which means that the source code is horrible
and that not all features work properly or aren't even fully implemented. Right now
only PXD is supported by this tool. In the future I plan to include more detectors.

The bad code quality is also due to the fact, that the data format is just horrendous
and no one should be forced to work with this. Getting the data to a usable state took
quite a lot of work and reverser engineering of what could only be described as the
concoction of a mad man.

## How to use this?

This is a single class, that needs to be instantiated, it doesn't take any arguments.
Just import it like this:

```python
from rootable import Rootable
```

Then you can create an instance:

```python
loadFromRoot = Rootable()
```

and load the root file and all the data:

```python
loadFromRoot.open('/root-files/slow_pions_2.root')
loadFromRoot.getClusters()
loadFromRoot.getCoordinates()
loadFromRoot.getLayers()
loadFromRoot.getDigits()
loadFromRoot.getMatrices()
loadFromRoot.getMCData()
```

The user can define which tree is to be loaded by adding its name using a colon:

```python
loadFromRoot.open('/root-files/slow_pions_2.root:tree')
```

This is not necessary, because the code defaults to 'tree' as the tree name.

One can as well open several files at once:

```python
loadFromRoot.open('/root-files/slow_pions_2.root', '/root-files/QED.root')
```

One can now specify that ROI unselected digits should be read and to reconstruct
the cluster data from them. this is still iffy, after including ROI unselected
clusters, one cannot load monte carlo information and the u/v mapping is still
very wonky.

```python
loadFromRoot.open('/root-files/slow_pions_2.root', includeUnselected=True)
```

So far mixing opening multiply files, with and without ROI unselected clusters
doesn't work.


The 'get' commands don't have any return value, but instead work in-place.
Then all data is stored inside the object as dict:

```python
loadFromRoot.data
```

Here follows a list of keywords contained in the dict:

- cluster data:
    - 'eventNumber': int
    - 'clsCharge': int
    - 'seedCharge': int
    - 'clsSize': int
    - 'uSize': int
    - 'vSize': int
    - 'uPosition': float
    - 'vPosition': float
    - 'sensorID': int
    - 'detector': str
    - 'roiSelected': bool
    - 'fileName': str
- coordinates:
    - 'xPosition': float
    - 'yPosition': float
    - 'zPosition': float
- layers:
    - 'layer': int
    - 'ladder': int
- digits:
    - 'uCellIDs': array
    - 'vCellIDs': array
    - 'cellCharges': array
- matrices:
    - 'matrix': array
- Monte Carlo data:
    - 'momentumX': float
    - 'momentumY': float
    - 'momentumZ': float
    - 'pdg': int
    - 'clsNumber': int

Since the class is subscriptable one can access every element directly using the keywords
like this:

```python
loadFromRoot['eventNumber']
```

or

```python
loadFromRoot[0]
```

will return either the array containing the event numbers of the first entry of every
array contained in the classes dict.

It is possible to filter through the data:

```python
loadFromRoot.where('clsSize == 1')
loadFromRoot.where('clsSize > 1')
loadFromRoot.where('clsSize > 1', 'layer == 1')
```

or even:

```python
loadFromRoot.where('eventNumber in [0,1,2]')
```

And finally you can convert the dict into a structured Numpy array by simply writing:

```python
loadFromRoot.asStructuredArray()
```

This last command returns a Numpy array. From there the user can save it using
Numpys build-in functions, convert it to Pandas or use it in any way that is
compatible with Numpy.

Alternatively one can get it as a pandas dataframe, which doesn't handle 2D array
properly. So if one uses the pixel matrices a dataframe is not advisable.

```python
loadFromRoot.asDataFrame(popMatrices=True)
```


The class itself is iterable, it's a bit different from typical python dicts,
I iterate over rows and return it as a dict, not sure if that's actually useful.

In certain instances it can be very usefull to stack certain columns together, for
example when one wants to calculate the distance from the origin. Then one can
stack the positions:

```python
loadFromRoot.stack('xPosition', 'yPosition', 'zPosition', toKey: 'position')
```


## Installation

You will need to the [wheel](https://pypi.org/project/wheel/) and [setuptools](https://pypi.org/project/setuptools/) packages of python in order to install
Download the repo, navigate in the terminal to the folder and run the following script:

```zsh
 pip3 install .
```

or

```zsh
pip install .
```

It depends on how python and/or your `PATH` is configured.
