import numpy as np
from numpy.typing import ArrayLike
from typing import Iterable, Any
import re


class FancyDict:
    def __init__(self, data: dict = None) -> None:
        self.data = data if data is not None else {}

    def __getitem__(self, index: str | int | ArrayLike):
            """
            this makes the class subscriptable, one can retrieve one coloumn by using
            strings as keywords, or get a row by using integer indices or arrays
            """
            if isinstance(index, str):
                return self.data[index]
            return self.__class__({key: value[index] for key, value in self.data.items()})

    def __setitem__(self, index: str | int | ArrayLike, value: dict | Any) -> None:
        """
        Allows setting the value of a column by using strings as keywords,
        setting the value of a row by using integer indices or arrays,
        or setting a specific value using a tuple of key and index.
        :param index: The column name, row index, or tuple of key and index.
        :param value: The value to set.
        """
        if isinstance(index, str):
            assert len(value) == len(self.data[list(self.data.keys())[0]]), 'value should have same length as data'
            self.data[index] = value
        elif isinstance(index, tuple) and len(index) == 2 and isinstance(index[0], str) and isinstance(index[1], int):
            key, idx = index
            assert key in self.data, f"key {key} not found in data"
            self.data[key][idx] = value
        else:
            assert isinstance(value, dict), "value must be a dictionary when setting rows"
            assert set(value.keys()) == set(self.data.keys()), "keys of value must match keys of data"
            for key in self.data:
                self.data[key][index] = value[key]

    def set(self, keyWord: str, value: list | np.ndarray) -> None:
        """
        an in-place method for setting values
        """
        if keyWord in self.data:
            self.data[keyWord] = np.concatenate((self.data[keyWord], value))
        else:
            self.data[keyWord] = np.array(value)

    def extend(self, value: dict, axis: int = None) -> None:
        """
        an in-place method for extending certain keys
        """
        assert isinstance(value, dict), "value must be a dictionary when setting rows"
        assert set(value.keys()).issubset(set(self.data.keys())), "keys of value must be a subset of keys of data"
        for key in value:
            self.data[key] = np.concatenate((self.data[key], value[key]), axis=axis)

    def where(self, *conditions: str) -> dict:
        """
        Filters the data based on the provided conditions.
        :param conditions: List of conditions as strings for filtering. The keys should be the names of the data fields, and the conditions should be in a format that can be split into key, operator, and value.
        :return: Instance of the class containing the filtered data.
        """
        filteredData = self.data.copy()
        mask = np.ones(len(next(iter(self.data.values()))), dtype=bool)  # Initial mask allowing all elements

        # Applying the conditions to create the mask
        for condition in conditions:
            match = re.match(r'(\w+)\s*([<>=]=?| in )\s*(.+)', condition)
            if match is None:
                raise ValueError(f"Invalid condition: {condition}")

            key, op, value = match.groups()
            op = op.strip()  # remove any leading and trailing spaces

            if op == 'in':
                value = eval(value)
                mask &= np.isin(self.data[key], value)
            else:
                try:
                    # Attempt to convert value to float or boolean
                    if value.lower() in ['true', 'false']:
                        comparisionValue = value.lower() == 'true'
                    else:
                        comparisionValue = float(value)
                except ValueError:
                    # If conversion fails, treat it as a string
                    comparisionValue = value
                
                fieldValues = self.data[key]

                # Determine the correct comparison to apply
                operation = {
                    '==': np.equal,
                    '<': np.less,
                    '>': np.greater,
                    '<=': np.less_equal,
                    '>=': np.greater_equal,
                }.get(op)

                if operation is None:
                    raise ValueError(f"Invalid operator {op}")

                mask &= operation(fieldValues, comparisionValue)

        # Applying the mask to filter the data
        for key, values in filteredData.items():
            filteredData[key] = values[mask]

        return self.__class__(data=filteredData)

    def __repr__(self) -> str:
        return f'fancyDict({repr(self.data)})'

    def __iter__(self) -> Iterable:
        keys = list(self.data.keys())
        numRows = len(self.data[keys[0]])

        for i in range(numRows):
            yield {key: self.data[key][i] for key in keys}

    def __len__(self) -> int:
        return len(self.data)

    def keys(self) -> list:
        return list(self.data.keys())

    def items(self) -> list:
        return self.data.items()

    def values(self) -> list:
        return self.data.values()

    def get(self, key: str) -> np.ndarray:
        return self.data.get(key)

    def pop(self, key: str) -> None:
        return self.data.pop(key)

    @property
    def numClusters(self) -> int:
        key = list(self.keys())[0]
        return len(self.data[key])

    def stack(self, *columns, toKey: str, pop: bool = True) -> None:
        """
        Stacks specified columns into a single column and stores it under a new key.
        :param columns: The columns to stack.
        :param toKey: The new key where the stacked column will be stored.
        :param pop: Whether to remove the original columns.
        """
        # Check that all specified columns exist
        for column in columns:
            if column not in self.data:
                raise KeyError(f"Column '{column}' does not exist.")

        # Column stack the specified columns
        stackedColumn = np.column_stack([self.data[col] for col in columns])

        # Flatten if it's 1D for consistency
        if stackedColumn.shape[1] == 1:
            stackedColumn = stackedColumn.flatten()

        # Store it under the new key
        self.data[toKey] = stackedColumn

        # Remove the original columns if pop is True
        if pop:
            for column in columns:
                self.data.pop(column)
