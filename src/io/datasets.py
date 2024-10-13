"""Dataset readers and writers."""

from pathlib import Path

import pandas as pd

from .base import BaseReader, BaseWriter


class CSVReader(BaseReader):
    """Reader for CSV files."""

    def read(self, path: Path) -> pd.DataFrame:
        """Read data from CSV file.

        Args:
            path (Path): Path to CSV file.

        Returns:
            pd.DataFrame: Data read from CSV file.
        """
        data = pd.read_csv(path)

        if self.limit is not None:
            data = data.head(self.limit)

        return data


class CSVWriter(BaseWriter):
    """Writer to CSV files."""

    def write(self, path: Path, data: pd.DataFrame) -> None:
        """Write data to CSV file.

        Args:
            path (Path): Path to CSV file.
            data (pd.DataFrame): Data to write.
        """
        data.to_csv(path)
