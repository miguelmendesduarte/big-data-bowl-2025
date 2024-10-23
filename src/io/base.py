"""Base classes for file I/O."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class BaseReader(ABC):
    """Base class for file readers."""

    def __init__(self, limit: Optional[int] = None) -> None:
        """Initialize BaseReader.

        Args:
            limit (Optional[int], optional): Maximum number of rows to read.
                Defaults to None.
        """
        self.limit = limit

    @abstractmethod
    def read(self, path: Path) -> Any:
        """Read data from file.

        Args:
            path (Path): Path to file.

        Raises:
            FileNotFoundError: If file does not exist.
            NotImplementedError: To be implemented in child class.

        Returns:
            Any: Data read from file.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        raise NotImplementedError()


class BaseWriter(ABC):
    """Base class for file writers."""

    @abstractmethod
    def write(self, path: Path, data: Any) -> None:
        """Write data to file.

        Args:
            path (Path): Path to file.
            data (Any): Data to write.

        Raises:
            FileNotFoundError: If file does not exist.
            NotImplementedError: To be implemented in child class.
        """
        raise NotImplementedError()
