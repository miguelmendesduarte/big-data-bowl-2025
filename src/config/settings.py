"""Global application settings."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

BASE_DIR: Path = Path(__file__).parent.parent.parent.absolute()

# Filenames
GAMES_FILENAME = "games.csv"
PLAYERS_FILENAME = "players.csv"
PLAYER_PLAYS_FILENAME = "player_plays.csv"
PLAYS_FILENAME = "plays.csv"
TRACKING_FILENAME_TEMPLATE = "tracking_week_{week}.csv"
LOGO_FILENAME_TEMPLATE = "{team}.png"


class LogLevel(StrEnum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings."""

    # Directories
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    ASSETS_DIR: Path = BASE_DIR / "assets"
    LOGOS_DIR: Path = ASSETS_DIR / "logos"

    # Files
    GAMES_FILE: Path = Field(
        default=Path(GAMES_FILENAME), description="Path to games file."
    )
    PLAYERS_FILE: Path = Field(
        default=Path(PLAYERS_FILENAME), description="Path to players file."
    )
    PLAYER_PLAYS_FILE: Path = Field(
        default=Path(PLAYER_PLAYS_FILENAME), description="Path to player plays file."
    )
    PLAYS_FILE: Path = Field(
        default=Path(PLAYS_FILENAME), description="Path to plays file."
    )
    TRACKING_FILES: str = Field(
        default=TRACKING_FILENAME_TEMPLATE, description="Template for tracking files."
    )
    LOGO_FILES: str = Field(
        default=LOGO_FILENAME_TEMPLATE, description="Template for team logo."
    )

    # Logging
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO, description="Log level to use.")
    LOG_FORMAT: str = Field(
        default="{asctime}|{filename:>15s}:{lineno:03d}|{levelname:^7s} - {message}"
    )

    def get_data_file_path(self, file: Path, processed: bool = False) -> Path:
        """Get path to file in data directory.

        Args:
            file (Path): Path to file.
            processed (bool, optional): Whether to use processed directory.
                Defaults to False (use raw directory).

        Returns:
            Path: Path to file.
        """
        directory = self.PROCESSED_DIR if processed else self.RAW_DIR

        return directory / file

    def get_tracking_file_path(self, week: int, processed: bool = False) -> Path:
        """Get path to tracking file.

        Args:
            week (int): Week number.
            processed (bool, optional): Whether to use processed directory.
                Defaults to False - use raw directory.

        Raises:
            ValueError: Week must be between 1 and 9.

        Returns:
            Path: Path to tracking file.
        """
        if not 1 <= week <= 9:
            raise ValueError("Week must be between 1 and 9.")

        tracking_file_path: Path = Path(self.TRACKING_FILES.format(week=week))

        return self.get_data_file_path(tracking_file_path, processed=processed)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application settings.

    Returns:
        Settings: Application settings.
    """
    return Settings()
