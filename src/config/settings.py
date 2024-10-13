"""Global application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

BASE_DIR: Path = Path(__file__).parent.parent.parent.absolute()


class Settings(BaseSettings):
    """Application settings."""

    # Directories
    DATA_DIR: Path = Path.joinpath(BASE_DIR, "data")
    RAW_DIR: Path = Path.joinpath(DATA_DIR, "raw")
    PROCESSED_DIR: Path = Path.joinpath(DATA_DIR, "processed")

    # Files
    GAMES_FILE: Path = Field(
        default=Path("games.csv"), description="Path to games file."
    )
    PLAYERS_FILE: Path = Field(
        default=Path("players.csv"), description="Path to players file."
    )
    PLAYER_PLAYS_FILE: Path = Field(
        default=Path("player_plays.csv"), description="Path to player_play file."
    )
    PLAYS_FILE: Path = Field(
        default=Path("plays.csv"), description="Path to plays file."
    )
    TRACKING_FILES: str = Field(
        default="tracking_week_{week}.csv", description="Template for tracking files."
    )

    # Logging
    LOG_LEVEL: str = Field(default="DEBUG")
    LOG_FORMAT: str = Field(
        default="{asctime}|{filename:>15s}:{lineno:03d}|{levelname:^7s} - {message}"
    )

    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, log_level: str) -> str:
        """Validate log level.

        Args:
            log_level (str): Log level.

        Returns:
            str: Validated log level.
        """
        if log_level not in LOG_LEVELS:
            raise ValueError(f"Invalid log level: {log_level}")

        return log_level

    def tracking_file_path(self, week: int) -> Path:
        """Get tracking file path for given week.

        Args:
            week (int): Week number.

        Raises:
            ValueError: Week must be between 1 and 9.

        Returns:
            Path: Path to tracking file.
        """
        if not 1 <= week <= 9:
            raise ValueError("Week must be between 1 and 9.")

        return Path(self.TRACKING_FILES.format(week=week))

    def get_file_path(self, file: Path, processed: bool = False) -> Path:
        """Get path to file.

        Args:
            file (Path): Path to file.
            processed (bool, optional): Whether to use processed directory.
                Defaults to False (use raw directory).

        Returns:
            Path: Path to file.
        """
        directory = self.PROCESSED_DIR if processed else self.RAW_DIR

        return Path.joinpath(directory, file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application settings.

    Returns:
        Settings: Application settings.
    """
    return Settings()
