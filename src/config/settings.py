"""Global application settings."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application settings."""

    # Base data directory
    DATA_DIR: Path = Field(default=Path("data"))

    # Data subdirectories
    RAW_DIR: Path = Field(init=False, default=Path("raw"))
    PROCESSED_DIR: Path = Field(init=False, default=Path("processed"))

    # Files
    GAMES_FILE: Path = Field(default=Path("games.csv"))
    PLAYERS_FILE: Path = Field(default=Path("players.csv"))
    PLAYER_PLAYS_FILE: Path = Field(default=Path("player_plays.csv"))
    PLAYS_FILE: Path = Field(default=Path("plays.csv"))
    TRACKING_FILES: str = Field(
        default="tracking_week_{week}.csv", description="Template for tracking files."
    )

    # Logging
    LOG_LEVEL: str = Field(default="DEBUG")
    LOG_FORMAT: str = Field(
        default="{asctime}|{filename:>15s}:{lineno:03d}|{levelname:^7s} - {message}"
    )

    def __post_init__(self) -> None:
        """Post initialization to set derived paths."""
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"

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

    def get_tracking_file_path(self, week: int) -> Path:
        """Get tracking file path for given week.

        Args:
            week (int): Week number.

        Raises:
            ValueError: Week must be between 1 and 9.

        Returns:
            Path: Path to tracking file.
        """
        if week < 1 or week > 9:
            raise ValueError("Week must be between 1 and 9.")
        return Path(self.TRACKING_FILES.format(week=week))


settings = Settings()
