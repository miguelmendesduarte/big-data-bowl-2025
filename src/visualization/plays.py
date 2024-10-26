"""Module for visualizing plays."""

import logging

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import PillowWriter
from matplotlib.artist import Artist
from matplotlib.patches import Ellipse

from ..core.teams import TEAMS
from .field import FIELD_WIDTH, create_football_field

BALL_COLOR = "#8B5B29"
LOS_COLOR = "#0000FF"
FIRST_DOWN_COLOR = "#FFFF00"

logger = logging.getLogger(__name__)


def animate_play(
    game_id: int,
    play_id: int,
    games_data: pd.DataFrame,
    plays_data: pd.DataFrame,
    tracking_data: pd.DataFrame,
    save: bool = False,
) -> None:
    """Animate a play based on tracking data.

    Args:
        game_id (int): ID of the game.
        play_id (int): ID of the play.
        games_data (pd.DataFrame): Game data.
        plays_data (pd.DataFrame): Play data.
        tracking_data (pd.DataFrame): Tracking data.
        save (bool, optional): Whether to save the animation. Defaults to False.

    Raises:
        ValueError: If the game or play IDs are not found in the data.
    """
    if game_id not in games_data.gameId.values:
        raise ValueError(f"Game ID {game_id} not found in games data.")
    if play_id not in plays_data.playId.values:
        raise ValueError(f"Play ID {play_id} not found in plays data.")

    logger.info(f"Animating play {play_id} of game {game_id}")

    home_team_abbr = games_data[games_data.gameId == game_id].homeTeamAbbr.iloc[0]
    visitor_team_abbr = games_data[games_data.gameId == game_id].visitorTeamAbbr.iloc[0]
    home_team = TEAMS[home_team_abbr]
    visitor_team = TEAMS[visitor_team_abbr]

    logger.info(f"Home team: {home_team.full_name}")
    logger.info(f"Visitor team: {visitor_team.full_name}")

    fig, ax = create_football_field(home_team=home_team)

    tracking_data = tracking_data[
        (tracking_data.gameId == game_id) & (tracking_data.playId == play_id)
    ]
    play_data = plays_data[
        (plays_data.gameId == game_id) & (plays_data.playId == play_id)
    ]

    line_of_scrimmage = play_data.yardlineNumber.iloc[0]
    first_down = play_data.yardlineNumber.iloc[0] + play_data.yardsToGo.iloc[0]
    down = play_data.down.iloc[0]
    play_description = play_data.playDescription.iloc[0]

    logger.info(f"Line of scrimmage: {line_of_scrimmage} yard line")
    logger.info(f"First down: {first_down} yard line")
    logger.info(f"Down: {down}")
    logger.info(f"Play description: {play_description}")

    # Draw line of scrimmage
    ax.plot(
        (line_of_scrimmage + 10, line_of_scrimmage + 10),
        (0, FIELD_WIDTH),
        color=LOS_COLOR,
        linewidth=1.2,
        alpha=0.8,
    )
    # Draw first down marker (if not goal to go)
    if first_down < 110:
        ax.plot(
            (
                first_down + 10,
                first_down + 10,
            ),
            (0, FIELD_WIDTH),
            color=FIRST_DOWN_COLOR,
            linewidth=1.2,
            alpha=0.8,
        )

    player_scatter: list[Artist] = []
    player_texts: list[Artist] = []
    ball_ellipse = None

    def update(frame: int) -> list[Artist]:
        """Update the animation.

        Args:
            frame (int): Frame number.

        Returns:
            list[Artist]: List of artists updated.
        """
        nonlocal player_scatter, ball_ellipse, player_texts
        for scatter in player_scatter:
            scatter.remove()
        player_scatter = []

        for text in player_texts:
            text.remove()
        player_texts = []

        if ball_ellipse is not None:
            ball_ellipse.remove()

        current_frame_data = tracking_data[tracking_data.frameId == frame]
        ball_data = current_frame_data[current_frame_data.nflId.isna()]
        player_data = current_frame_data[~current_frame_data.nflId.isna()]

        # Football
        if not ball_data.empty:
            ball_x = ball_data.x.values[0] + 0.31
            ball_y = ball_data.y.values[0]

            ball_ellipse = Ellipse(
                xy=(ball_x, ball_y),
                width=0.8,
                height=0.5,
                facecolor=BALL_COLOR,
                edgecolor="black",
                linewidth=1,
                zorder=4,
            )
            ax.add_patch(ball_ellipse)

        # Players
        for _, row in player_data.iterrows():
            x = row.x
            y = row.y
            team_color = (
                (home_team.primary_color, home_team.secondary_color)
                if row.club == home_team.abbreviation
                else (visitor_team.primary_color, visitor_team.secondary_color)
            )
            jersey_number = row.jerseyNumber

            scatter = ax.scatter(
                x,
                y,
                s=110,
                facecolor=team_color[0],
                edgecolor=team_color[1],
                linewidth=1,
                zorder=2,
            )
            text = ax.text(
                x,
                y,
                f"{int(jersey_number)}",
                color=team_color[1],
                ha="center",
                va="center",
                fontsize=5,
                fontweight="bold",
                zorder=3,
            )

            player_scatter.append(scatter)
            player_texts.append(text)

        return player_scatter + player_texts + ([ball_ellipse] if ball_ellipse else [])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=tracking_data.frameId.unique(),
        interval=100,
        repeat=True,
        repeat_delay=1000,
    )

    if save:
        ani.save(f"{game_id}_{play_id}.gif", writer=PillowWriter(fps=10, bitrate=1800))

    plt.show()
