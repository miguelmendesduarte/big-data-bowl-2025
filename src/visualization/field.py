"""Module for creating the football field for visualization."""

import matplotlib.image as mpl_image
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..config.settings import get_settings
from ..core.teams import Team

FIELD_LENGTH = 120
FIELD_WIDTH = 53.3
FIELD_COLOR = "#3B7A57"
LINE_COLOR = "#FFFFFF"

FIGURE_SIZE: tuple[int, int] = (12, 6)

settings = get_settings()


def _draw_field(ax: Axes) -> None:
    """Draw the main football field rectangle.

    Args:
        ax (Axes): Axes to draw on.
    """
    field = patches.Rectangle(
        (0, 0),
        FIELD_LENGTH,
        FIELD_WIDTH,
        linewidth=0,
        edgecolor=LINE_COLOR,
        facecolor=FIELD_COLOR,
    )
    ax.add_patch(field)


def _add_yard_number(ax: Axes, line: int) -> None:
    """Add yard number at the specified line.

    Args:
        ax (Axes): Axes to draw on.
        line (int): Yard number line.
    """
    number = f"{line - 10 if line <= 50 else 120 - line - 10}"
    ax.text(
        line,
        FIELD_WIDTH / 6,
        number,
        fontsize=16,
        color=LINE_COLOR,
        horizontalalignment="center",
    ).set_alpha(0.8)
    ax.text(
        line,
        5 * FIELD_WIDTH / 6,
        number,
        fontsize=16,
        color=LINE_COLOR,
        horizontalalignment="center",
        rotation=180,
    ).set_alpha(0.8)


def _add_triangles(ax: Axes, line: int) -> None:
    """Add triangles below and above the yard number.

    Args:
        ax (Axes): Axes to draw on.
        line (int): Yard number line.
    """
    if line < 60:
        _draw_left_triangle(ax, line, -2.5, FIELD_WIDTH / 6 + 1.25)
        _draw_left_triangle(ax, line, -2.5, 5 * FIELD_WIDTH / 6 - 0.25)
    if line > 60:
        _draw_right_triangle(ax, line, 2.5, FIELD_WIDTH / 6 + 1.25)
        _draw_right_triangle(ax, line, 2.5, 5 * FIELD_WIDTH / 6 - 0.25)


def _draw_left_triangle(ax: Axes, line: int, x_offset: float, y_offset: float) -> None:
    """Draw triangle at the specified position (10 to 40 yard lines).

    Args:
        ax (Axes): Axes to draw on.
        line (int): Yard number line.
        x_offset (float): Offset in x direction.
        y_offset (float): Offset in y direction.
    """
    triangle = patches.Polygon(
        [
            (line + x_offset, y_offset),
            (line + (x_offset + 1), y_offset + 0.25),
            (line + (x_offset + 1), y_offset - 0.25),
        ],
        color=LINE_COLOR,
        alpha=0.5,
    )
    ax.add_patch(triangle)


def _draw_right_triangle(ax: Axes, line: int, x_offset: float, y_offset: float) -> None:
    """Draw triangle at the specified position (60 to 100 yard lines).

    Args:
        ax (Axes): Axes to draw on.
        line (int): Yard number line.
        x_offset (float): Offset in x direction.
        y_offset (float): Offset in y direction.
    """
    triangle = patches.Polygon(
        [
            (line + x_offset, y_offset),
            (line + (x_offset - 1), y_offset + 0.25),
            (line + (x_offset - 1), y_offset - 0.25),
        ],
        color=LINE_COLOR,
        alpha=0.5,
    )
    ax.add_patch(triangle)


def _draw_yard_numbers(ax: Axes) -> None:
    """Draw yard numbers and triangles on the field.

    Args:
        ax (Axes): Axes to draw on.
    """
    for line in range(20, 101, 5):
        if line % 10 == 0:
            _add_yard_number(ax, line)
            _add_triangles(ax, line)


def _draw_field_lines(ax: Axes) -> None:
    """Draw the field lines.

    Args:
        ax (Axes): Axes to draw on.
    """
    for line in range(10, 111):
        if line % 5 == 0:
            linewidth = 1.5 if line in {10, 110} else 1
            ax.plot(
                (line, line),
                (0, FIELD_WIDTH),
                color=LINE_COLOR,
                linewidth=linewidth,
                alpha=0.8,
            )
        elif line in {12, 108}:
            ax.plot(
                (line, line),
                (FIELD_WIDTH / 2 - 0.25, FIELD_WIDTH / 2 + 0.25),
                color=LINE_COLOR,
                linewidth=0.5,
            )
        _draw_inner_lines(ax, line)


def _draw_inner_lines(ax: Axes, line: int) -> None:
    """Draw the inner lines.

    Args:
        ax (Axes): Axes to draw on.
        line (int): Yard number line.
    """
    offsets = [0, FIELD_WIDTH * 4 / 10, FIELD_WIDTH * 6 / 10, FIELD_WIDTH - 0.5]
    for offset in offsets:
        ax.plot((line, line), (offset, offset + 0.5), color=LINE_COLOR, linewidth=0.5)


def _draw_endzone_text(ax: Axes, home_team: Team) -> None:
    """Draw endzone text.

    Args:
        ax (Axes): Axes to draw on.
        home_team (Team): Home team.
    """
    _draw_endzone(
        ax,
        5,
        home_team.location,
        home_team.primary_color,
        home_team.secondary_color,
        90,
    )
    _draw_endzone(
        ax, 115, home_team.name, home_team.primary_color, home_team.secondary_color, -90
    )


def _draw_endzone(
    ax: Axes,
    x_position: int,
    text: str,
    primary_color: str,
    secondary_color: str,
    rotation: int,
) -> None:
    """Draw endzone text.

    Args:
        ax (Axes): Axes to draw on.
        x_position (int): X position.
        text (str): Text to write on the endzone.
        primary_color (str): Primary color of the home team.
        secondary_color (str): Secondary color of the home team.
        rotation (int): Rotation of the text.
    """
    ax.text(
        x_position,
        FIELD_WIDTH / 2,
        text,
        fontsize=42,
        color=primary_color,
        horizontalalignment="center",
        weight="bold",
        verticalalignment="center",
        rotation=rotation,
        path_effects=[path_effects.withStroke(linewidth=2, foreground=secondary_color)],
    ).set_alpha(0.5)


def _draw_team_logo(ax: Axes, home_team: Team) -> None:
    """Draw team logo in the middle of the field.

    Args:
        ax (plt.Axes): Axes to draw on.
        home_team (Team): Home team.
    """
    logo = mpl_image.imread(home_team.get_logo_file_path())
    ax.imshow(
        logo,
        aspect="auto",
        zorder=1,
        alpha=0.3,
        extent=(
            FIELD_LENGTH / 2 - 7,
            FIELD_LENGTH / 2 + 7,
            FIELD_WIDTH / 2 - 7,
            FIELD_WIDTH / 2 + 7,
        ),
    )


def create_football_field(home_team: Team) -> Axes:
    """Create the football field.

    Args:
        home_team (Team): Home team.

    Returns:
        Axes: Axes to draw on.
    """
    _, ax = plt.subplots(figsize=FIGURE_SIZE)

    _draw_field(ax)
    _draw_yard_numbers(ax)
    _draw_field_lines(ax)
    _draw_endzone_text(ax, home_team)
    _draw_team_logo(ax, home_team)

    # Add noise to the field for realism
    noise = np.random.rand(200, 200)
    ax.imshow(noise, extent=(0, FIELD_LENGTH, 0, FIELD_WIDTH), cmap="Greys", alpha=0.05, zorder=1)

    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.axis("off")
    plt.tight_layout()
    
    return ax
