"""Command line entry point: `python -m retarget_lab run --config config/default.yaml`."""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="Humanoid rig & motion retargeting lab.")
DEFAULT = "config/default.yaml"


@app.command()
def run(
    config: str = typer.Option(DEFAULT, help="run configuration"),
    clips: str = typer.Option("", help="comma-separated clip ids (default: all)"),
    skip_render: bool = typer.Option(False, help="skip the Blender side-by-side renders"),
    skip_blender: bool = typer.Option(False, help="reuse existing Blender outputs; redo only the analysis"),
    force_character: bool = typer.Option(False, help="recreate the MB-Lab character .blend"),
):
    """Full pipeline: character -> skeletons -> retarget every clip -> checks -> renders -> report -> export."""
    from .runner import run_all

    run_all(config, [c for c in clips.split(",") if c] or None, skip_render, skip_blender, force_character)


@app.command("fetch-data")
def fetch_data(config: str = DEFAULT):
    """Download the configured CMU BVH clips (skips files that already exist)."""
    from . import pipeline as P

    P.step_fetch_data(P.load_ctx(config))


@app.command()
def character(config: str = DEFAULT, force: bool = False):
    """Create assets/mblab_character.blend headlessly with MB-Lab."""
    from . import pipeline as P

    P.step_character(P.load_ctx(config), force=force)


@app.command()
def skeleton(config: str = DEFAULT):
    """Export the MB-Lab skeleton JSON and both hierarchy diagrams."""
    from . import pipeline as P

    P.step_skeletons(P.load_ctx(config))
