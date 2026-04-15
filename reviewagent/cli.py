from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# Windows: default loopback so http://127.0.0.1 works; binding 0.0.0.0 is IPv4-only and
# ``localhost`` often resolves to ::1; mapping to :: without dual-stack breaks IPv4 loopback.
_DEFAULT_SERVER_HOST = "127.0.0.1" if sys.platform == "win32" else "0.0.0.0"

from reviewagent.agent import create_review_orchestrator


def _emit_review_batch_result(paths: list[Path], out: dict) -> None:
    """Single-file output matches legacy shape; multiple files use a batch wrapper."""
    if len(paths) == 1:
        r = out["results"][0]
        drop = {"index", "path", "filename", "inferred_content_type"}
        click.echo(
            json.dumps(
                {k: v for k, v in r.items() if k not in drop},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        click.echo(json.dumps(out, indent=2, ensure_ascii=False))


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug mode")
def cli(debug):
    pass


@cli.command()
@click.argument("content", required=False)
@click.option(
    "--file",
    "-f",
    "file_paths",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Local file to review (repeat -f for multiple files, in order)",
)
def check(content, file_paths):
    """Review inline text, or load one or more files with -f/--file."""
    orchestrator = create_review_orchestrator()

    if file_paths:
        from reviewagent.api.batch import moderate_paths_sync
        from reviewagent.ingest import load_local_file_for_review

        p_list = list(file_paths)
        if len(p_list) == 1:
            fp = p_list[0]
            try:
                inferred_ct, payload = load_local_file_for_review(fp)
            except (OSError, ValueError) as e:
                raise click.ClickException(str(e)) from e

            result = orchestrator.moderate_payload(inferred_ct, payload)
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            return

        out = moderate_paths_sync(p_list)
        _emit_review_batch_result(p_list, out)
        return

    if content is None:
        raise click.UsageError(
            "Provide text to review, or use --file /path/to/file (repeat -f for multiple files)."
        )

    user_input = (
        "First infer the content type (text/image/video/audio), then review accordingly. "
        "If the input is a local path or URL, choose the right tools and return the final JSON.\n\n"
        f"Input to review: {content}"
    )
    result = orchestrator.run_sync(user_input)
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


@cli.command("file")
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def file_cmd(paths: tuple[Path, ...]) -> None:
    """Load one or more local files and review (type inferred automatically)."""
    from reviewagent.api.batch import moderate_paths_sync

    p_list = list(paths)
    if not p_list:
        raise click.UsageError("Pass at least one file path.")
    out = moderate_paths_sync(p_list)
    _emit_review_batch_result(p_list, out)


@cli.command()
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def checkfile(paths: tuple[Path, ...]) -> None:
    """[compat] Same as `file`, supports multiple paths."""
    file_cmd(paths)


@cli.command()
def interactive():
    orchestrator = create_review_orchestrator()

    click.echo("Content review REPL (type 'quit' to exit)")
    click.echo("-" * 40)

    while True:
        user_input = click.prompt("Input")

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        result = orchestrator.run_sync(user_input)
        if result.get("success"):
            click.echo(f"\nReview: {result['response']}\n")
        else:
            click.echo(f"\nError: {result.get('error', result)}\n", err=True)


@cli.command()
@click.option(
    "--api-url",
    envvar="REVIEW_AGENT_API_BASE_URL",
    default="http://127.0.0.1:18080",
    show_default=True,
    help="Review API base URL (TUI is HTTP-only)",
)
@click.option(
    "--with-server",
    is_flag=True,
    help="Start local FastAPI (uvicorn) in a background thread, then launch TUI",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind address when using --with-server",
)
@click.option(
    "--port",
    default=18080,
    show_default=True,
    type=int,
    help="Port when using --with-server",
)
def tui(api_url, with_server, host, port):
    import os

    if with_server:
        from reviewagent.tui.embedded_server import ensure_local_server

        api_url = ensure_local_server(host=host, port=port)

    os.environ["REVIEW_AGENT_API_BASE_URL"] = api_url.rstrip("/")

    from reviewagent.tui import run_tui

    run_tui(api_base=api_url.rstrip("/"))


@cli.group()
def knowledge():
    """RAG knowledge: ingest Markdown/plain text into the vector index (needs rag.enabled: true)."""


@knowledge.command("ingest")
@click.argument("paths", nargs=-1, type=click.Path(exists=True), required=True)
def knowledge_ingest(paths: tuple[str, ...]) -> None:
    """Chunk .txt / .md under paths and write into Chroma."""
    from reviewagent.rag.store import ingest_paths

    try:
        n = ingest_paths([Path(p) for p in paths])
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Wrote {n} text chunks to the vector store.")


@knowledge.command("ingest-config")
def knowledge_ingest_config() -> None:
    """Scan and index rag.knowledge_dirs from config.yaml."""
    from reviewagent.rag.store import ingest_configured_directories

    try:
        n = ingest_configured_directories()
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Indexed configured directories; ~{n} chunks written this batch.")


@knowledge.command("clear")
@click.confirmation_option(prompt="Delete the entire knowledge vector collection?")
def knowledge_clear() -> None:
    from reviewagent.rag.store import clear_knowledge_index

    clear_knowledge_index()
    click.echo("Knowledge collection cleared (run ingest again to rebuild).")


@cli.command("server")
@click.option(
    "--host",
    default=_DEFAULT_SERVER_HOST,
    show_default=True,
    help="Listen address (Windows default: loopback; use 0.0.0.0 for all IPv4 interfaces / LAN)",
)
@click.option("--port", default=18080, help="Server port")
def server(host, port):
    """Start HTTP API (uvicorn)."""
    from reviewagent.uvicorn_support import prepare_uvicorn_event_loop, win_preflight_tcp_bind

    prepare_uvicorn_event_loop()
    win_preflight_tcp_bind(host, port)

    import uvicorn

    from reviewagent.api.server import create_app

    uvicorn.run(create_app(), host=host, port=port)


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
