def run_cli():
    from .app import run_cli as _run_cli
    return _run_cli()


__all__ = ['run_cli']
