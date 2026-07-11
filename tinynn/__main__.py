"""Entry point for ``python -m tinynn ...``; delegates to :func:`tinynn.cli.main`."""

from __future__ import annotations

from .cli import main

raise SystemExit(main())
