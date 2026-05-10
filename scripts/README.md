# scripts/

This directory holds **operational** entry points (one-shot tools, ad-hoc shell
helpers). Library-level CLIs live in `daity/scripts/` and are wired through
`pyproject.toml` console scripts.

For Phase 0 the entry point is:

```bash
make audit
# == python -m daity.scripts.phase0_audit
```
