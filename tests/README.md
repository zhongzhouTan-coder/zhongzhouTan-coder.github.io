# Test Organization

Tests mirror the domain folders under `scripts/`:

- `checks/` covers repository validators.
- `wiki/` covers search and link-graph behavior.
- `ingestion/` covers source conversion workflows.
- `repositories/` covers remote parsing and pinned checkout management.
- `fixtures/` is reserved for reusable, non-generated test data.

Run the suite with workspace-managed dependencies:

```bash
./scripts/run-in-workspace.sh npm test
```
