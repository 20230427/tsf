# Statistical audit bundle

Regenerate all artifacts from the preserved result JSON files with:

```text
python scripts/generate_statistical_audit.py
```

`complete_statistical_audit.json` is the source of truth. The Markdown and
LaTeX appendices are rendered only after that JSON is serialized and read
back; they do not recompute statistics. The LaTeX fragment requires
`booktabs` and `longtable`.

The audit reports paired seed-level or run-level observations, per-horizon
and horizon-averaged effects, 95% Student-t confidence intervals, t/df/raw
p-values, BH q-values, exact sign-flip sensitivity p-values, relative effects,
and paired standardized effects (`dz`). Family IDs, sizes, and complete member
lists are recorded in the JSON and appendices.

The existing result files are legacy artifacts: they preserve run order and a
run count but not actual seed identifiers. Consequently every such observation
has `seed_id: null` and an anonymous `pair_index`. Pair indices must never be
reported as seed IDs. Future result files that record `seed` or `seed_id` on
every run are matched by that explicit identifier instead of position.

For the current five-run unified comparisons, a two-sided exact sign-flip test
has a minimum attainable p-value of 0.0625. For three-run component tests the
minimum is 0.25. These exact results are sensitivity checks and make the
small-sample limitation visible alongside the manuscript's paired t tests.
