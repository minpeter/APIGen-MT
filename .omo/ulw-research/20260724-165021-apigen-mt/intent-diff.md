# Intent vs. Reality

| intent_id | expected truth | observed reality | diff | violated invariant | intent source | supporting observations | status | claim ids |
|---|---|---|---|---|---|---|---|---|
| I001 | APIGen-MT cleanup preserves every shipped public behavior. | Pending baseline and post-change comparison. | Unknown. | Behavior-preserving cleanup. | User request; remove-ai-slops contract. | O001 | unknown | C001 |
| I002 | The upgrade implements the strongest relevant method supported by the original paper and later evidence. | Eight-lane synthesis selected deterministic replay/outcome integrity and rejected unsupported broad replication. | Decision complete; implementation pending. | Evidence-backed best implementation. | User request; `README.md`. | O002, O004-O010 | partial | C002, C003, C007 |
| I003 | Generated trajectories are validated by outcomes and environment state, not only tool-call names/order. | Five behavior-level REDs prove current final acceptance ignores tampered output/state and multi-turn verification. | Deterministic acceptance must flip RED to GREEN. | Stateful semantic validity. | APIGen-MT paper premise; `README.md`. | O002, O009, O010 | violated | C004, C006, C007 |
| I004 | The CLI has a reproducible local path that proves generation/validation without paid external writes. | Help works without keys; generation currently appears API-dependent. | A deterministic fixture-backed surface may be required. | Real-surface observability. | User verification requirement. | O003 | unknown | C005 |

## Expected truths

- Research must converge across the original paper, official implementation, citation graph, and adjacent/follow-up methods.
- Cleanup must follow delete/reuse/stdlib/simplify ordering and retain boundary error handling unless adversarial proof supports narrowing.
- Every behavior delta must have a faithful failing-first proof and real-surface verification.
