# APIGen-MT — Official-Release Brief

## 1. Authoritative repos & datasets (SHA-pinned)

| Artifact | Location | Pinned SHA |
|---|---|---|
| Official org repo (xLAM) | `github.com/SalesforceAIResearch/xLAM` | `a88aa3aeddbc2d7d6aa7a87687d1a085c34e2aec` |
| Official dataset | `huggingface.co/datasets/Salesforce/APIGen-MT-5k` | `abc4a517d67c541f85f6470cbd8fd3186b36830e` |
| Project site (paper/models/dataset/blog) | `github.com/apigen-mt/apigen-mt.github.io` | `main` (not SHA-pinned) |
| Paper | arXiv:2504.03601 | v2 |

Permalinks:
- https://github.com/SalesforceAIResearch/xLAM/blob/a88aa3aeddbc2d7d6aa7a87687d1a085c34e2aec/README.md
- https://huggingface.co/datasets/Salesforce/APIGen-MT-5k/blob/abc4a517d67c541f85f6470cbd8fd3186b36830e/apigen-mt_5k.json

## 2. What IS and IS NOT released

**Released:**
- **Dataset** `APIGen-MT-5k`: 5000 SFT-ready multi-turn conversations, CC-BY-NC-4.0, gated. Single file `apigen-mt_5k.json`.
- **Models** (xLAM family) and **paper + project website** describing the method.

**NOT released (verified):**
- **No pipeline code.** The xLAM tree at the pinned SHA contains no `apigen` directory — only `actionstudio/` (training) and `criticLAM`. The project site offers Paper/Models/Dataset/Blog buttons and **no Code button**; the dataset card's `Code:` field points to the *website* repo, confirming no code repo exists. The README states data is "**partially released due to internal regulations**."
- **No generation internals in the dataset:** no blueprint `(q, a_gt, o_gt)`, no ground-truth action sequences, no environment/API state, no sampler outputs, no validator verdicts. Only final conversations ship.

**Consequence:** the paper's Phase-1 machinery (samplers, review committee, three-stage validation, Reverse Task Recombination) is reconstructable only from paper text + released outputs, not inspectable code.

## 3. Schema fields relevant to deterministic verification

From HF datasets-server `/info` + `/first-rows`, each record is exactly:

```
conversations: [{from: string, value: string}]   # role vocab: human | gpt | function_call | observation
tools:        string                              # JSON-serialized tool schemas
system:       string                              # full domain policy (τ-bench Airline Agent Policy)
```

Verification-relevant properties of the released data:
- **`system` is a natural-language policy** (current-time stamp; book/modify/cancel/refund rules; "one tool call at a time"; "obtain explicit user confirmation (yes)"; "transfer iff out of scope"). This is the reference object for **policy-compliance** validation — the stage the local PoC lacks.
- **Tool set is τ-bench airline:** `book_reservation`, `cancel_reservation`, `get_reservation_details`, `get_user_details`, `search_direct_flight`, `search_onestop_flight`, `update_reservation_*`, `send_certificate`, `calculate`, `list_all_airports`, **`think`** (explicit reasoning tool), `transfer_to_human_agents`.
- **Role grammar enables deterministic checks:** `function_call`/`observation` pairs let you replay calls against an executable env; `human` turns show **incremental information reveal** and **adversarial/incorrect claims** (e.g., a user wrongly asserting gold membership, corrected via `get_user_details`); `gpt` turns show clarifying questions, pre-mutation confirmations, policy denials, and human transfers.
- **No `a_gt`/`o_gt`/state fields** — output-based equivalence checking (the paper's Stage-2 alignment + rejection-sampling acceptance) is **not reproducible from the release alone**; only the conversation surface is.

## 4. Five most important local divergences (absolute file:line)

The local repo `minpeter/APIGen-MT` is a **blueprint-forward golden-trajectory synthesizer over stateful BFCL tools**: it implements the paper's Phase-1 "generate → validate → feedback" skeleton, but replaces the defining Phase-2 agent–human interplay with scripted forward generation.

**D1 — No simulated human; user turns are pre-scripted (kills Phase 2).**
`/home/minpeter/github.com/minpeter/APIGen-MT/src/apigen_multi_turn.py:1176` (`_generate_turn_query`), docstring at `:1182` "Use the blueprint's pre-written user query for this turn", returns the blueprint query verbatim at `:1210`. The paper's simulated human that *incrementally reveals* info/sub-goals does not exist; user turns are authored up front in the Stage-0 blueprint.

**D2 — No autonomous test agent; "agent" emits the blueprint's `expected_tools`.**
`/home/minpeter/github.com/minpeter/APIGen-MT/src/apigen_step_by_step.py:518` (`_generate_next_step`), constrained at `:532` (`tools_remaining = [t for t in expected_tools if t not in tools_used]`) and prompted at `:577` (`"tool_name": "exact_name_from_expected_tools_list"`). The same generator model fills in the planned tools — there is no separate policy model interacting with a user, and no rejection sampling over agent trajectories (grep for `committee|majority|best.of.n|self.critique` → none; the only `rejection` hits at `apigen_multi_turn.py:1266,1334` are datapoint-retry, not rejection sampling).

**D3 — Environment is mutated to fit the task (inverts the verification direction).**
`/home/minpeter/github.com/minpeter/APIGen-MT/src/apigen_step_by_step.py:975` (`_stage1_5_adjust_initial_state`), invoked at `:809-810`. The LLM edits live API state (plus hard-coded fixups) so planned tools succeed. The paper generates a task solvable in a **fixed** environment and verifies the trajectory against it; here the environment is coerced to the task, so "success" is partly self-fulfilling.

**D4 — Wrong domain + no policy-compliance stage.**
`/home/minpeter/github.com/minpeter/APIGen-MT/src/generate_step_by_step.py:132-134` (default tool pool `bfcl_v3_tools_with_outputs.jsonl`; tools under `tools/` are Gorilla/BFCL: file/math/posting/vehicle/travel/message/ticket/trading). The official data is **τ-bench airline with a rich `system` policy**. README confirms the gap: `/home/minpeter/github.com/minpeter/APIGen-MT/README.md:97` ("Policy compliance checks are not implemented"). Deterministic validators exist (arg/entity/cross-turn consistency) but there is no check against a domain policy document.

**D5 — Reverse Task Recombination absent.**
`/home/minpeter/github.com/minpeter/APIGen-MT/README.md:100` ("Does not implement Reverse Task Recombination") and `:114` (TODO). Grep for `recombin|building block|compositional` across `src/` → none. Complex tasks are generated whole in Stage 0, not composed from independently validated simpler blocks as the paper specifies.

*(Supporting anchors: stateful env is real — `/home/minpeter/github.com/minpeter/APIGen-MT/src/tool_manager.py:764` `initialize_api_state`, `:862` `invoke_python_tool`; LLM "virtual executor" at `:1233` is disabled in favor of Python tools. Single judge client throughout — no committee.)*

## 5. Verification notes
Remote facts cross-checked across ≥2 sources (repo tree + README + site HTML + HF `/info` & `/first-rows`). Local claims verified by reading the modules + targeted greps; line anchors from LSP/grep. The user's repo was not modified.

## Remaining uncertainty
- Exact arXiv appendix prompt wording not fetched; paper-prompt comparisons are against the authors' project-page method summary (labeled intent). The released `system` policy text *is* verbatim primary evidence.
- `apigen-mt.github.io` cited at `main`, not a pinned SHA.

## EXPAND leads
- Fetch arXiv:2504.03601v2 HTML appendix to capture verbatim Phase-1/Phase-2 prompts and exact Stage-1/2/3 validator definitions, then re-score D2/D4 against primary prompt text.
- Statistically profile all 5000 released rows (role n-grams, `think`/`transfer_to_human_agents` frequency, avg turns/task, policy-violation→transfer rate) via datasets-server pagination to quantify the Phase-2 behaviors D1/D2 lack.
- Pin `apigen-mt.github.io` HEAD SHA and extract `img/pipeline.png`/`tau_pipeline.png` box labels to recover the exact sampler/validator stage diagram absent from text.
- Confirm single-turn `APIGen` (arXiv:2406.18518) code-release status and compare its `xlam-function-calling-60k` validator fields against APIGen-MT to trace what Salesforce actually open-sourced of the original pipeline.
