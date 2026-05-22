# Round 0 Contract

## Mainline Objective
Complete ALL plan tasks (task1-14): fix S2T, get all 4 modes producing correct output inline, generate precision table matching investigation doc, verify BF16 regression, clean up.

## Target ACs
All ACs (1-9). This is Round 0 — the full plan is the objective.

## Blocking Side Issues In Scope
- S2T MLIR legalization + NaN (task1-2)
- cutlass-dsl must be 4.4.2 (verified)

## Queued Side Issues Out of Scope
- Rebase onto latest public/main (post-PR cleanup)
- 59 WIP commit squashing (post-PR cleanup)

## Round Success Criteria
1. All 4 modes produce correct output (cos >= 0.99) on the inline kernel
2. TFLOPS within 3% of investigation table for all modes
3. BF16 path has zero regression
4. Full precision table generated and matches investigation doc
5. No separate flash_fwd_sm100_fp4.py import in interface.py
