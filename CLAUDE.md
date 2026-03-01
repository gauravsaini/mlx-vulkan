# MLX Vulkan Backend — Claude Project Instructions

## TIMELINE.md Protocol (MANDATORY)

After every major commit to this repository, Claude MUST update `TIMELINE.md` at the project root.

**What counts as a major commit:**
- Any new GPU primitive implementation (new `eval_gpu` body)
- Any shader change (`*.comp`) that fixes a bug or adds functionality
- Any build system or architecture change
- Bug fixes to existing GPU dispatch
- Any commit that changes test pass/fail counts

**Format to follow** (append a new entry — do NOT overwrite existing ones):

```markdown
## UPDATED ON : YYYY-MM-DD

### <commit-type> (<date>) (<time>) — <short headline>

1. **<Topic>**:
   - <bullet 1>
   - <bullet 2>
   - <bullet 3>

2. **Tests** (before → after):
   - Stage NN: X/Y → A/B
   - ...

3. **Files changed**: list the key files
```

**When to update:**
- Immediately after `git commit` (before or after `git push`)
- One entry per logical batch of related commits — not per file changed
- Keep each entry ≤ 30 lines

---

## Project Context

- **Repo**: `https://github.com/gauravsaini/mlx-vulkan`
- **Working dir**: `/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/`
- **Build**: `cmake --build build_vulkan -j4` from `mlx-src/`
- **Post-build .so copy**: `cp build_vulkan/core.cpython-314-darwin.so python/mlx/core.cpython-314-darwin.so`
- **Plan doc**: `PLAN.md` — update Phase 8 checkboxes and Known Issues after each session
- **Memory**: `~/.claude/projects/-Users-ektasaini-Desktop-mlx-vulkan/memory/MEMORY.md`

## Pipeline Cache Rule

**CRITICAL**: Bump `kPipelineCacheVersion` in `mlx-src/mlx/backend/vulkan/device.cpp` whenever any shader push constant layout changes. Stale cache causes MoltenVK to SIGKILL at import time.

Current version: **2** (filename: `~/.cache/mlx_vulkan_pipeline_cache_v2.bin`)

## Indexing Push Constant Rule

ALL indexing ops (Gather, GatherAxis, ScatterAxis) MUST use `kIndexPushSize = 44` (11-field `IndexPushConst`). Never call `get_pipeline("indexing", ...)` with a different size — the pipeline cache key is name-only.

## Test Commands

```bash
# Run a specific stage
PYTHONPATH=python timeout 30 python3 ../tests/vulkan/test_stage13_indexing.py

# Run all stages summary
for f in ../tests/vulkan/test_stage*.py; do
  name=$(basename $f)
  result=$(PYTHONPATH=python timeout 30 python3 $f 2>&1 | grep "Results:")
  echo "$name: $result"
done
```
