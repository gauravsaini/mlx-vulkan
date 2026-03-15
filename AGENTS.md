# Agent Instructions

- **Goal**: Implement the `Compile` primitive for the Vulkan backend to enable fused graph execution, avoiding CPU fallbacks and eager execution overhead during LLM inference.
- **Documentation First**: Always keep `PLAN.md` and `TIMELINE.md` updated with progress.
- **Frequent Commits**: Commit changes frequently to track progress and provide rollback points.
- **Bold Action**: Do not shy away from massive architectural undertakings like `mlx/backend/metal/compiled.cpp`. "Nothing is massive." Just start working on it.
