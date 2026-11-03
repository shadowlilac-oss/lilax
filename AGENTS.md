# Agent Instructions

## Strict Single Responsibility

- Apply single responsibility throughout the repository: directories, modules, classes, and functions must each have one coherent responsibility and one reason to change.
- Existing god directories, files, classes, and functions are defects to refactor, not patterns to copy or exceptions to preserve.
- Separate unrelated configuration domains, validation, runtime initialization, persistence, orchestration, and reporting into their responsible units. Do not collect them in a central schema, utility, manager, or coordinator implementation.
- Split by domain and behavior, not arbitrary line counts, numbered file fragments, or generic dumping grounds. Moving a god implementation into another file or hiding it behind a facade does not resolve the violation.
- Large configuration models must compose focused domain models and validators; a single class must not retain every domain's implementation after its file is split.
- Preserve behavior, defaults, validation order, public imports, external type identities, and caller contracts during structural refactors. Read the original implementations and update actual callers; do not invent replacement behavior or dependency shims.
- Publish disjoint ownership and agree cross-module interfaces before parallel edits. Refactoring does not authorize tests, builds, installs, execution, or unrelated behavioral changes.

## GPU Workload Coordination

- Execution is authorized, subject to environment restrictions and any narrower task instructions.
- Coordinate all long-running GPU workloads through one designated executor. At most one training run or other long-running GPU workload may run at a time across all agents.
- Confirm the preceding GPU workload has finished before starting another. CPU-only work may proceed independently; do not bypass coordination by launching GPU work in another process, container, or agent.
