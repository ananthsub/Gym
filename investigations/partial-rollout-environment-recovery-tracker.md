# Partial rollout checkpointing environment recovery tracker

| Environment / variant | Environment family | Interaction pattern | State model | Recovery requirement | Suggested next level |
| --- | --- | --- | --- | --- | --- |
| `mcqa` | Pure answer verifiers | Single-turn | Stateless | Token prefix, completed model response, and verifier result | L2 — token-prefix recovery |
| `calendar` | Pure answer verifiers | Single-turn | Stateless | Token prefix, completed model response, and verifier result | L2 — token-prefix recovery |
| `reasoning_gym` | Pure answer verifiers | Single-turn | Stateless | Token prefix, completed model response, and verifier result | L2 — token-prefix recovery |
| `equivalence_rule` | Pure answer verifiers | Single-turn | Stateless | Token prefix, completed model response, and verifier result | L2 — token-prefix recovery |
| `ether0` | Pure answer verifiers | Single-turn | Stateless | Token prefix, completed model response, and verifier result | L2 — token-prefix recovery |
| `structured_outputs` | Pure answer verifiers | Single-turn | Stateless | Token prefix, completed model response, and verifier result | L2 — token-prefix recovery |
| `format_verification` | Pure answer verifiers | Single-turn | Stateless | Token prefix, completed model response, and verifier result | L2 — token-prefix recovery |
| `math_with_judge` | Math and code verifiers | Single-turn | Stateless after response | Token prefix, response, and verification result | L2 — token-prefix recovery |
| `code_gen` | Math and code verifiers | Single-turn | Stateless after response | Token prefix, response, and verification result | L2 — token-prefix recovery |
| `nvarc` | Math and code verifiers | Single-turn | Stateless after response | Token prefix, response, and verification result | L2 — token-prefix recovery |
| `equivalence_llm_judge` | LLM judges | Single-turn + judge | Stateless, external result | Token prefix plus durable judge request/result receipt | L2 — token-prefix recovery |
| `abstention` | LLM judges | Single-turn + judge | Stateless, external result | Token prefix plus durable judge request/result receipt | L2 — token-prefix recovery |
| `inverse_if` | LLM judges | Single-turn + judge | Stateless, external result | Token prefix plus durable judge request/result receipt | L2 — token-prefix recovery |
| `multichallenge` | LLM judges | Single-turn + judge | Stateless, external result | Token prefix plus durable judge request/result receipt | L2 — token-prefix recovery |
| `jailbreak_detection` | LLM judges | Single-turn + judge | Stateless, external result | Token prefix plus durable judge request/result receipt | L2 — token-prefix recovery |
| `genrm_compare` | Group-level judges | Sibling-coupled | Stateless, group-coupled | Completed sibling IDs, group membership, comparison phase, and GenRM result | L1 — completed-sibling recovery |
| `single-step tool comparison` | Tool-action verifiers | Single-step tool proposal | Stateless | Model response/tool call and verifier result | L2 — token-prefix recovery |
| `swe_pivot` | Tool-action verifiers | Single-step tool proposal | Stateless | Model response/tool call and verifier result | L2 — token-prefix recovery |
| `terminal_multi_harness` | Tool-action verifiers | Single-step tool proposal | Stateless | Model response/tool call and verifier result | L2 — token-prefix recovery |
| `workplace_assistant` | Logical in-memory tools | Multi-turn | Logical in-memory state | Replay committed actions first; add a versioned logical snapshot if needed | L4 replay → L5 selected Gym snapshot |
| `indirect_prompt_injection` | Logical JSON tools | Multi-turn | Logical JSON state | Replay committed actions first; add a versioned logical snapshot if needed | L4 replay → L5 selected Gym snapshot |
| `math_formal_lean` | Turn-based stateless verifier | Multi-turn | Agent state only | Persist previous attempts, compiler feedback, correction prompt, and current turn | L4 — replay-based environment recovery |
| `Litmus tool-using rows` | Replayable Python sandbox | Multi-turn | Replayable sandbox history | Recreate a sandbox and replay successful Python cells to the last durable turn | L4 — replay-based environment recovery |
| `ns_tools` | Stateful Python sandbox | Multi-turn | Stateful sandbox | Replay tool history initially; add sandbox snapshot support if replay is insufficient | L4 replay → L6 selected sandbox recovery |
| `swe_agents (swe_teacher.yaml)` | Full SWE workspace | Multi-turn | Filesystem + agent state | Restore transcript, control-flow phase, tool results, and workspace filesystem | L6 — selected sandbox recovery |
