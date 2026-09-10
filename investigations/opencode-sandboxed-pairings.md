# How the sandboxed OpenCode agent works with each SWE-style resources server

Status: analysis, 2026-09-10, against upstream main at `9506eb87c`.

Four resources servers allow `opencode_sandboxed_agent`: `swebench`, `deepswe`, `terminal_bench_2_1`, and `swebench_pro`. `vibench_agent` subclasses it, and `terminus_2_sandboxed_agent` copies its handoff. This note describes what the agent does, what each server does, where the pairings hold together by luck, and what that means for the sandbox contract in the design.

## What the agent does

The agent has one job per rollout: get into a sandbox that already contains the task, run the OpenCode program there, and hand the result to the verifier.

1. It posts the whole task row to `/seed_session` and keeps the cookies the server sets, so the server can find the same session at verify time.
2. It reads exactly one field from the seed response, `sandbox_handle`, and treats it as a sandbox id. It ignores every other field, including deepswe's full descriptor and swebench_pro's terminal session id.
3. It reconnects to that sandbox by id through the provider named in its own config. Reconnecting by id only works for OpenSandbox and E2B, because only those providers can rebuild a handle in another process. Reconnecting by id also loses the working directory the server set, so OpenCode starts wherever the image's default directory is.
4. If the seed response has no `sandbox_handle`, the agent quietly starts its own sandbox from a hardcoded SWE-bench image (the first astropy instance) and runs the task against an unrelated repository. This path never runs today because all four servers return a handle, but it turns a missing field into a silent wrong answer.
5. Inside the sandbox it runs one shell command: install OpenCode (from a staged script or by downloading the installer), write the OpenCode config into an environment variable with the model server address, and run `opencode run` with the task text. That single command gets the agent's three-hour timeout. The three follow-up commands that export the transcript get the default 180-second timeout, so a slow export can lose the transcript of a run that succeeded.
6. It copies the transcript out of the sandbox to a results directory under its own source tree, turns it into a Responses API output, and optionally snapshots OpenCode's SQLite database for observability.
7. It posts the row plus the response to `/verify`, then stops the sandbox after verify returns, then adds a seven-kilobyte hardcoded OpenCode system prompt to the front of the returned conversation.

The sandbox stop is outside the error handling. If the OpenCode run raises, the agent never calls verify and never stops the sandbox. Nothing in this stack sets a failure class, so an install failure, an unreachable model server, or a missing transcript all come back as a normal rollout with reward zero.

## What each resources server does

| | swebench | deepswe | terminal_bench_2_1 | swebench_pro |
| --- | --- | --- | --- | --- |
| Seed creates the sandbox from | the SWE-bench instance image, default directory | the task's pinned image, directory `/app` | the task's Docker image, default directory | the task image by digest, directory `/app` |
| Seed returns | `sandbox_handle`, a bare id | `sandbox_handle` and `sandbox_descriptor` from `serialize()` | `sandbox_handle`, a bare id | `sandbox_handle` and `pty_session_id` |
| Seed on a retried session | overwrites its record and leaks the first sandbox | stops the previous sandbox first | overwrites and leaks | stops the previous sandbox first |
| Where the server keeps the sandbox | a dictionary in the process, keyed by session cookie | same | same | same, plus a shutdown sweep at process exit |
| How verify finds the agent's work | runs `git diff` in the agent's sandbox, in a directory it learns by running `pwd` in a second fresh sandbox of the same image | runs the task's collect hook, which diffs committed work only | uploads the tests into the agent's sandbox and runs them there | runs `git add -N . && git diff` in `/app`, then removes files that were untracked in the pristine image |
| Where the tests run | a fresh sandbox with the patch applied | a fresh sandbox with the patch applied | the agent's own sandbox | a fresh sandbox per attempt, up to a retry budget |
| Who stops the agent's sandbox | verify, then the agent again | verify, then the agent again | verify, then the agent again | verify, then the agent again |
| Missing session at verify | crashes with a key error | returns reward zero with a message | crashes with a key error | caught and reported as an extraction error |
| Untracked new files count | no | only if committed | not applicable | yes |
| Reverification mode | unknown | unsupported | declared stateless, which is wrong: verify needs the live sandbox | unknown |

None of the four servers set `num_workers`, so each runs as one process. That is the only reason the per-process dictionaries work. Raising `num_workers` on any of them would let seed land on one worker and verify on another, and the failure would be a key error at verify time, not a startup error.

## Where the pairings hold together by luck

- The handoff is one field wide. deepswe already sends a full descriptor with the working directory, and swebench_pro sends a terminal session id, and the agent reads neither. The comment in the agent still says the field comes from swebench specifically.
- The working directory is lost on reconnect. deepswe and swebench_pro set `/app` on their spec. The agent lands in `/app` only because those images happen to default to it.
- swebench learns the working directory by running `pwd` in a different container. It works because both containers come from the same image.
- swebench's diff has no `git add -N .`, so a new file the model created is dropped from the patch. swebench_pro does it correctly. The agent's own code comments assume every server uses the `add -N` form.
- Every pairing stops the agent's sandbox twice. The second stop succeeds only because OpenSandbox ignores a stop for a sandbox that is already gone.
- Sandbox creation code is copied six times: four servers, the agent, and vibench. Each copy merges provider metadata a little differently, truncates the instance id or not, and applies the CPU thread caps or not.
- terminal_bench_2_1 tells reverification it is stateless. Reverification would pass its guard and then crash looking for a live sandbox.

## vibench and terminus_2 do it differently

`vibench_agent` inherits the OpenCode driving code but reverses ownership: the resources server returns no sandbox, the agent creates its own from a build image with the task files pre-staged, packs the finished application into a tarball, stops the sandbox before verify, and passes the tarball path. Its docstring says why: only OpenSandbox can reconnect by id, so a server-created sandbox cannot work on Docker or Apptainer. It also rewrites the model address to `host.docker.internal` for Docker, which the base agent does not do.

`terminus_2_sandboxed_agent` uses the same `sandbox_handle` field but indexes it directly, so a missing field fails loudly instead of starting the astropy image. It drives the sandbox from the agent process through tmux and calls the model from the agent process, so nothing inside the sandbox talks to the model server.

## What this means for the sandbox contract

The four pairings are the same pattern with four hand-written variants. The design turns the pattern into a contract and removes the variants.

- One record replaces `sandbox_handle`, `sandbox_descriptor`, and `pty_session_id`. It carries the provider name, the serialized pointer from `serialize()`, the working directory, who created the sandbox, and what it can do. deepswe already produces almost exactly this. The agent connects from the record and starts in the right directory.
- The base resources server owns the session table. A server that creates a sandbox calls one helper in seed and returns the record. The table is keyed by rollout id and attempt, not by cookie, and a retried seed stops the previous sandbox before creating a new one, which deepswe and swebench_pro already do and swebench and terminal_bench_2_1 do not.
- Whoever creates the sandbox stops it. The agent never stops a sandbox it did not create. That removes the double stop and the dependence on the provider ignoring a second kill.
- Verify runs before the creator stops the sandbox, and the server copies out what it grades first. The two grading shapes stay as they are: swebench, deepswe, and swebench_pro copy a patch out and test it in a fresh sandbox; terminal_bench_2_1 tests the live sandbox. Patch extraction uses one shared implementation with `add -N` and a size limit, so new files count everywhere.
- The agent runs inside the task sandbox as a placement decision, not as a separate agent directory. The OpenCode install, config, and run command are the harness's job; the reconnect, working directory, model address, and cleanup are the framework's.
- A missing sandbox in the seed response is an error before any model call, never a fallback to a hardcoded image.
- Failures get a class. An install failure or a missing transcript is reported as an infrastructure failure, so it lands in the failure sidecar instead of counting as a wrong answer.
- Reconnecting by id stays OpenSandbox-only until Docker, Apptainer, and local providers can rebuild a handle on the same host, which is a small provider change and does not need a sandbox server for any of these four benchmarks.
