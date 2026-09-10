# Episode orchestration design questions, answered

Status: reference, 2026-09-10. Each answer states the decision, the reason, and where the evidence is. The full contracts are in `episode-orchestration-design.md`; the code facts are in `opencode-sandboxed-pairings.md`.

## How is a sandbox represented when it crosses from one server to another

As one small JSON record. It holds the provider's name as written in the config, the pointer the provider returns from `serialize()`, the working directory, who created the sandbox, and what the sandbox can do. It never holds the live `AsyncSandbox` object and never a bare id string.

Reason: today swebench and terminal_bench_2_1 return a bare id, deepswe returns an id plus a full descriptor, swebench_pro returns an id plus a terminal session id, and the OpenCode agent reads only the id. Reconnecting from a bare id loses the working directory and works only for providers that can rebuild a handle from an id. deepswe's descriptor already carries what the record needs.

## Who creates a sandbox, and who destroys it

Whoever calls `start()` owns it and is the only one who calls `stop()`. Every other server connects, works, and disconnects. Creation follows who knows the image:

- The resources server creates the task sandbox when the benchmark supplies the image. This is swebench, deepswe, terminal_bench_2_1, and swebench_pro today, and it stays that way.
- The processor creates a sandbox for the agent when the benchmark has none and the agent still needs isolation, for example a CLI agent on a math benchmark.
- The resources server creates and keeps sandboxes its own tools use, as litmus and ns_tools do, and the agent never sees them.
- The verifier creates a fresh sandbox for grading when the benchmark grades a patch, and destroys it itself.

Reason: every current pairing stops the agent's sandbox twice, once in verify and once in the agent. It only works because OpenSandbox ignores a stop for a sandbox that is already gone.

## What does the agent program receive, and what must it never receive

It receives a working directory, the addresses it may call, and credential files with rollout-scoped access. It never receives the sandbox record, a provider credential, verifier data, or another participant's input.

Reason: a program running inside a sandbox is the untrusted part of the system. If it holds the record it can reconnect as the owner and destroy the box, or reach the provider's control plane. The placement branch on `upstream/ffrujeri/sandboxes` shows the risk: its worker payload is the whole agent config, including provider API keys.

## Do Gym's own Python agents need a sandbox

No. simple_agent and its copies run a Python loop and call tools over HTTP. The dangerous work happens inside environment tools, which run in a sandbox the resources server owns. The agent itself runs in a helper process outside the orchestrator's event loop, so a crash or a blocking call in one agent cannot stall every rollout on that server.

## Where does a CLI agent run

Inside a sandbox, always, in production. The preferred place is the task sandbox the benchmark created, because the task files are already there and the verifier grades that box. If the benchmark has no sandbox, the processor creates one from the agent's provider and image. A missing sandbox fails before any model call. There is no fallback to running the CLI on the host.

Reason: OpenCode, Claude Code, Codex, Pi, and OpenClaw run third-party programs that read files, spawn processes, and reach the network. Today `opencode_sandboxed_agent` already runs OpenCode inside the benchmark's sandbox, so the placement is proven; only the ownership around it is wrong.

## Agent as an HTTP server versus a Python class: what changes

Nothing outside an agent calls its `/v1/responses` today. The only callers are the agent calling itself and `remote_agent`, which calls a user's own service. So the agent's server is a private hop, and the question is what the process boundary buys.

Keeping the server gives each agent its own Python environment, crash isolation, and the existing start-up and health machinery, at the cost of a network hop, a queue, and a timeout layer per rollout, plus routes that exist only so a rollout id survives the agent calling itself. It also cannot reach into a sandbox, which is why every sandboxed agent became a second directory.

Making the agent a Python class lets the processor decide where it runs. For a CLI agent, the sandbox is the process boundary and a stronger one. The processor writes a request file into the sandbox, starts the agent program there as a normal user, and reads a result file back. There is no HTTP server inside the sandbox. The agent still calls the model server directly, so token capture is unchanged. For a Python agent, the helper process replaces the server as the isolation boundary. A user's own HTTP service stays supported through one executor that calls its `/v1/responses` and keeps seeding and grading in the processor.

## Where does the resources server keep the sandbox between seed and verify

In a table keyed by rollout id and attempt, not by session cookie. A resources server that creates sandboxes either runs with one worker, or stores its own reconnect descriptor so any worker can find the sandbox again. The base class owns this table; a server that creates a sandbox calls one helper in seed and returns the record.

Reason: all four servers keep a dictionary in one process keyed by the session cookie. None of them sets `num_workers`, and that is the only reason the dictionary works. swebench and terminal_bench_2_1 crash with a key error if verify lands on a different worker; deepswe returns reward zero; swebench_pro reports an extraction error.

## What happens when the same rollout is seeded twice

Seed for an existing rollout and attempt stops or reuses the previous sandbox before creating a new one. deepswe and swebench_pro already do this. swebench and terminal_bench_2_1 overwrite their entry and leak the old sandbox until its time limit.

## How does the verifier get the agent's work

The resources server extracts it, before the owner destroys the sandbox, in one of two shapes that stay exactly as they are per benchmark:

- Copy a patch out and grade it in a fresh sandbox: swebench, deepswe, swebench_pro. Each keeps its own extraction rule. swebench must add `git add -N` so new files count and must handle binary changes; deepswe requires committed work; swebench_pro filters files that were untracked in the pristine image.
- Grade the live sandbox: terminal_bench_2_1 uploads the tests into the agent's sandbox and runs them there. A fresh sandbox would discard the state being graded.

Whatever is graded leaves the sandbox as a bounded payload or a durable reference, never as a path, because the sandbox may be gone when the caller reads the result.

## Who installs the CLI, and where does the binary come from

The harness owns the command that runs the tool. The executor owns getting the tool into the sandbox: from the image when it is prebuilt, from a digest-verified bundle when it must vary independently of the image, or from a downloaded installer in development only. Production rollouts do not download OpenCode per task. Today the agent's one shell command does the install and the run together; splitting them is what makes a prebuilt image possible.

## What happens when the harness fails for reasons that are not the model's fault

The result carries a failure class, and the rollout goes to the failure sidecar instead of counting as reward zero. An install failure, a model server the sandbox could not reach, or a lost transcript are infrastructure failures. Today none of these set a class; they come back as a well-formed rollout with reward zero, and the only sign is a diagnostic field nothing reads.

## Is a sandbox server needed

Not for any benchmark in the repository. OpenSandbox and E2B can rebuild a handle in another process from the descriptor. Docker, Apptainer, and the local provider can be given same-host reconnect by id, name, or path, which is a small provider change. The sandbox server from PR 2085 is the adapter for sharing a host-local provider across hosts, which nothing in the repository does today. Configuration selects it explicitly; it is never inserted in front of a provider that can reconnect on its own.

## How does routing work, and what do NeMo RL and verl still see

Source datasets carry no agent. Run configuration selects the processor deployment, the resources server, the participants and their harnesses, and the models. The collector materializes that into the request. For as long as trainers read `agent_ref.name`, the collector derives it from the execution name before dispatch and the processor echoes it on the result. NeMo RL main reads the resolved `agent_ref` after `run_examples` and keys token capture by `_ng_rollout_id`; the verl recipe raises if a dataset row lacks `agent_ref`. Both keep working, and upstream's collate change that strips `agent_ref` from rows is the one break that needs a repair independent of this design.

## Is the processor base class an interface

Yes. `EpisodeProcessor` is a protocol with one `process` method and no inherited body. `StandardEpisodeProcessor` is the concrete lifecycle, and a benchmark that owns its whole interaction implements the protocol directly. The public RFC gives the base class a concrete `run()` with no hooks; this design does not.

## Does the design support more than one agent in an episode

Yes, from the first request. Every request carries participants, each with a role, a harness, and its own model bindings, plus a primary participant and a schedule. A single-agent episode is one participant. A policy plus a simulated user is two. The primary participant's trajectory is what NeMo RL trains on; the others keep separate capture identifiers so their tokens stay out of the policy update.

## What can start now

Making new datasets agent-agnostic, extracting harness behavior behind the Python contract, moving the shared `/run` lifecycle into the standard processor inside existing deployments, adding the supervised helper process, extending seed with the optional workspace record, and giving `AsyncSandbox` a borrowed connection that cannot stop the sandbox. None of these change routing, and the first real target is OpenCode with swebench, then deepswe, swebench_pro, and terminal_bench_2_1.
