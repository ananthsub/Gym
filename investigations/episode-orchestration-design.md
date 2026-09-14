# Episode Architecture for NeMo Gym

## Problem summary

Today, rollout collection sends `POST /run` to an agent server. That method commonly seeds resources-server state, runs the agent, asks the resources server to verify the result, and performs cleanup.

This creates several problems:

- Episode orchestration is embedded in each agent server's `/run`: resources-session setup, agent invocation, verification, and cleanup.
- Gym has no framework-owned intervention point for this sequence, so changing the episode protocol requires changing agent-server code.
- Shared behavior such as sandbox setup, resources access, deadlines, cancellation, cookies, and cleanup is implemented repeatedly and inconsistently across agent servers.
- Agent and resources-server code can both believe they own the same sandbox.
- A bare sandbox ID does not describe how another server connects, where commands run, or which operations are allowed.

The proposed architecture moves episode-level ordering into an episode processor. A processor may compose Gym servers or run a self-contained external framework whose components do not fit Gym's agent-server and resources-server boundaries.

An agent's `/v1/responses` implementation remains responsible for agent behavior, including calls to resources-server tools. The episode processor owns the surrounding episode sequence.

## Scope

### Requirements

- **Episode processor**
  - Exposes `/run` and owns the episode protocol and final result.
  - Composes agent and resources servers when those boundaries fit, or implements a self-contained external protocol.
  - Applies deadlines and cancellation and coordinates cleanup on every handled exit path.
  - Asks participating servers to clean up the state they own. A self-contained processor cleans up state created within its own protocol.
  - Preserves existing Gym dataset, evaluation, and NeMo RL interfaces during migration.
  - Uses Gym's configuration, startup, addressing, readiness, and telemetry mechanisms.
- **Agent server**
  - Owns agent behavior, dependencies, sessions, and objects it creates.
  - Continues to expose `POST /v1/responses`.
  - Can operate a resources-server-owned task sandbox without authority to destroy it.
  - Can provision and clean up its own sandbox when no task sandbox is supplied.
- **Resources server**
  - Owns task setup, stateful tools, verification, and benchmark-specific cleanup.
  - May create a task sandbox and grant the agent restricted access.
  - Validates its benchmark-specific verification result and returns it through the common episode contract.
  - Retains ownership of its state and task sandbox through verification and cleanup.

### Episode, rollout, task, and session

- A **task** is dataset content identified by `TaskIdentity`.
- A **rollout** is one logical sample produced for that task.
- An **attempt** is one physical execution of a rollout. A retry increments the attempt.
- An **episode** is the complete lifecycle for one attempt, including all interaction turns, verification, and cleanup.
- A **resources session** is resources-server state created when a processor uses a resources server.
- An **agent session** is agent-server-local state created when a processor delegates behavior to an agent server.

`POST /v1/responses` is an agent invocation, not the definition of an interaction turn. A processor may use one or more agent invocations during a turn, and each invocation may contain multiple model and tool calls.

An exported rollout is a training or evaluation projection of an episode. The initial single-agent protocol produces one primary rollout from one episode.

## Components and ownership

### Responsibilities

- **Resources server:** When used, owns task setup, benchmark state, tools, verification, and every sandbox or service it creates.
- **Episode processor:** Owns `/run`, protocol ordering, deadline enforcement, and final result publication. A resources-backed processor asks participating servers to clean up their own state; a self-contained processor cleans up the state it creates.
- **Agent server:** When used, owns the agent implementation, its dependencies, its agent session, local subprocesses, and any fallback sandbox it creates.
- **Model server:** Exposes Gym's model API and proxies inference requests to the configured inference endpoint.
- **Sandbox server:** Optionally holds sandbox-provider state that cannot be reconstructed in another process.

```mermaid
flowchart TB
    Caller[Rollout caller] -->|POST /run| Processor[Episode processor]
    Processor -->|seed and verify| ResourcesServer[Resources server]
    Processor -->|create session, responses, close| Agent[Agent server]
    Agent -->|inference| Model[Model server]
    ResourcesServer -->|owns task sandbox| TaskSandbox[Task sandbox]
    ResourcesServer -->|borrower access| Processor
    Processor -->|passes access at session creation| Agent
    Agent -->|operates when access is present| TaskSandbox
    Agent -->|may own fallback sandbox| AgentSandbox[Agent-owned sandbox]
    ResourcesServer -.->|optional allocation| SandboxServer[Sandbox server]
    SandboxServer -.->|borrower operations| Agent
```



The diagram shows the resources-backed single-agent deployment. A concrete processor may use a different subset of servers.

In the resources-backed deployment, ownership does not move with access:

- The resources server stops a task sandbox it created.
- An agent session disconnects from borrowed sandbox access.
- An agent server stops a fallback sandbox it created.
- The processor transports sandbox access between these servers but does not use it.

### Protocol, behavior, and hosting

- The **episode protocol** defines processing order, verification timing, and cleanup. `SingleAgentEpisodeProcessor` is the initial implementation.
- **Agent behavior** defines how one `/v1/responses` request produces one `NeMoGymResponse`.
- **Agent hosting** provides the process, virtual environment, configuration, and scaling boundary for that behavior.

Changing OpenCode to Terminus-2 changes agent behavior. Adding a simulated user changes the episode protocol. Moving an agent to another image changes hosting. Integrating a self-contained external framework may place its protocol-specific components directly in a processor.

### Server types

The architecture adds a fourth server type and an optional fifth:

```text
BaseServerTypeConfig
├── ResponsesAPIModelServerTypeConfig
├── ResourcesServerTypeConfig
├── ResponsesAPIAgentServerTypeConfig
├── EpisodeProcessorServerTypeConfig
└── SandboxServerTypeConfig
```

The matching references are:

```python
class EpisodeProcessorRef(BaseModel):
    type: Literal["episode_processors"]
    name: str


class SandboxServerRef(BaseModel):
    type: Literal["sandbox_servers"]
    name: str
```

`EpisodeProcessorRef` and `SandboxServerRef` join `AgentServerRef`, `ResourcesServerRef`, and `ModelServerRef` in Gym's reference validation and `ServerClient` addressing.

## Common contracts and resources-backed flow

### Resources-backed single-agent flow

The rollout caller sends one request to the episode processor and receives one final result. The concrete processor decides which servers participate and owns the resulting protocol.

`SingleAgentEpisodeProcessor` uses the resources-backed flow below. The resources server prepares the task first. Seed creates a resources session and may return agent-visible tool access and `SandboxAccess`. The processor retains the resources session for verification and cleanup. The resources server remains the owner of its session state and task sandbox.

This processor creates an agent session before invoking the Responses API. `AgentSessionCreateRequest` contains the immutable episode ID, resources-tool access, and optional `SandboxAccess`. Sandbox selection happens while the agent server creates that session:

- When resources supplied `SandboxAccess`, the agent session connects to that sandbox as a borrower.
- When resources supplied no sandbox access and the agent requires a sandbox, the agent session creates its configured fallback sandbox.
- When the agent requires no sandbox, the agent session initializes only its tool configuration, observation state, and agent-local state.

The processor closes its agent session before asking the resources server to verify. The resources server then verifies the response and any state it owns. Finally, the processor closes the resources session, and each server destroys only the objects it owns.

```mermaid
sequenceDiagram
    autonumber
    participant C as Rollout caller
    participant P as Episode processor
    participant R as Resources server
    participant A as Agent server
    participant M as Model server
    participant TS as Resources-owned task sandbox
    participant AS as Agent-owned fallback sandbox

    C->>P: POST /run with EpisodeRequest
    P->>P: Validate, admit, and apply deadline
    P->>R: POST /seed_session
    R->>R: Create resources session and agent-visible tool access

    alt Resources provides a task sandbox
        R->>TS: Create and seed task sandbox
        R-->>P: Seed response with SandboxAccess
        P->>A: POST /v1/agent_sessions with SandboxAccess
        A->>TS: Connect as borrower
    else Resources provides no task sandbox
        R-->>P: Seed response without SandboxAccess
        P->>A: POST /v1/agent_sessions without SandboxAccess
        alt Agent requires its own sandbox
            A->>AS: Create configured fallback sandbox
        else Agent requires no sandbox
            A->>A: Initialize non-sandbox session state
        end
    end

    A->>A: Start agent-local state and observations
    A-->>P: AgentSessionCreateResponse
    P->>A: POST /v1/responses with agent session ID
    loop Agent activation
        A->>M: Model request
        M-->>A: Model response or tool call
        opt Resources-server tool call
            A->>R: Invoke tool with resources-tool access
            R-->>A: Model-visible tool result
        end
        opt Agent uses a sandbox
            alt Borrowed task sandbox
                A->>TS: Operate borrowed task sandbox
            else Agent-owned fallback sandbox
                A->>AS: Operate owned fallback sandbox
            end
        end
    end
    A-->>P: NeMoGymResponse

    P->>A: POST /v1/agent_sessions/close with AgentSessionCloseRequest
    alt Borrowed task sandbox
        A->>TS: Disconnect borrower
    else Agent-owned fallback sandbox
        A->>AS: Destroy fallback sandbox
    else No sandbox
        A->>A: Release agent-local state
    end
    A-->>P: AgentSessionCloseResponse

    alt Valid response and agent close succeeds
        P->>R: POST /verify with owner authorization
        R-->>P: EpisodeVerification
    else Invalid response or agent close failure
        P->>P: Skip verification and build failed response
    end

    P->>R: POST /close_session with ResourcesSessionCloseRequest
    R->>R: End delegated tool access
    opt Resources owns a task sandbox
        R->>TS: Destroy task sandbox
    end
    R-->>P: ResourcesSessionCloseResponse
    P-->>C: EpisodeResponse
```



### Self-contained processor

A processor does not have to use a resources server or agent server. It may validate `task_data`, run an external framework, call model servers, produce `EpisodeVerification`, and clean up its own state entirely inside `process()`.

This path is appropriate when the external framework's state, participants, tools, and verifier form one protocol that does not map cleanly onto Gym's existing server boundaries. The processor deployment supplies that framework's dependency and isolation boundary. It can later delegate individual responsibilities to Gym servers without changing the caller-facing episode contract.

### Identity

```python
class TaskIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_source: str
    task_id: str


class EpisodeId(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_id: str
    attempt: NonNegativeInt = 0
    group_id: str | None = None
    member_index: NonNegativeInt | None = None
    group_size: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        group_fields = (
            self.group_id,
            self.member_index,
            self.group_size,
        )
        if any(value is not None for value in group_fields):
            if any(value is None for value in group_fields):
                raise ValueError("group fields must be supplied together")
            if self.member_index >= self.group_size:
                raise ValueError("member_index must be less than group_size")
        return self
```

`EpisodeId` is the immutable correlation model carried through processor, resources-server, agent-server, model-call, observation, and logging paths.

- `rollout_id` remains stable when a failed rollout is retried.
- `attempt` identifies the physical retry.
- The three group fields are present only when the episode belongs to a group.
- `group_id`, `member_index`, and `group_size` remain stable when `attempt` increments.

`task_id` is separate because it identifies dataset content rather than an execution.

Current Gym constructs rollout correlation from `_ng_task_index`, `_ng_rollout_index`, and optional `_ng_attempt_index`. Compatibility translation uses the task and rollout indices for the stable `rollout_id` and keeps `_ng_attempt_index` in `attempt`.

### Request and response

```python
type JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


class EpisodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task: TaskIdentity
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    task_data: dict[str, JsonValue]
    deadline: datetime | None = None


class EpisodeFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "invalid_request",
        "unavailable",
        "agent",
        "verification",
        "deadline",
        "internal",
    ]
    message: str
    retryable: bool


class EpisodeVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float
    reward_components: dict[str, float] = Field(default_factory=dict)
    mask_sample: bool = False
    verifier_data: dict[str, JsonValue] = Field(default_factory=dict)


class EpisodeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task: TaskIdentity
    response: NeMoGymResponse | None = None
    verification: EpisodeVerification | None = None
    agent_observations: AgentObservationBundle | None = None
    failure: EpisodeFailure | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.verification is None) == (self.failure is None):
            raise ValueError("exactly one of verification or failure is required")
        if self.verification is not None and self.response is None:
            raise ValueError("a verified episode requires a response")
        return self
```

Validation enforces:

- Exactly one of `verification` or `failure` is present.
- A verified episode has a valid `response`. `mask_sample` may be true when the verifier accepts a result affected by partial infrastructure failure.
- An unverified episode may retain a valid partial agent response.
- The response `EpisodeId` and `TaskIdentity` match the request.
- `verification.reward` and every value in `verification.reward_components` are finite. The concrete verifier defines how component values produce the scalar reward.
- `verification.verifier_data` contains the remaining fields from the validated benchmark-specific verifier response.
- Failure messages are bounded and contain no credentials or internal paths.

### Grouped episodes

The rollout planner assigns group fields before dispatch. The episode processor does not infer groups from arrival order or prompt content.

- `group_id` identifies one planned group.
- `member_index` gives each member a stable position.
- `group_size` states how many members must arrive.
- A retry keeps the same group fields and increments `attempt`.
- The component implementing verification owns verifier-specific group state.

Worker-local dictionaries cannot coordinate a group spread across replicas. A grouped verifier therefore requires shared state or routing that keeps every member with the same verifier owner.

## From dataset to `/run`

Users do not add agent-session fields to datasets.

The flow is:

1. Dataset loading reads `responses_create_params`, task fields, and verifier metadata.
2. Rollout materialization assigns task, rollout, attempt, and optional group fields.
3. Compatibility translation constructs `TaskIdentity`, `EpisodeId`, and `task_data`.
4. Rollout collection sends `EpisodeRequest` to the selected episode processor.
5. The processor validates common fields before performing network calls.
6. `SingleAgentEpisodeProcessor` passes `task_data` unchanged to the resources server, which validates its benchmark-specific model. A self-contained processor validates its own task-data model.
7. The processor runs its protocol. A resources-backed single-agent processor passes exactly `responses_create_params` as the agent's `/v1/responses` body.
8. The processor returns `EpisodeResponse`, and compatibility layer restores the existing result shape when required for backward compatibility.

Existing datasets continue to carry `agent_ref` during migration. The compatibility resolver maps a migrated name to an episode-processor deployment. Native configuration selects an `EpisodeProcessorRef`.

## Resources-server session

This section applies to processors that delegate task state or verification to a resources server.

### Seed and access

The resources server creates its episode state during seed. The processor does not open an empty remote session before calling seed.

```python
class EpisodeSeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task: TaskIdentity
    task_data: dict[str, JsonValue]


class SWEBenchSeedRequest(EpisodeSeedRequest):
    task_data: SWEBenchTaskData


class MCPServerMetadata(BaseModel):
    server_name: str
    url_path: str = "/mcp"
    transport: Literal["http"] = "http"
    headers: dict[str, str]


class EpisodeSeedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    resources_tools: MCPServerMetadata | None = None
    sandbox_access: SandboxAccess | None = None
```

`EpisodeSeedRequest` validates the common envelope. Each resources server binds `/seed_session` to a concrete subclass that narrows `task_data` to its benchmark-specific Pydantic model, as illustrated by `SWEBenchSeedRequest`. FastAPI performs that validation before `seed_session()` runs. The processor posts the same JSON without importing or interpreting the benchmark model.

The processor retains the resources-server reference, session ID, and private transport state established by seed. It uses that state for verification and cleanup. It passes only agent-visible tool access and optional sandbox access to the agent server.

### Tools

Tool schemas may already appear in `responses_create_params.tools`. A resources server may also return `MCPServerMetadata` for tools exposed from the seeded session.

The processor passes that metadata during agent-session creation. The agent server configures the agent implementation before activation:

- clients that support per-server HTTP headers receive the MCP endpoint and scoped headers;
- a header-incapable black-box agent requires an agent-server-owned proxy that adds the scoped headers;
- an agent integration that supports neither path rejects the session.

The processor does not proxy individual tool calls. The resources server authorizes each call against the seeded session and returns the model-visible result. Tool credentials are never placed in model input or episode output.

### Verification and cleanup

```python
class EpisodeVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class ResourcesSessionCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resources_session_id: str


class ResourcesSessionCloseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
```

The processor uses three resources-server operations:

- `POST /seed_session`: validates task data, creates resources-server state, and returns `EpisodeSeedResponse`.
- `POST /verify`: accepts `EpisodeVerifyRequest` in the resources session established by seed and returns `EpisodeVerification`.
- `POST /close_session`: accepts `ResourcesSessionCloseRequest`, releases resources-server state, and returns `ResourcesSessionCloseResponse`.

The close request identifies the resources session. Session authorization remains in HTTP metadata established during seed.

The resources server first validates its concrete benchmark response. It places `reward`, `reward_components`, and `mask_sample` in their common fields and copies the remaining validated fields into `verifier_data`. The processor does not interpret those fields.

`/seed_session` and `/verify` preserve current Gym route names. `/close_session` is a new required lifecycle endpoint.

Cleanup is safe to retry while the owning worker remains alive. Resources-server-owned external objects use bounded TTLs. The initial design does not provide exactly-once execution or recovery across worker restart.

## Sandbox access

```python
class DirectSandboxConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["direct"]
    sandbox_provider: str
    descriptor: dict[str, JsonValue]


class SandboxServerConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["sandbox_server"]
    sandbox_ref: SandboxRef


class SandboxAccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection: Annotated[
        DirectSandboxConnection | SandboxServerConnection,
        Field(discriminator="kind"),
    ]
    workdir: str
```

`SandboxAccess` is a serializable reference that lets an agent server connect to a resources-server-owned task sandbox. The connection discriminator tells the agent whether to reconnect through a provider or a sandbox server. `workdir` identifies where agent commands run.

The agent receives a borrower interface for command execution and file transfer within the task workdir. The resources server retains the owner authority used to stop the sandbox after verification.

Borrower restrictions require enforcement by the provider or sandbox server. A Python facade alone prevents accidental misuse but is not an authorization boundary.

For direct access, `sandbox_provider` names an existing top-level Gym sandbox-provider block such as `sandbox`. The resources server creates and serializes the sandbox with that named configuration. The agent server resolves the same name through `server_client.global_config_dict`, constructs the provider, and calls `connect(descriptor)`. No borrowed-sandbox provider setting is added to agent configuration.

Direct handoff requires a named provider configuration because an inline provider mapping may contain credentials and has no stable reference. The agent-server environment must have the provider implementation installed; otherwise agent-session creation fails before inference. Agent-session close disconnects the borrowed provider client without calling the owner operation that destroys the physical sandbox.

### Seed behavior

- When `EpisodeSeedResponse.sandbox_access` is present, the resources server requires the agent to operate that task sandbox.
- When it is absent, the agent follows its own configuration.
- Absence does not mean the resources server failed to provision a required sandbox. That failure makes seed fail.
- The resources server may create other private sandboxes for tools or verification without exposing them.

The processor passes `SandboxAccess` unchanged during agent-session creation.

### Optional sandbox server

Direct handoff is valid when the provider can serialize and reconnect its sandbox across processes. Its access restrictions are limited to what that provider enforces; the agent server must expose only borrower operations and disconnect without destroying the sandbox.

Providers that cannot reconnect directly, or deployments that require server-enforced borrower authorization, use a sandbox server such as the server proposed in [PR #2085](https://github.com/NVIDIA-NeMo/Gym/pull/2085). `SandboxServerConnection.sandbox_ref` carries that server's sandbox ID, operate lease, endpoint, and workdir. Exact lease, revocation, and provider configuration remain part of the sandbox-server design.

## Agent-server session

This section applies to processors that delegate agent behavior to an agent server.

### Why a session is required

The `/v1/responses` request body remains unchanged:

```text
POST /v1/responses

NeMoGymResponseCreateParamsNonStreaming
    -> NeMoGymResponse
```

The agent still needs episode-scoped state that is not part of the Responses API:

- immutable `EpisodeId`
- resources-tool access
- optional borrowed sandbox access
- agent-local subprocess or fallback-sandbox handles
- an observation buffer
- owned-object cleanup

The single-agent processor creates that state before calling `/v1/responses` and closes it afterward.

A future user-simulation processor can create assistant and simulated-user sessions once, call `/v1/responses` repeatedly according to the interaction protocol, and close both sessions before verification. The processor owns role visibility, ordering, and termination. A separate turn endpoint is unnecessary unless a future protocol cannot express an activation through the Responses API.

### Data models

```python
class AgentSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    resources_tools: MCPServerMetadata | None = None
    sandbox_access: SandboxAccess | None = None
    deadline: datetime | None = None


class AgentSessionCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_session_id: str


class AgentSessionCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_session_id: str


class AgentSessionCloseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_session_id: str
    agent_observations: AgentObservationBundle | None = None
```

### `POST /v1/agent_sessions`

The processor sends `AgentSessionCreateRequest` to the configured `AgentServerRef`.

The agent server:

1. Validates the episode, deadline, tool access, and sandbox access.
2. Configures resources tools and either connects the borrowed sandbox or creates its configured fallback sandbox.
3. Starts agent-local state and an observation buffer correlated with `EpisodeId`.
4. Registers cleanup for objects it owns and returns `AgentSessionCreateResponse`.

If setup fails, the agent server releases anything it already acquired and returns no usable session.

### `POST /v1/responses`

The processor sends the ordinary Responses body with:

```text
X-NeMo-Gym-Agent-Session-Id: <agent_session_id>
```

The agent server:

- rejects a missing, expired, closed, or mismatched session
- places the session's immutable `EpisodeId` in Gym's request-scoped correlation context
- exposes `EpisodeId`, resources-tool access, and optional sandbox access to the agent implementation
- performs one complete agent activation
- returns `NeMoGymResponse`

The endpoint does not seed resources-server state, verify a reward, clean up resources-server-owned state, or publish an episode result.

The initial single-agent processor performs one `/v1/responses` activation per session. Concurrent or repeated activations are rejected.

### `POST /v1/agent_sessions/close`

The processor sends `AgentSessionCloseRequest` in the body.

The agent server:

1. Stops the activation and agent-owned subprocesses.
2. Stops an agent-owned fallback sandbox or disconnects from a borrowed sandbox.
3. Flushes observations and returns `AgentSessionCloseResponse`.

A successful close means agent-controlled activity has stopped. If close fails, the processor does not begin final verification. Session expiry triggers cleanup while the worker remains alive; provider TTLs bound external objects after worker loss.

## Single-agent processor

### Configuration

```python
class BaseEpisodeProcessorConfig(BaseRunServerInstanceConfig):
    pass


class SingleAgentEpisodeProcessorConfig(BaseEpisodeProcessorConfig):
    resources_server: ResourcesServerRef
    agent_server: AgentServerRef


class Tau2EpisodeProcessorConfig(BaseEpisodeProcessorConfig):
    model_server: ModelServerRef
    user_model_server: ModelServerRef
```

Each concrete processor declares the servers it uses. Task data cannot select executable code, credentials, another processor, or a sandbox provider.

For `SingleAgentEpisodeProcessor`, agent configuration owns model selection and behavior-specific options, while resources-server configuration owns verification options. A self-contained processor owns its external framework's configuration.

### Processing order

The presence of `EpisodeSeedResponse.sandbox_access` selects the sandbox lifecycle. The processor does not infer ownership from the agent type, provider, or task name.

#### The resources server provides the task sandbox

The resources server creates the task sandbox during seed and retains owner authority. The agent receives only `SandboxAccess`, connects while its session is active, and disconnects before verification. The resources server can then inspect stable task state and destroy the sandbox during resources-session cleanup.

This ordering is required for Terminal Bench-style verification. Successful agent close ends agent-controlled activity but does not destroy resources-server-owned state. Verification runs after that boundary and before resources-server cleanup.

#### The agent requires its own sandbox

When seed returns no `sandbox_access`, the agent follows its own configuration. A sandbox-dependent agent may create a fallback sandbox as agent-session-local state. Agent close destroys that sandbox before verification, so the verifier can use the model response and resources-server-owned state but cannot inspect the fallback sandbox.

This path is valid only when verification does not need the live contents of the agent-owned sandbox. If success depends on files, services, packages, permissions, or processes in that box, the resources server must provide the task sandbox during seed. The initial design deliberately has no generic artifact transfer from an agent-owned sandbox to a verifier.

#### Each server cleans up the objects it owns

The processor applies the same unwind order after success, failure, deadline expiry, or caller cancellation. It invokes each server's lifecycle endpoint, and each server releases the objects it owns.

The protocol requires:

- `SandboxAccess` is delegated only at agent-session creation. A `/v1/responses` body never carries infrastructure credentials.
- The processor closes every agent session before final verification.
- Live state in an agent-owned fallback sandbox is unavailable after agent close. Supporting verification of that state would require a new retained-artifact or ownership-transfer contract, which is outside the initial design.

```python
async def process(
    request: EpisodeRequest,
    context: EpisodeContext,
) -> EpisodeResponse:
    seed = await context.resources_server.seed(
        EpisodeSeedRequest(
            episode_id=request.episode_id,
            task=request.task,
            task_data=request.task_data,
        )
    )

    try:
        session = await self.create_agent_session(
            AgentSessionCreateRequest(
                episode_id=request.episode_id,
                resources_tools=seed.resources_tools,
                sandbox_access=seed.sandbox_access,
                deadline=request.deadline,
            )
        )

        try:
            agent_response = await self.call_agent_responses(
                session,
                request.responses_create_params,
            )
        finally:
            close_response = await self.close_agent_session(
                session,
                AgentSessionCloseRequest(
                    agent_session_id=session.agent_session_id,
                ),
            )

        verification = await context.resources_server.verify(
            EpisodeVerifyRequest(
                episode_id=request.episode_id,
                responses_create_params=request.responses_create_params,
                response=agent_response,
            )
        )

        return EpisodeResponse(
            episode_id=request.episode_id,
            task=request.task,
            response=agent_response,
            verification=verification,
            agent_observations=close_response.agent_observations,
        )
    finally:
        await context.resources_server.close_session(
            ResourcesSessionCloseRequest(
                resources_session_id=seed.resources_session_id,
            )
        )
```

This pseudocode shows order, not error-handling implementation. `BaseEpisodeProcessor.run()` provides:

- native or compatibility request validation;
- deadline clamping;
- worker-local admission;
- exception classification;
- response validation;
- telemetry;
- native or compatibility HTTP projection.

`call_agent_responses` either returns a validated `NeMoGymResponse` or raises a classified agent error. `close_agent_session` raises when it cannot confirm that agent-controlled activity stopped.

A valid agent response proceeds to verification only after every agent session closes successfully. Failure to produce a valid `NeMoGymResponse` or close every agent session skips verification.

### Failure behavior

- Invalid input fails before seed.
- A valid verifier result includes reward and `mask_sample`, even when the verifier masks a partial infrastructure failure.
- If no valid verifier result can be obtained, `EpisodeResponse.failure` describes the failure and may retain a partial agent response.
- Agent and resources-server cleanup still run on handled failures and caller cancellation.
- Cleanup failures after verification are reported through telemetry rather than training or evaluation data.

No failure path invents a successful native reward.

The compatibility projector maps native failures to the behavior required by each migrated deployment.

## Agent-server scaling

Agent sessions contain process-local objects. All calls for one session must reach the same process.

The initial deployment uses one Uvicorn worker per agent-server replica and scales through replicas or NeMo RL shards. The processor retains the resolved replica endpoint and agent-session ID until close. An `AgentServerRef` name alone does not provide session affinity.

The same rule applies to process-local resources sessions: seed, tools, verify, and cleanup use one directly resolved resources-server replica. A resources server behind a load-balanced address must provide affinity routing or shared session state.

If an agent-server replica dies, its process-local sessions are lost. The episode fails and may restart with a higher attempt. The initial design does not migrate a live CLI process or sandbox client to another replica.

Future routing or shared-state implementations can preserve the same agent-session API.

## Deployment examples

### Simple Agent

Simple Agent uses resources-server tools and no sandbox.

```yaml
reasoning_gym_simple_agent:
  episode_processors:
    single_agent_episode_processor:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: reasoning_gym
      agent_server:
        type: responses_api_agents
        name: simple_agent
      max_concurrent_episodes: 32
      queue_timeout_seconds: 300
      shutdown_grace_seconds: 60
      default_episode_timeout_seconds: 3600
      compatibility:
        expected_agent_name: reasoning_gym_simple_agent

simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      num_workers: 1
      model_server:
        type: responses_api_models
        name: agent_model
      max_steps: null
```

Session creation fails before inference if the resources server supplies sandbox access that Simple Agent cannot use.

### OpenCode with an agent-owned sandbox

Reasoning Gym supplies no task-sandbox access. OpenCode creates and owns its configured fallback sandbox.

```yaml
reasoning_gym_opencode:
  episode_processors:
    single_agent_episode_processor:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: reasoning_gym
      agent_server:
        type: responses_api_agents
        name: opencode_agent
      max_concurrent_episodes: 16
      queue_timeout_seconds: 300
      shutdown_grace_seconds: 60
      default_episode_timeout_seconds: 10800

opencode_agent:
  responses_api_agents:
    opencode_agent:
      entrypoint: app.py
      num_workers: 1
      model_server:
        type: responses_api_models
        name: agent_model
      opencode_version: 1.17.11
      opencode_max_context_window: 262144
      opencode_config: {}
      sandbox_provider: sandbox
      sandbox_config:
        image: approved-opencode-runtime@sha256:...
        workdir: /workspace
        ttl_s: 18000
        ready_timeout_s: 1200
        resources:
          cpu: 2
          memory_mib: 8192
          disk_gib: 30
        provider_options: {}
```

The agent session starts and later stops this fallback sandbox.

### OpenCode with a resources-server-owned task sandbox

SWE-bench creates the task sandbox during seed. The same OpenCode agent uses the returned borrower access instead of its fallback configuration.

```yaml
opencode_swebench:
  episode_processors:
    single_agent_episode_processor:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: swebench
      agent_server:
        type: responses_api_agents
        name: opencode_agent
      max_concurrent_episodes: 16
      queue_timeout_seconds: 300
      shutdown_grace_seconds: 60
      default_episode_timeout_seconds: 10800

swebench:
  resources_servers:
    swebench:
      entrypoint: app.py
      num_workers: 1
      evaluation_timeout: 1800
      sandbox_provider: sandbox
      sandbox_config:
        ttl_s: 18000
        ready_timeout_s: 1200
        resources:
          cpu: 2
          memory_mib: 16384
          disk_gib: 30
        provider_options: {}
```

The resources server retains owner authority, verifies while the task sandbox is alive, and stops it after verification. The agent session only disconnects. For direct access, both servers resolve the same named `sandbox_provider` block and the provider must support `serialize()` and `connect()`.

For a process-bound provider, the resources server uses the optional sandbox server. Its provider and lease configuration remain defined by the sandbox-server design.

### Terminus-2

Terminus-2 remains an agent server with Harbor and terminal-agent dependencies in its own virtual environment.

```yaml
reasoning_gym_terminus_2:
  episode_processors:
    single_agent_episode_processor:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: reasoning_gym
      agent_server:
        type: responses_api_agents
        name: terminus_2_agent
      max_concurrent_episodes: 8
      queue_timeout_seconds: 300
      shutdown_grace_seconds: 60
      default_episode_timeout_seconds: 10800

terminus_2_agent:
  responses_api_agents:
    terminus_2_agent:
      entrypoint: app.py
      num_workers: 1
      model_server:
        type: responses_api_models
        name: agent_model
      concurrency: 1
      workspace_root: null
      max_turns: null
      parser_name: json
      command_timeout_sec: 1800
      timeout: 10800
```

With sandbox access, terminal commands operate the borrowed task sandbox. Without access, they operate the configured local workspace. The Terminus-2 Python dependencies remain in the agent-server environment in either case.

### Self-contained Tau2 processor

Tau2 can run as an episode processor without a Gym resources server or agent server. The processor calls the policy and simulated-user model servers while the Tau2 library owns the interaction, tools, state, and evaluation.

```yaml
tau2:
  episode_processors:
    tau2_episode_processor:
      entrypoint: app.py
      model_server:
        type: responses_api_models
        name: policy_model
      user_model_server:
        type: responses_api_models
        name: policy_model
      num_workers: 1
      user_llm_args: {}
      max_steps: 200
      debug: false
      print_step_counts: false
      datasets:
      - name: example
        type: example
        jsonl_fpath: episode_processors/tau2/data/example.jsonl
```

The processor validates the Tau2 task data, runs the complete simulation, converts the trajectory to `NeMoGymResponse`, and returns `EpisodeVerification`. A later Tau2 integration may delegate participants or state to Gym servers without changing `EpisodeRequest` or `EpisodeResponse`.

## Verification remains benchmark-defined

The base processor does not define a generic sandbox harvester or submission format. A concrete processor may verify directly or delegate verification to a resources server. The verifier validates benchmark-specific results and returns their additional fields in `EpisodeVerification.verifier_data`.

## Compatibility and migration

Unmigrated agents retain their existing episode-level `/run` and do not use an episode processor.

### Migrated compatibility boundary

A migrated deployment:

1. Keeps the legacy deployment name.
2. Resolves that name under `episode_processors`.
3. Validates the existing `BaseRunRequest`.
4. Converts the materialized rollout into `EpisodeRequest`.
5. Runs `SingleAgentEpisodeProcessor`.
6. Projects the native result back to the existing response shape.

The translator and projector belong to the configured compatibility processor. They preserve the current deployment's request and result contract without adding a dynamic adapter registry.

Rollout logging persists `verifier_data` as part of `EpisodeResponse`. The compatibility projector restores those fields at the top level before existing result logging and aggregation.

### Effect on evaluation and training

Each migration adapter is responsible for preserving the existing dataset, command, concurrency, and result interfaces for that deployment.

During migration, routing changes behind the existing deployment name. Native callers later address `EpisodeProcessorRef` directly.

NeMo RL sharding changes its run target from an agent-server deployment to an episode-processor deployment. Each shard can bind a processor replica to an agent-server replica. Native NeMo RL consumption can adopt `EpisodeResponse` separately from the processor migration.

## Service constraints

### Worker-local state

The initial implementation keeps admission and session state within one worker. It does not provide transparent failover or cluster-wide attempt fencing. A worker failure fails the attempt; provider TTLs bound externally allocated objects left behind.

### Telemetry

Episode processors and sandbox servers register with Gym's existing telemetry initialization. Every server propagates `EpisodeId` and trace context.

## Extension points

Future processors may add interaction APIs and scheduling rules. They can reuse agent and resources sessions when those server boundaries fit the protocol.

Delivery order, implementation workstreams, and integration gates live in [episode-orchestration-milestones.md](episode-orchestration-milestones.md).

## Evidence from current Gym

### Simple Agent

Current `SimpleAgent.run()`:

1. sends the materialized row to the resources server's `/seed_session`;
2. carries resources-server cookies into a self-call to `/v1/responses`;
3. runs the model-and-tool loop;
4. sends the response to the resources server's `/verify`;
5. projects the verifier result.

The migrated processor owns steps 1, 2, 4, and 5. The agent server retains the model-and-tool loop. Agent-session creation carries episode context and configures resources tools before the Responses call.

### OpenCode sandbox pairings

SWE-bench, DeepSWE, SWE-bench Pro, and Terminal Bench currently create task sandboxes before OpenCode runs. The current agent receives a bare sandbox ID and reconnects through an independently configured provider.

The pairings demonstrate:

- the resources server must retain sandbox ownership through verification
- the agent needs complete borrower connection data and the selected workdir
- a provider that cannot reconnect across processes needs a sandbox server
- agent-server and resources-server cleanup must not both stop the task sandbox
- benchmark-specific extraction must remain in the resources server

### GDPVal

GDPVal verification depends on benchmark-specific deliverables. Its resources server must collect them into resources-owned state during the interaction or derive them from the agent response after close. The episode processor does not define a generic artifact-harvesting contract.

### Tau2

Current `Tau2Agent.run()` calls Tau2's `run_single_task()` directly and returns the library's reward. Its configuration references policy and simulated-user model servers but no Gym resources server, and its `/v1/responses` implementation is intentionally absent. The existing component is therefore an episode-level integration currently hosted under the agent-server type.