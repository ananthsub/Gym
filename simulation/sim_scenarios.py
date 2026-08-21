# Scenario matrix for the token-id-capture simulation. See sim_stack.py.
#
# Verdicts:
#   OK             — healthy scenario behaved as the design promises
#   SAFE_MASKED    — adverse scenario was correctly refused (masked), nothing fabricated
#   BUG_REPRODUCED — a known review finding reproduced exactly as predicted
#   SURPRISE       — behavior matched neither the design promise nor the predicted bug

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nemo_gym.token_id_capture import TokenCaptureStore, install_token_sink
from nemo_gym.token_id_capture.consumer import trajectories_from_source

from sim_stack import (
    DuplicatingSource,
    FailingSink,
    Harness,
    MemorySink,
    MemorySource,
    PreDispatchFailingSink,
    SimEngine,
    build_server,
    uninstall_sinks,
    verify_no_fabrication,
)


@dataclass
class Outcome:
    name: str
    verdict: str
    masked: bool | None
    statuses: list[str]
    supplied: list[bool]
    delivered_calls: int
    engine_calls: int
    fabrication: str
    http: list[int]
    note: str


def entry_facts(entries: list[Any]) -> tuple[list[str], list[bool]]:
    ordered = sorted(entries, key=lambda e: e.created_at)
    statuses = [str(e.parent_resolution.value) if e.parent_resolution else "none" for e in ordered]
    supplied = [bool(getattr(e, "prefix_supplied", False)) for e in ordered]
    return statuses, supplied


def delivered_count(built: dict | None) -> int:
    if not built or not built.get("rebuilt_response"):
        return 0
    return sum(1 for item in built["rebuilt_response"].get("output", []) if item.get("generation_token_ids"))


def run_scenario(
    name: str,
    *,
    engine: SimEngine,
    supply: bool,
    drive: Callable[[Harness], None],
    use_dir: bool = True,
    installed_sink: Callable[[Path], Any] | None = None,
    source_wrapper: Callable[[Any], Any] | None = None,
    expect_masked: bool | None = None,
    expect_statuses: list[str] | None = None,
    expect_delivered: int | None = None,
    bug: str | None = None,
    note: str = "",
) -> Outcome:
    tmp = Path(tempfile.mkdtemp(prefix=f"sim_{name}_"))
    rid = f"sim-{name}"
    sink_obj = None
    try:
        if installed_sink is not None:
            sink_obj = installed_sink(tmp)
            install_token_sink(sink_obj)
        client = build_server(engine, str(tmp) if use_dir else None, supply)
        harness = Harness(client=client, rollout_id=rid)
        drive(harness)

        if isinstance(sink_obj, MemorySink):
            source: Any = MemorySource(sink_obj)
            entries = list(sink_obj.entries.get(rid, {}).values())
        else:
            source = TokenCaptureStore(tmp)
            entries = source.read_entries(rid)
        if source_wrapper is not None:
            source = source_wrapper(source)

        built = asyncio.run(trajectories_from_source(rid, source))
        statuses, supplied = entry_facts(entries)
        masked = bool(built.get("mask_sample")) if built else None
        delivered = delivered_count(built)
        ok, fabrication = verify_no_fabrication(built or {}, engine)

        # Completeness: an unmasked delivery must cover every call the engine served for
        # the main chain — fewer calls than served, unmasked, means silent truncation.
        # (Scenarios that intentionally abandon calls set expect_delivered explicitly.)
        truncated = (not masked) and delivered < engine.calls and delivered > 0
        if expect_delivered is not None and delivered == expect_delivered:
            truncated = False

        if not ok:
            verdict = "SURPRISE"
        elif bug is not None:
            predicted = (
                (bug == "F1" and truncated and masked is False)
                or (bug == "F3" and masked is True and all(s in ("root", "unresolved") for s in statuses))
                or (bug == "F10" and masked is True)
                or (bug == "F11" and masked is True and statuses.count("root") > 1)
                or (bug == "F15" and masked is False and delivered == 0)
                or (bug == "F7" and 500 in harness.statuses)
                or (bug == "F13" and masked is True)
                or (bug == "F16" and masked is True and statuses and statuses[-1] == "unresolved")
            )
            if predicted:
                verdict = "BUG_REPRODUCED"
            elif masked:
                verdict = "NOT_REPRODUCED_SAFE"  # predicted defect didn't fire; behavior was safe
            elif ok and not truncated:
                verdict = "FIXED"  # predicted defect gone AND the rollout is healthy-trainable
            else:
                verdict = "SURPRISE"
        else:
            matches = True
            if expect_masked is not None and masked is not expect_masked:
                matches = False
            if expect_statuses is not None and statuses != expect_statuses:
                matches = False
            if expect_delivered is not None and delivered != expect_delivered:
                matches = False
            if truncated:
                matches = False
            verdict = ("OK" if not expect_masked else "SAFE_MASKED") if matches else "SURPRISE"

        return Outcome(
            name=name,
            verdict=verdict,
            masked=masked,
            statuses=statuses,
            supplied=supplied,
            delivered_calls=delivered,
            engine_calls=engine.calls,
            fabrication=fabrication,
            http=harness.statuses,
            note=note,
        )
    finally:
        uninstall_sinks()


def three_turns(harness: Harness) -> None:
    harness.turn("first question")
    harness.turn("second question")
    harness.turn("third question")


def scenarios() -> list[Outcome]:
    results: list[Outcome] = []

    # --- healthy paths -----------------------------------------------------------------
    results.append(
        run_scenario(
            "faithful_linear",
            engine=SimEngine(),
            supply=True,
            drive=three_turns,
            expect_masked=False,
            expect_statuses=["root", "resolved", "resolved"],
            expect_delivered=3,
            note="baseline: faithful harness, prefix supply on",
        )
    )

    def tool_flow(harness: Harness) -> None:
        harness.turn("use the tool")
        harness.history.append({"role": "tool", "tool_call_id": "call-1", "content": "tool says 42"})
        harness.turn("now answer")

    results.append(
        run_scenario(
            "tool_args_reserialized",
            engine=SimEngine(
                scripted=[
                    {
                        "content": "calling",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"query": "x", "k": 3}'},
                            }
                        ],
                    },
                    {"content": "final"},
                    {"content": "extra"},
                ]
            ),
            supply=True,
            drive=lambda h: (
                setattr(
                    h,
                    "echo_mutator",
                    lambda m: {
                        **m,
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {
                                    "name": c["function"]["name"],
                                    # harness re-serializes args: key order + spacing change
                                    "arguments": '{"k":3,"query":"x"}',
                                },
                            }
                            for c in m.get("tool_calls", [])
                        ]
                    }
                    if m.get("tool_calls")
                    else m,
                ),
                tool_flow(h),
            )[-1],
            expect_masked=False,
            expect_statuses=["root", "resolved"],
            expect_delivered=2,
            note="harness reorders tool-call JSON keys; canonicalization must still match",
        )
    )

    # --- drift: the reason prefix supply exists ---------------------------------------
    results.append(
        run_scenario(
            "drift_no_supply",
            engine=SimEngine(drift_on_rerender=True),
            supply=False,
            drive=three_turns,
            expect_masked=True,
            note="engine re-tokenizes history differently; lineage resolves but token chain breaks; must mask",
        )
    )
    results.append(
        run_scenario(
            "drift_with_supply",
            engine=SimEngine(drift_on_rerender=True),
            supply=True,
            drive=three_turns,
            expect_masked=False,
            expect_statuses=["root", "resolved", "resolved"],
            expect_delivered=3,
            note="same drift, but exact-prefix supply preserves the chain — the flagship value demo",
        )
    )

    # --- harness mutations -------------------------------------------------------------
    results.append(
        run_scenario(
            "think_stripped",
            engine=SimEngine(scripted=[{"content": "<think> hidden reasoning </think> visible answer"}]),
            supply=True,
            drive=lambda h: (
                setattr(h, "echo_mutator", lambda m: {**m, "content": (m.get("content") or "").split("</think>")[-1].strip()}),
                three_turns(h),
            )[-1],
            expect_masked=True,
            note="inline <think> stripped by harness: fingerprint miss, no supply, masked (documented 2181 gap)",
        )
    )
    results.append(
        run_scenario(
            "reminder_inserted",
            engine=SimEngine(),
            supply=True,
            drive=lambda h: (
                setattr(h, "pre_echo_insert", lambda n: {"role": "system", "content": "reminder"} if n == 2 else None),
                three_turns(h),
            )[-1],
            expect_masked=False,
            note="adjacency hole: system item inserted between context and echo still RESOLVES; with supply the item has no token effect",
        )
    )
    results.append(
        run_scenario(
            "history_edited",
            engine=SimEngine(),
            supply=True,
            drive=lambda h: (
                three_turns(h),
                h.history.__setitem__(0, {"role": "user", "content": "a summarized question"}),
                h.turn("fourth question"),
            )[-1],
            expect_masked=True,
            note="context edited (compaction-like): digest verification must refuse the parent",
        )
    )

    def fork(harness: Harness) -> None:
        harness.turn("plan the work")
        base = list(harness.history)
        harness.call(base + [{"role": "user", "content": "branch A"}])
        harness.call(base + [{"role": "user", "content": "branch B"}])

    results.append(
        run_scenario(
            "fork_subagents",
            engine=SimEngine(),
            supply=True,
            drive=fork,
            expect_masked=True,
            note="two children of one parent: two chains, masked (F11 yield policy)",
        )
    )
    results.append(
        run_scenario(
            "aux_title_call",
            engine=SimEngine(),
            supply=True,
            drive=lambda h: (three_turns(h), h.call([{"role": "user", "content": "write a short title"}]))[-1],
            expect_masked=True,
            bug="F11",
            note="auxiliary call becomes a second root and masks the whole rollout (F11)",
        )
    )

    # --- retries -----------------------------------------------------------------------
    def retry_nonfinal(harness: Harness) -> None:
        harness.turn("first question")
        request = list(harness.history) + [{"role": "user", "content": "second question"}]
        harness.call(request)  # times out client-side; harness retries:
        harness.turn("second question")
        harness.turn("third question")

    results.append(
        run_scenario(
            "retry_nonfinal_identical",
            engine=SimEngine(scripted=[{"content": "one"}, {"content": "two"}, {"content": "two"}, {"content": "three"}]),
            supply=True,
            drive=retry_nonfinal,
            bug="F16",
            expect_delivered=3,  # the kept chain; the abandoned duplicate is correctly excluded
            note="identical mid-rollout retry: BOTH copies match the continuation, ambiguity masks the rest (yield regression vs legacy builder)",
        )
    )
    results.append(
        run_scenario(
            "retry_nonfinal_divergent",
            engine=SimEngine(scripted=[{"content": "one"}, {"content": "two-a"}, {"content": "two-b"}, {"content": "three"}]),
            supply=True,
            drive=retry_nonfinal,
            expect_masked=False,
            expect_delivered=3,
            note="retry with a DIFFERENT sampled answer: the echo identifies the kept copy, chain stays trainable (abandoned attempt excluded)",
        )
    )

    def retry_final(harness: Harness) -> None:
        harness.turn("first question")
        request = list(harness.history) + [{"role": "user", "content": "second question"}]
        harness.call(request)
        harness.call(request)

    results.append(
        run_scenario(
            "retry_final_identical",
            engine=SimEngine(scripted=[{"content": "one"}, {"content": "same answer"}, {"content": "same answer"}]),
            supply=True,
            drive=retry_final,
            bug="F15",
            note="final call retried with identical tokens: hunting F15's unmasked-empty delivery",
        )
    )
    results.append(
        run_scenario(
            "retry_only_call_identical",
            engine=SimEngine(scripted=[{"content": "same"}, {"content": "same"}]),
            supply=True,
            drive=lambda h: (h.call([{"role": "user", "content": "q"}]), h.call([{"role": "user", "content": "q"}]))[-1],
            bug="F15",
            note="single-call rollout retried identically: hunting F15's unmasked-empty delivery",
        )
    )

    # --- empty generation parent (F13 terrain) ----------------------------------------
    results.append(
        run_scenario(
            "empty_generation_parent",
            engine=SimEngine(scripted=[{"content": "one"}, {"content": "", "empty_generation": True}, {"content": "three"}]),
            supply=True,
            drive=three_turns,
            bug="F13",
            note="call 2 generates nothing; call 3's parent terrain (F13): expect masked, record exact path",
        )
    )

    # --- prefix-supply proof and backend behaviors ------------------------------------
    results.append(
        run_scenario(
            "engine_ignores_prefix_stable",
            engine=SimEngine(honor_prefix=False),
            supply=True,
            drive=three_turns,
            expect_masked=False,
            note="backend ignores the field but re-renders prefix-stably: proof passes (contiguity proof, not extension proof)",
        )
    )
    results.append(
        run_scenario(
            "proof_bundle_shape",
            engine=SimEngine(proof_shape="bundle"),
            supply=True,
            drive=three_turns,
            bug="F7",
            note="engine returns prompt ids in the message bundle: F7 predicts a spurious hard failure (500)",
        )
    )

    # --- sink failures -----------------------------------------------------------------
    results.append(
        run_scenario(
            "sink_outage_final_call",
            engine=SimEngine(),
            supply=True,
            drive=three_turns,
            installed_sink=lambda tmp: FailingSink(TokenCaptureStore(tmp), fail_on_call=3),
            bug="F1",
            note="put+mark_incomplete both fail on the FINAL call: F1 predicts unmasked truncated delivery",
        )
    )
    results.append(
        run_scenario(
            "sink_outage_pre_dispatch",
            engine=SimEngine(),
            supply=True,
            drive=three_turns,
            installed_sink=lambda tmp: PreDispatchFailingSink(TokenCaptureStore(tmp), fail_on_call=3),
            expect_masked=False,
            expect_delivered=2,
            note="backend already down when call 3 arrives: intent fails PRE-generation, call fails at zero cost, delivered 2/2 is the complete story",
        )
    )
    results.append(
        run_scenario(
            "sink_outage_mid_rollout",
            engine=SimEngine(),
            supply=True,
            drive=three_turns,
            installed_sink=lambda tmp: FailingSink(TokenCaptureStore(tmp), fail_on_call=2),
            expect_masked=True,
            note="same outage mid-rollout: the next call cannot resolve its parent, sample masks (2180 repairs F1 here)",
        )
    )

    # --- misconfiguration and transport ------------------------------------------------
    results.append(
        run_scenario(
            "custom_sink_no_lineage",
            engine=SimEngine(),
            supply=True,
            drive=three_turns,
            use_dir=False,
            installed_sink=lambda tmp: MemorySink(),
            bug="F3",
            note="custom sink, no lineage store, one worker: F3 predicts all-UNRESOLVED with no visibility",
        )
    )
    results.append(
        run_scenario(
            "duplicate_snapshot_entry",
            engine=SimEngine(),
            supply=True,
            drive=three_turns,
            source_wrapper=DuplicatingSource,
            bug="F10",
            note="at-least-once transport duplicates one snapshot entry: F10 predicts a healthy rollout masks",
        )
    )

    return results


def main() -> None:
    results = scenarios()
    rows = []
    for r in results:
        rows.append(
            {
                "scenario": r.name,
                "verdict": r.verdict,
                "masked": r.masked,
                "statuses": r.statuses,
                "prefix_supplied": r.supplied,
                "delivered_calls": r.delivered_calls,
                "engine_calls": r.engine_calls,
                "fabrication_check": r.fabrication,
                "http_statuses": r.http,
                "note": r.note,
            }
        )
    out = Path(__file__).parent / "scenario_results.json"
    out.write_text(json.dumps(rows, indent=1))
    width = max(len(r.name) for r in results)
    print(f"{'scenario':<{width}}  {'verdict':<15} masked  statuses / delivered")
    for r in results:
        print(
            f"{r.name:<{width}}  {r.verdict:<15} {str(r.masked):<7} "
            f"{','.join(r.statuses) or '-'}  d={r.delivered_calls}/{r.engine_calls}  {r.fabrication}"
        )
    surprises = [r for r in results if r.verdict == "SURPRISE"]
    print(f"\n{len(results)} scenarios: "
          f"{sum(r.verdict == 'OK' for r in results)} OK, "
          f"{sum(r.verdict == 'SAFE_MASKED' for r in results)} SAFE_MASKED, "
          f"{sum(r.verdict == 'BUG_REPRODUCED' for r in results)} BUG_REPRODUCED, "
          f"{len(surprises)} SURPRISE")
    for r in surprises:
        print(f"  SURPRISE: {r.name} — masked={r.masked} statuses={r.statuses} "
              f"delivered={r.delivered_calls}/{r.engine_calls} http={r.http} ({r.fabrication})")


if __name__ == "__main__":
    main()
