#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ray GCS burst-tolerance tuning for ng_run at high N (~256+ sub-servers).
#
# When `nemo_gym/cli.py:RunHelper.start` fans out 2N+1 child Popens in parallel
# during the spawn loop, each child's `ray.init(address=…)` produces a
# RegisterWorker RPC against the head's single GCS. At N=256 (514 children)
# this storm overwhelms GCS: the queue backs up, client connects time out
# at the default 5 s ceiling, raylet kills the unregistered workers, and ng_run
# never reaches "All N/N servers ready". See
# `investigations/nemo-gym-scale-testing.md` §10.3.3 (assumption A12) for the
# full diagnosis.
#
# This file is a workaround, not a fix. It increases GCS throughput, extends
# timeouts to absorb the burst, and reduces background RPC noise so the burst
# has more room. The architectural fix (staggered spawn loop in `RunHelper.start`)
# is documented in the same section and is not on this critical path.
#
# Source this file from the Slurm launcher so the env propagates into sbatch
# via `--export=ALL`, and from any driver wrapper that exec's into a long-lived
# shell on a compute node. All `export RAY_xxx="${RAY_xxx:-…}"` are guarded so
# explicit per-run overrides win.
#
# Config reference: https://github.com/ray-project/ray/blob/master/src/ray/common/ray_config_def.h
# Source: adapted from the nemo-rl scale-launcher's GCS burst-tuning block.

# --- GCS RPC thread pools ---
# gcs_server_rpc_server_thread_num (default max(1, num_cpus/4)): threads that
#   poll for incoming gRPC requests. 64 ensures the GCS can drain large bursts
#   of actor creation RPCs without excessive queueing.
# num_server_call_thread (default max(1, num_cpus/4)): threads that send gRPC
#   replies.
# gcs_max_active_rpcs_per_handler auto-scales to rpc_thread_num * 100.
export RAY_gcs_server_rpc_server_thread_num="${RAY_gcs_server_rpc_server_thread_num:-64}"
export RAY_num_server_call_thread="${RAY_num_server_call_thread:-32}"

# --- Registration and RPC timeouts ---
# With thread tuning, GCS should drain even the worst burst in ~60s. 120s gives
# 2x margin while keeping failure detection under 2 minutes.
#
# worker_register_timeout_seconds (default 60s): raylet kills unregistered
#   workers after this timeout.
# gcs_rpc_server_connect_timeout_s (default 5s): initial GCS connection
#   timeout. Extremely tight at scale — workers fail to get cluster ID.
# gcs_rpc_server_reconnect_timeout_s (default 60s): max reconnection wait.
# gcs_server_request_timeout_seconds (default 60s): synchronous GCS requests.
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-120}"
export RAY_gcs_rpc_server_connect_timeout_s="${RAY_gcs_rpc_server_connect_timeout_s:-120}"
export RAY_gcs_rpc_server_reconnect_timeout_s="${RAY_gcs_rpc_server_reconnect_timeout_s:-120}"
export RAY_gcs_server_request_timeout_seconds="${RAY_gcs_server_request_timeout_seconds:-120}"

# --- Reduce background RPC noise during init ---
# raylet_report_resources_period_milliseconds (default 100ms): at many nodes
#   the resource-report RPCs/sec hitting GCS add up quickly. 500ms reduces
#   this 5x, freeing GCS capacity during the burst.
# task_events_report_interval_ms (default 1000ms): task status pushed to GCS
#   for dashboard observability. 5000ms reduces this 5x. Dashboard will show
#   slightly stale task status during init — acceptable tradeoff.
export RAY_raylet_report_resources_period_milliseconds="${RAY_raylet_report_resources_period_milliseconds:-500}"
export RAY_task_events_report_interval_ms="${RAY_task_events_report_interval_ms:-5000}"

# --- Resource broadcast batching ---
# gcs_resource_broadcast_max_batch_size (default 1 = disabled): GCS broadcasts
#   resource updates to all nodes via ray_syncer. With batch_size=1, every
#   update is sent individually — at N nodes reporting every 500ms, the
#   fan-out is O(N^2) messages/sec. Batching collapses these into fewer,
#   larger messages, dramatically reducing GCS send-side overhead.
# gcs_resource_broadcast_max_batch_delay_ms: max delay before flushing a
#   partial batch. 100ms is imperceptible for scheduling decisions.
export RAY_gcs_resource_broadcast_max_batch_size="${RAY_gcs_resource_broadcast_max_batch_size:-512}"
export RAY_gcs_resource_broadcast_max_batch_delay_ms="${RAY_gcs_resource_broadcast_max_batch_delay_ms:-100}"

# --- GCS reply-path threads ---
# gcs_server_rpc_client_thread_num (default max(1, num_cpus/4)): threads for
#   GCS to send gRPC replies back to raylets. Match to the server (inbound)
#   thread count to avoid a reply-path bottleneck during actor creation bursts.
export RAY_gcs_server_rpc_client_thread_num="${RAY_gcs_server_rpc_client_thread_num:-64}"

# --- Increase GCS active RPC headroom ---
# gcs_max_active_rpcs_per_handler (default rpc_server_thread_num * 100):
#   maximum concurrent RPCs per handler in the GCS. With rpc_thread_num=64
#   the auto-computed default is 6400. Explicitly set higher to ensure the
#   handler queue doesn't become a bottleneck when thousands of actors
#   register simultaneously without staggering.
export RAY_gcs_max_active_rpcs_per_handler="${RAY_gcs_max_active_rpcs_per_handler:-12800}"

# --- Subscriber cleanup ---
# subscriber_timeout_ms (default 300,000 = 5 min): time before a dead
#   subscriber is cleaned up from the GCS pub/sub publisher. At scale, dead
#   subscribers accumulate stale long-poll connections. 60s is aggressive but
#   safe — live subscribers reconnect well within this window.
export RAY_subscriber_timeout_ms="${RAY_subscriber_timeout_ms:-60000}"

# --- Cap idle task worker pool ---
# num_workers_soft_limit (default -1 = number of CPUs): on high-core-count nodes
#   the raylet accumulates many idle worker processes, each holding a GCS
#   connection. Cap at 16 — our scale-sim sub-servers are external uvicorn
#   processes, not Ray workers, so 16 is more than enough for any internal Ray
#   tasks ng_run spawns.
export RAY_num_workers_soft_limit="${RAY_num_workers_soft_limit:-16}"

# --- Reduce gRPC keepalive overhead ---
# grpc_keepalive_time_ms (default 10s): interval between keepalive pings to
#   the GCS server. With many sub-servers at 10s, keepalive alone generates
#   many pings/sec on GCS. 60s reduces this 6x during the init burst.
# grpc_keepalive_timeout_ms (default 20s): timeout for keepalive ACK.
#   Must be >= keepalive_time to avoid false dead-connection detection during
#   GCS load spikes.
export RAY_grpc_keepalive_time_ms="${RAY_grpc_keepalive_time_ms:-60000}"
export RAY_grpc_keepalive_timeout_ms="${RAY_grpc_keepalive_timeout_ms:-60000}"

# --- Disable observability features not in use ---
# enable_timeline (default true): every actor/task creation emits a timeline
#   event to GCS. Disabling eliminates thousands of GCS writes during init.
# event_stats (default true): internal event loop statistics collection.
#   Disabling reduces per-event overhead in the GCS event loop.
export RAY_enable_timeline="${RAY_enable_timeline:-false}"
export RAY_event_stats="${RAY_event_stats:-false}"

# --- Reduce syncer, heartbeat, and health-check background traffic ---
# ray_syncer_message_refresh_interval_ms (default 3s): periodic state refresh
#   across all nodes via ray_syncer. 10s reduces traffic 3x.
# core_worker_internal_heartbeat_ms (default 1s): worker heartbeat to raylet.
#   At many sub-servers heartbeating at 1s adds indirect GCS load via
#   raylet resource aggregation. 5s is sufficient.
# health_check_period_ms (default 3s): nodes health-check GCS every 3s.
#   At many sub-servers this aggregates fast. 10s is acceptable during init
#   where node-failure detection latency is not critical.
export RAY_ray_syncer_message_refresh_interval_ms="${RAY_ray_syncer_message_refresh_interval_ms:-10000}"
export RAY_core_worker_internal_heartbeat_ms="${RAY_core_worker_internal_heartbeat_ms:-5000}"
export RAY_health_check_period_ms="${RAY_health_check_period_ms:-10000}"

# Optional sanity print — set RAY_BURST_VERBOSE=1 to see what was applied.
if [[ "${RAY_BURST_VERBOSE:-0}" == "1" ]]; then
  echo "[ray-burst-env] applied GCS tuning:"
  env | grep '^RAY_' | sort | sed 's/^/  /'
fi
