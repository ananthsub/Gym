#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ray GCS burst-tolerance tuning for ng_run at high sub-server fan-out (~128+).
#
# When `nemo_gym/cli.py:RunHelper.start` fans out 2N+1 child Popens in parallel
# during the spawn loop, each child's `ray.init(address=…)` produces a
# RegisterWorker RPC against the head's single GCS. At high N this storm
# overwhelms GCS: the queue backs up, client connects time out at the default
# 5 s ceiling, raylet kills the unregistered workers, and ng_run never reaches
# "All N/N servers ready".
#
# This is a workaround, not a fix. It increases GCS throughput, extends timeouts
# to absorb the burst, and reduces background RPC noise so the burst has more
# room. The architectural fix (a staggered spawn loop in `RunHelper.start`) is
# tracked separately and is not on this critical path.
#
# Source this file from the Slurm launcher so the env propagates into the
# compute-node environment (sbatch `--export=ALL`), and from any driver wrapper
# that exec's into a long-lived shell on a compute node. Every
# `export RAY_xxx="${RAY_xxx:-…}"` is guarded so explicit per-run overrides win.
#
# At low fan-out (the default reference suite tops out at n=32 agents) this is a
# harmless no-op — the tuned values only matter once GCS is actually contended.
#
# Config reference: https://github.com/ray-project/ray/blob/master/src/ray/common/ray_config_def.h

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
#   for dashboard observability. 5000ms reduces this 5x.
export RAY_raylet_report_resources_period_milliseconds="${RAY_raylet_report_resources_period_milliseconds:-500}"
export RAY_task_events_report_interval_ms="${RAY_task_events_report_interval_ms:-5000}"

# --- Resource broadcast batching ---
# gcs_resource_broadcast_max_batch_size (default 1 = disabled): GCS broadcasts
#   resource updates to all nodes via ray_syncer. With batch_size=1, every
#   update is sent individually — batching collapses these into fewer, larger
#   messages, reducing GCS send-side overhead.
# gcs_resource_broadcast_max_batch_delay_ms: max delay before flushing a partial
#   batch. 100ms is imperceptible for scheduling decisions.
export RAY_gcs_resource_broadcast_max_batch_size="${RAY_gcs_resource_broadcast_max_batch_size:-512}"
export RAY_gcs_resource_broadcast_max_batch_delay_ms="${RAY_gcs_resource_broadcast_max_batch_delay_ms:-100}"

# --- GCS reply-path threads ---
# gcs_server_rpc_client_thread_num (default max(1, num_cpus/4)): threads for GCS
#   to send gRPC replies back to raylets. Match the server (inbound) thread
#   count to avoid a reply-path bottleneck during actor creation bursts.
export RAY_gcs_server_rpc_client_thread_num="${RAY_gcs_server_rpc_client_thread_num:-64}"

# --- Increase GCS active RPC headroom ---
# gcs_max_active_rpcs_per_handler (default rpc_server_thread_num * 100): maximum
#   concurrent RPCs per handler in the GCS. Set higher so the handler queue does
#   not become a bottleneck when thousands of actors register simultaneously.
export RAY_gcs_max_active_rpcs_per_handler="${RAY_gcs_max_active_rpcs_per_handler:-12800}"

# --- Subscriber cleanup ---
# subscriber_timeout_ms (default 300,000 = 5 min): time before a dead subscriber
#   is cleaned up from the GCS pub/sub publisher. 60s is aggressive but safe —
#   live subscribers reconnect well within this window.
export RAY_subscriber_timeout_ms="${RAY_subscriber_timeout_ms:-60000}"

# --- Cap idle task worker pool ---
# num_workers_soft_limit (default -1 = number of CPUs): on high-core-count nodes
#   the raylet accumulates many idle worker processes, each holding a GCS
#   connection. Our scale-sim sub-servers are external uvicorn processes, not
#   Ray workers, so 16 is more than enough for any internal Ray tasks ng_run
#   spawns.
export RAY_num_workers_soft_limit="${RAY_num_workers_soft_limit:-16}"

# --- Reduce gRPC keepalive overhead ---
# grpc_keepalive_time_ms (default 10s): interval between keepalive pings to the
#   GCS server. 60s reduces this 6x during the init burst.
# grpc_keepalive_timeout_ms (default 20s): timeout for keepalive ACK. Must be
#   >= keepalive_time to avoid false dead-connection detection under GCS load.
export RAY_grpc_keepalive_time_ms="${RAY_grpc_keepalive_time_ms:-60000}"
export RAY_grpc_keepalive_timeout_ms="${RAY_grpc_keepalive_timeout_ms:-60000}"

# --- Disable observability features not in use ---
# enable_timeline (default true): every actor/task creation emits a timeline
#   event to GCS. Disabling eliminates many GCS writes during init.
# event_stats (default true): internal event loop statistics collection.
export RAY_enable_timeline="${RAY_enable_timeline:-false}"
export RAY_event_stats="${RAY_event_stats:-false}"
