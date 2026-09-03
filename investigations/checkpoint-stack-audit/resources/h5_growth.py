# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""H5: _locks and _tombstones are never pruned."""

import asyncio
import tracemalloc

from nemo_gym.checkpoint.resources import ResourcesCheckpointParticipant, ResourceSnapshot


async def main():
    async def export(r, a):
        return {}

    async def restore(s):
        pass

    p = ResourcesCheckpointParticipant(export_state=export, restore_states=restore)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    n = 100_000
    for i in range(n):
        rid = f"rollout-{i:06d}"
        p.lock_for(rid, 0)
        p.register(rid, 0)
        p.record_mutation(rid, 0)
        p.retire(rid, 0)  # what /verify success does
    after_retire = tracemalloc.get_traced_memory()[0]
    print(f"after {n} register+retire: _revisions={len(p._revisions)} _locks={len(p._locks)} "
          f"heap +{(after_retire - base) / 1e6:.1f} MB")
    snaps = [ResourceSnapshot(rollout_id=f"rollout-{i:06d}", attempt_index=0, state_revision=1, state={}) for i in range(n)]
    await p.restore(snaps)
    after_restore = tracemalloc.get_traced_memory()[0]
    print(f"after restore of {n}: _tombstones={len(p._tombstones)} _locks={len(p._locks)} _revisions={len(p._revisions)} "
          f"heap +{(after_restore - after_retire) / 1e6:.1f} MB")
    for i in range(n):
        p.retire(f"rollout-{i:06d}", 1)
    print(f"after retiring all N+1: _tombstones={len(p._tombstones)} _locks={len(p._locks)} _revisions={len(p._revisions)}")


asyncio.run(main())
