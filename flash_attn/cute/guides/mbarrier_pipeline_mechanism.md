# Mbarrier Pipeline Mechanism

## Overview

The mbarrier pipeline uses hardware memory barriers to synchronize producer-consumer data transfers in a multi-stage circular buffer. Each stage has its own mbarrier object that tracks arrival count and transaction count, with automatic phase transitions.

## Hardware Mbarrier Behavior

### Phase Completion

An mbarrier completes its current phase when **both** conditions are met:
1. **Arrival count reaches zero** (all expected threads/operations have arrived)
2. **Transaction count reaches zero** (all expected bytes have been transferred, for TMA operations)

### Automatic Phase Transition

When a phase completes, the hardware **atomically**:
1. Transitions the mbarrier to the next phase (phase parity flips: 0→1 or 1→0)
2. **Reinitializes the arrival count** to the expected arrival count (set during `mbarrier.init`)

**Important**: The phase transition and arrival count reset happen automatically upon phase completion. Software does not explicitly reset the count.

### Phase Tracking Requirement

Software must track the phase parity (0 or 1) for each barrier. If the last round's phase is odd and you call `mbarrier.try_wait.parity` with an **old phase**, it will return `True` immediately (because that phase already completed). You must use the **current phase**(even) to wait for the current.

## Pipeline Architecture

### Barrier Array Structure

- **`num_stages`** mbarriers are initialized, one per pipeline stage
- Each barrier is initialized with an **arrival count** (typically the cooperative group size)
- Each barrier maintains independent phase state

### Circular Buffer Pattern

```
Barrier Array: [barrier[0], barrier[1], ..., barrier[num_stages-1]]
                ↑                                    ↑
              wrap back to 0 after num_stages iterations
```

The pipeline cycles through barriers using a software `index` (0 to num_stages-1). When the index wraps, the software `phase` bit flips to track the new hardware phase.

### Phase Progression Example

Here's how the phase progresses through a pipeline with `num_stages=3`:

```python
# Cycle 1: phase=0
wait(barrier[0], phase=0)      # Wait for phase 0 to complete
# ... use data ...
release(barrier[0])            # Signal empty, phase still 0
advance()                      # index=1, phase=0

wait(barrier[1], phase=0)      # Wait for phase 0 to complete
# ... use data ...
release(barrier[1])            # Signal empty, phase still 0
advance()                      # index=2, phase=0

wait(barrier[2], phase=0)      # Wait for phase 0 to complete
# ... use data ...
release(barrier[2])            # Signal empty, phase still 0
advance()                      # index wraps to 0, phase flips to 1

# Cycle 2: phase=1
wait(barrier[0], phase=1)      # Hardware resets arrival count for barrier[0] at phase 1
                               # Now waiting on the SAME barrier[0] but with NEW phase=1
# ... use data ...
release(barrier[0])            # Signal empty, phase still 1
advance()                      # index=1, phase=1

# ... and so on, cycling through barriers with alternating phases
```

**Key Points**:
- Each barrier maintains its own phase state in hardware
- Software phase flips when the index wraps (after `num_stages` iterations)
- When you return to a barrier with a new phase, the hardware has automatically reset its arrival count for that phase
- The `release()` operation doesn't change the phase - it only signals arrival on the current phase

## Pipeline Methods and Hardware Instructions

### Producer Side

#### `producer_acquire(state)`
- **Maps to**: `mbarrier_wait` on the **empty barrier** (`sync_object_empty`)
- **Purpose**: Wait for buffer to be empty before writing
- **Hardware**: `mbarrier.try_wait.parity` with `state.index` and `state.phase`
- **Location**: `sm90.py:220-223` (base) or `sm100.py:188-191` (TMA variant)

#### `producer_commit(state)`
- **For AsyncThread producers**: Maps to `mbarrier_arrive` on the **full barrier** (`sync_object_full`)
- **For TMA producers**: **No-op** - the TMA copy instruction itself updates the transaction count
- **Hardware**: 
  - AsyncThread: `mbarrier.arrive` 
  - TMA: Transaction count is set by `mbarrier_arrive_and_expect_tx` (called separately, see below)
- **Location**: `sm90.py:221-222` (AsyncThread) or `sm100.py:187-191` (TMA no-op)

**Note**: For TMA operations, `mbarrier_arrive_and_expect_tx` is called directly (not through `producer_commit`):
```python
# flash_fwd_sm100_fp4.py:3092-3094
cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, self.tma_copy_bytes[K_or_V])
cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)
```

### Consumer Side

#### `consumer_wait(state)`
- **Maps to**: `mbarrier_wait` on the **full barrier** (`sync_object_full`)
- **Purpose**: Wait for buffer to be full (data ready) before reading
- **Hardware**: `mbarrier.try_wait.parity` with `state.index` and `state.phase`
- **Location**: `sm90.py:224-230`

#### `consumer_release(state)`
- **For AsyncThread consumers**: Maps to `mbarrier_arrive` on the **empty barrier** (`sync_object_empty`)
- **For UMMA consumers** (e.g., `PipelineTmaUmma`): Maps to `tcgen05.commit` on the **empty barrier**
- **Purpose**: Signal that buffer is empty after reading
- **Hardware**:
  - AsyncThread: `mbarrier.arrive`
  - UMMA: `tcgen05.commit` (which internally performs `mbarrier.arrive`)
- **Location**: 
  - `sm90.py:235-236` (AsyncThread)
  - `sm100.py:176-180` (UMMA with `tcgen05.commit`)

**Key Point**: `tcgen05.commit` is equivalent to `mbarrier_arrive` for UMMA operations. It signals that the consumer has finished using the buffer.

### Arrive Operations Mapping

The `arrive()` method in `MbarrierArray` routes to different hardware instructions based on the pipeline operation type:

- **`PipelineOp.AsyncThread`**: `mbarrier_arrive` (helpers.py:238-242)
- **`PipelineOp.TCGen05Mma`**: `tcgen05.commit` (helpers.py:247-255)
- **`PipelineOp.TmaLoad`**: `mbarrier_arrive_and_expect_tx` (helpers.py:257-259)
- **`PipelineOp.AsyncLoad`**: `cp_async_mbarrier_arrive_noinc` (helpers.py:244-245)

## Software State Management

### PipelineState

The `PipelineState` object tracks:
- **`index`**: Current barrier index (0 to num_stages-1)
- **`phase`**: Current phase parity (0 or 1)
- **`count`**: Total number of iterations

### `advance()` Method

```python
def advance(self):
    self._index += 1
    self._count += 1
    # Wrap index and flip phase when reaching num_stages
    if self._index == self.stages:
        self._index = Int32(0)
        self._phase = phase ^ 1
```

**Important**: `advance()` only updates **software state**. It does not reset hardware barriers. The hardware automatically resets arrival counts when phases complete.

## Typical Workflow

### Producer Loop (TMA Example)

```python
for block in range(num_blocks):
    # 1. Wait for buffer to be empty
    pipeline.producer_acquire(producer_state)
    
    # 2. Set transaction count and issue TMA copy
    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + stage, tx_bytes)
    cute.copy(tma_atom, gmem, smem, tma_bar_ptr=mbar_ptr + stage)
    
    # 3. Advance to next stage
    producer_state.advance()
```

### Consumer Loop (UMMA Example)

```python
for block in range(num_blocks):
    # 1. Wait for buffer to be full
    pipeline.consumer_wait(consumer_state)
    
    # 2. Use data (e.g., perform MMA)
    # ... compute with data in smem[consumer_state.index] ...
    
    # 3. Signal buffer is empty (via tcgen05.commit)
    pipeline.consumer_release(consumer_state)
    
    # 4. Advance to next stage
    consumer_state.advance()
```

## Phase Synchronization

### Why Phase Tracking is Critical

1. **Hardware automatically transitions phase** when completion conditions are met
2. **Software must track phase** to use correct parity in `wait()` calls
3. **Using old phase** in `wait()` returns immediately (phase already completed)
4. **Using current phase** waits for current or next completion

### Phase Flip Timing

- Phase flips happen **automatically** when a barrier's phase completes
- Software phase flips (via `advance()`) happen when **wrapping the index**
- These should align: when you wrap back to barrier[0], the hardware phase should have also flipped

## Summary Table

| Pipeline Method | Hardware Instruction | Barrier Type | Purpose |
|----------------|---------------------|--------------|---------|
| `producer_acquire()` | `mbarrier_wait` | empty | Wait for empty buffer |
| `producer_commit()` (AsyncThread) | `mbarrier_arrive` | full | Signal buffer full |
| `producer_commit()` (TMA) | no-op | - | TMA handles transaction count |
| `consumer_wait()` | `mbarrier_wait` | full | Wait for full buffer |
| `consumer_release()` (AsyncThread) | `mbarrier_arrive` | empty | Signal buffer empty |
| `consumer_release()` (UMMA) | `tcgen05.commit` → `mbarrier_arrive` | empty | Signal buffer empty |

## Key Takeaways

1. **Phase completion is automatic** - hardware resets arrival count when phase completes
2. **Software tracks phase** - must use correct phase in `wait()` calls
3. **`tcgen05.commit` = `mbarrier_arrive`** - for UMMA consumers
4. **TMA uses `mbarrier_arrive_and_expect_tx`** - sets both arrival and transaction count
5. **`advance()` updates software state only** - hardware resets happen automatically

