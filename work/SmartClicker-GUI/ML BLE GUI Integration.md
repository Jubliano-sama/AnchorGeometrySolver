# ML BLE GUI Integration

This note is for a laptop GUI that talks directly to the ML clicker over BLE and triggers range-data captures for offset-model training.

## BLE Device And GATT Service

Flash the clicker with the `ml_clicker` firmware preset. It advertises as `IMEC ML Clicker` and exposes the existing gateway BLE packet service from the clicker role.

Use these 128-bit UUIDs:

| Item | UUID | Direction |
| --- | --- | --- |
| Service | `494d4543-0001-4757-8000-000000000001` | Discover this service |
| Packet TX | `494d4543-0001-4757-8000-000000000002` | Subscribe to notifications from clicker to laptop |
| Packet RX | `494d4543-0001-4757-8000-000000000003` | Write command bytes from laptop to clicker |
| Log TX | `494d4543-0001-4757-8000-000000000004` | Optional text-log notifications |

Enable notifications on `Packet TX` before sending commands. The firmware treats the BLE characteristic as a byte stream: a complete protocol frame is COBS encoded and terminated with a zero byte. Notifications may be split by MTU, so the GUI must buffer incoming bytes until it sees `0x00`, then decode one COBS frame.

## Binary Frame Format

Each BLE packet is:

```text
COBS(proto_packet) + 0x00
```

The decoded `proto_packet` is little-endian:

| Offset | Size | Field |
| --- | ---: | --- |
| 0 | 1 | Magic `0xC1` |
| 1 | 1 | Version `0x01` |
| 2 | 1 | Message type |
| 3 | 1 | Flags |
| 4 | 8 | Source device ID |
| 12 | 8 | Destination device ID |
| 20 | 4 | Session ID |
| 24 | 2 | Sequence |
| 26 | 1 | TTL |
| 27 | 1 | Payload length |
| 28 | 4 | Message age in ms |
| 32 | N | TLV payload |
| 32+N | 2 | CRC-16/CCITT-FALSE over all preceding decoded bytes |

The TLV payload is a sequence of:

```text
type:u8, length:u8, value:length bytes
```

All multi-byte integer TLV values are little-endian. Signed fields are encoded as two's-complement values of the documented width. CRC-16/CCITT-FALSE uses initial value `0xffff`, polynomial `0x1021`, no reflection, and no final XOR.

## Start Collection Command

Send one `MSG_COMMAND` packet to `Packet RX`.

Packet header:

| Field | Value |
| --- | --- |
| Message type | `MSG_COMMAND = 0x40` |
| Flags | `0` |
| Source device ID | Any stable non-zero host ID chosen by the GUI |
| Destination device ID | Clicker `DEVICE_ID`, `GATEWAY_ID`, or broadcast `0` |
| Session ID | GUI-chosen collection/session number |
| Sequence | GUI-chosen command sequence |
| TTL | `1` |

Payload TLVs:

| TLV | Type | Length | Required | Meaning |
| --- | ---: | ---: | --- | --- |
| `TLV_COMMAND_ID` | `0x10` | 2 | Yes | `CMD_ML_START_COLLECTION = 0x8000` or `CMD_ML_START_FAST_RANGING = 0x8001` |
| `TLV_SAMPLE_COUNT` | `0x0F` | 1 or 2 | No | Samples per selected anchor. Valid range is `1..100`; default build value is `8`. |
| `TLV_DISCOVERY_SLOT_COUNT` | `0x4C` | 1 or 2 | No | Fresh discovery reply slots and selected-anchor cap. Valid range is `1..8`; default build value is `8`. |

`CMD_ML_START_COLLECTION` is the full diagnostics / CIR mode. It preserves the existing behavior: fresh UWB discovery, scheduled range samples, and post-burst diagnostic/CIR packets.

`CMD_ML_START_FAST_RANGING` is range-data only. It uses the same packet framing and sample-count/discovery-slot TLVs, but intentionally emits only scheduled range sample reports plus the final command result. The firmware may skip fresh discovery when at least four anchors have actively ranged within the last 30 seconds; otherwise it falls back to fresh discovery automatically. The GUI does not decide whether the cache is valid.

## Collection Limits And Timeouts

There is one BLE-connected clicker. The selectable devices in this command path are anchors. `TLV_DISCOVERY_SLOT_COUNT` controls both the fresh UWB discovery reply slots and the maximum number of anchors selected into the schedule.

Current limits:

| Limit | Value |
| --- | ---: |
| Selected anchors per collection | `1..8`, capped by `TLV_DISCOVERY_SLOT_COUNT` |
| Samples per selected anchor | `1..100` |
| Maximum scheduled sample notifications | `8 * 100 = 800` |
| Scheduled sample stride | `33000 us` |
| Maximum scheduled UWB burst | about `26.4 s` |

Use the longer full-diagnostics command-result timeout for `CMD_ML_START_COLLECTION`; `75 s` is a practical GUI default because it covers discovery, post-burst diagnostics, BLE notification drain, and host scheduling jitter. Fast ranging can use a shorter timeout, but should still allow the maximum scheduled burst, discovery fallback, BLE drain, and host scheduling jitter.

For progress display, expect `selected_anchor_count * requested_samples_per_anchor` scheduled sample notifications when at least one anchor is found. Failed DS-TWR attempts are still emitted as sample rows with non-zero `TLV_RANGE_STATUS`, so they count toward progress and should be stored.

## Notifications To Parse

Subscribe to `Packet TX` and parse complete COBS-delimited frames. During a collection the clicker emits:

| Message | Meaning |
| --- | --- |
| `MSG_CLICK_REPORT = 0x20` with `FLAG_DIAGNOSTIC = 0x10` | One ML range sample notification. Store these as training rows. |
| `MSG_CLICK_REPORT = 0x20` with `FLAG_DIAGNOSTIC = 0x10` and no `TLV_SAMPLE_COUNT` | Full diagnostics mode only: one post-burst diagnostic packet for an anchor. Store these with the matching collection/anchor, not as training rows. |
| `MSG_COMMAND_RESULT = 0x41` with `FLAG_DIAGNOSTIC = 0x10` | Final command result for the trigger. `FLAG_ERROR = 0x40` is set when status is not OK. |

The GUI should not wait for the command result before collecting samples. The ML clicker keeps BLE packet and log notifications quiet while the UWB collection is active, buffers sample packets in firmware, then emits the buffered `MSG_CLICK_REPORT` frames in schedule order before the final command result.

For fast ranging, absence of post-burst diagnostic/CIR packets is expected and should not be reported as packet loss.

## ML Sample TLVs

Each ML sample notification is a diagnostic `MSG_CLICK_REPORT`. These packets are intentionally measurement-only; the large diagnostics are sent once per anchor after the scheduled samples finish. Important sample TLVs:

| TLV | Type | Width | Notes |
| --- | ---: | ---: | --- |
| `TLV_CLICKER_ID` | `0x0B` | 8 | Clicker ID |
| `TLV_ANCHOR_ID` | `0x0A` | 8 | Anchor ID |
| `TLV_EVENT_SEQ` | `0x06` | 4 | Collection event number generated by firmware |
| `TLV_TIMESTAMP_MS` | `0x07` | 8 | Sample timestamp in the firmware's local timestamp domain |
| `TLV_DISTANCE_MM` | `0x0C` | 4 signed | Primary range distance |
| `TLV_QUALITY` | `0x0D` | 1 | Range quality, `0..100` |
| `TLV_RANGE_STATUS` | `0x21` | 1 | `0` means `RANGE_OK`; non-zero values should still be stored as failed samples |
| `TLV_SAMPLE_INDEX` | `0x0E` | 2 | Index of this scheduled exchange in the collection |
| `TLV_SAMPLE_COUNT` | `0x0F` | 2 | Total scheduled sample slots for the collection |
| `TLV_DISTANCE_SAMPLES_MM` | `0x25` | 4 signed | Single-sample array for this notification |
| `TLV_RANGE_ROUND_INDICES` | `0x28` | 1 | Schedule round index |
| `TLV_SEQUENCE_START_TIMESTAMPS_MS` | `0x29` | 8 | Timestamp for the single sample slot |

## Post-Burst Diagnostic TLVs

After all scheduled samples for the collection complete, the clicker runs one extra diagnostic DS-TWR exchange per discovered anchor. These diagnostic packets are also `MSG_CLICK_REPORT | FLAG_DIAGNOSTIC`, but `TLV_SAMPLE_COUNT` is absent. The GUI should attach them to the collection with `TLV_EVENT_SEQ` and to the anchor with `TLV_ANCHOR_ID`.

Every post-burst diagnostic packet includes:

| TLV | Type | Width | Notes |
| --- | ---: | ---: | --- |
| `TLV_CLICKER_ID` | `0x0B` | 8 | Clicker ID |
| `TLV_ANCHOR_ID` | `0x0A` | 8 | Anchor ID |
| `TLV_EVENT_SEQ` | `0x06` | 4 | Collection event number |
| `TLV_TIMESTAMP_MS` | `0x07` | 8 | Firmware local timestamp for the diagnostic exchange |
| `TLV_DISTANCE_MM` | `0x0C` | 4 signed | Diagnostic exchange range result |
| `TLV_QUALITY` | `0x0D` | 1 | Diagnostic exchange quality |
| `TLV_RANGE_STATUS` | `0x21` | 1 | Diagnostic exchange status |
| `TLV_BURST_ID` | `0x34` | 4 | Burst identity |
| `TLV_EXCHANGE_STRIDE_US` | `0x35` | 2 | Scheduled sample stride |
| `TLV_BURST_DURATION_MS` | `0x36` | 2 | Scheduled responder burst duration |
| `TLV_DIAG_SOURCE` | `0x57` | 1 | `0=summary`, `1=clicker response RX`, `2=anchor final RX`, `3=anchor CIR window` |

The first post-burst packet for an anchor is the diagnostic summary. It may include:

| TLV | Type | Width | Notes |
| --- | ---: | ---: | --- |
| `TLV_UWB_RSL_DBM` | `0x24` | 1 signed | Anchor-side RX signal level when present |
| `TLV_UWB_CIR_SAMPLE` | `0x26` | 6 | Compact anchor CIR sample when present |
| `TLV_DIAG_STATUS_FLAGS` | `0x33` | 4 | Bitfield for clicker/anchor diagnostics present/missing/truncated |
| `TLV_DIAG_BYTES_CAPTURED` | `0x39` | 4 | Captured diagnostic bytes before BLE packetization |
| `TLV_DIAG_BYTES_TRANSMITTED` | `0x3A` | 4 | Diagnostic bytes sent over BLE |
| `TLV_DIAG_BYTES_TRUNCATED` | `0x3B` | 4 | Captured bytes not transmitted |
| `TLV_DIAG_FRAMES_DROPPED` | `0x3C` | 4 | Diagnostic frames dropped by firmware |
| `TLV_REPORT_FRAGMENT_COUNT` | `0x3D` | 2 | Summary packet fragment count, currently `1` |
| `TLV_PHY_CONFIG_ID` | `0x42` | 1 | UWB channel/PHY config identifier |
| `TLV_UWB_CLOCK_OFFSET_RAW` | `0x4D` | 2 signed | Raw DWM3000 clock-offset sample when present |
| `TLV_UWB_CARRIER_INTEGRATOR` | `0x4E` | 4 signed | Raw DWM3000 carrier-integrator sample when present |
| `TLV_UWB_RAW_TIMESTAMPS` | `0x53` | 24 | Six little-endian `u32` values: `poll_tx`, `poll_rx`, `resp_tx`, `resp_rx`, `final_tx`, `final_rx` |
| `TLV_CLICKER_DIAG_BYTES` | `0x40` | variable | Compact clicker diagnostic bytes when present |
| `TLV_ANCHOR_DIAG_BYTES` | `0x41` | variable | Compact anchor diagnostic bytes when present |

Raw RX diagnostic blocks use `TLV_DIAG_SOURCE = 1` for the clicker response RX and `TLV_DIAG_SOURCE = 2` for the anchor final RX. The raw block is carried in `TLV_UWB_RX_DIAG_BYTES = 0x54` and is currently 108 bytes when present. The fixed byte order is the firmware's packed `dwt_rxdiag_t`: `ipatovRxTime`, `ipatovRxStatus`, `ipatovPOA`, `stsRxTime`, `stsRxStatus`, `stsPOA`, `sts2RxTime`, `sts2RxStatus`, `sts2POA`, `tdoa`, `pdoa`, `xtalOffset`, `ciaDiag1`, `ipatovPeak`, `ipatovPower`, `ipatovF1`, `ipatovF2`, `ipatovF3`, `ipatovFpIndex`, `ipatovAccumCount`, `stsPeak`, `stsPower`, `stsF1`, `stsF2`, `stsF3`, `stsFpIndex`, `stsAccumCount`, `sts2Peak`, `sts2Power`, `sts2F1`, `sts2F2`, `sts2F3`, `sts2FpIndex`, `sts2AccumCount`.

CIR is sent as one or more packets with `TLV_DIAG_SOURCE = 3`. The firmware captures a first-path-centered window, not the complete 2048-sample accumulator: 192 complex samples, currently 64 samples before the integer first-path index and 128 samples after it, clamped to the DWM3000 accumulator edges. This is 1152 bytes per anchor because each complex sample is 6 bytes: signed 24-bit real followed by signed 24-bit imaginary. Eight anchor windows therefore fit in 9216 bytes. Each packet carries:

| TLV | Type | Width | Notes |
| --- | ---: | ---: | --- |
| `TLV_DIAG_FRAGMENT_INDEX` | `0x55` | 2 | Zero-based fragment index |
| `TLV_DIAG_FRAGMENT_COUNT` | `0x56` | 2 | Number of fragments for this CIR block |
| `TLV_UWB_CIR_BYTE_OFFSET` | `0x50` | 2 | Byte offset of this chunk in the CIR byte array |
| `TLV_UWB_CIR_TOTAL_BYTES` | `0x51` | 2 | Total CIR window bytes captured by the anchor, currently `1152` |
| `TLV_UWB_CIR_FIRST_PATH_INDEX` | `0x52` | 2 | First-path index from DWM3000 diagnostics |
| `TLV_UWB_CIR_START_INDEX` | `0x58` | 2 | Absolute accumulator sample index of the first sample in this CIR window |
| `TLV_UWB_CIR_FULL_CHUNK` | `0x4F` | variable | Raw CIR chunk bytes |

Reconstruct CIR by grouping packets on `(event_seq, anchor_id, burst_id, cir_start_index, cir_first_path_index, cir_total_bytes)`, sorting by `TLV_UWB_CIR_BYTE_OFFSET`, and concatenating `TLV_UWB_CIR_FULL_CHUNK`. Sample `n` in the concatenated byte array corresponds to absolute accumulator sample `cir_start_index + n`. Preserve partial groups; they are still useful for debugging packet loss.

## CIR Processing Note

For the GUI/data logger, keep the CIR as raw I/Q data. Each complex sample is exactly 6 bytes:

```text
real:i24 little-endian, imaginary:i24 little-endian
```

Decode both fields as signed 24-bit integers and store them as separate `int32` columns. Do not store only magnitude; magnitude and phase are derived features and can be regenerated later.

Recommended saved CIR table, one row per complex sample:

| Column | Meaning |
| --- | --- |
| `event_seq` | Collection event number |
| `anchor_id` | Anchor that captured this CIR window |
| `burst_id` | Diagnostic burst identity |
| `cir_start_index` | Absolute accumulator index for local sample 0 |
| `cir_first_path_index` | DWM3000 first-path index |
| `cir_sample_index` | Local sample index in the saved window, `0..255` |
| `cir_abs_sample_index` | `cir_start_index + cir_sample_index` |
| `cir_real` | Signed 24-bit real component, saved as `int32` |
| `cir_imag` | Signed 24-bit imaginary component, saved as `int32` |

Use a columnar format such as Parquet for larger runs. Keep the range samples in a separate table keyed by `(event_seq, anchor_id)`, and join CIR later by the same keys plus `burst_id` when needed. If a CIR block is incomplete, save the rows that arrived and mark the block as partial in metadata instead of dropping it.

For visualization, compute derived arrays from the stored columns:

```text
magnitude = sqrt(cir_real^2 + cir_imag^2)
power_db = 10 * log10(max(cir_real^2 + cir_imag^2, 1))
phase = atan2(cir_imag, cir_real)
```

Plot `magnitude` or `power_db` against `cir_abs_sample_index`, and draw a vertical marker at `cir_first_path_index`. A second plot of `cir_real` and `cir_imag` against the same x-axis is useful when checking sign-extension or byte-order bugs.

Store the whole decoded packet and every TLV you recognize. Unknown TLVs should be preserved or ignored without failing the row; the firmware may add diagnostics later.

## Command Result TLVs

The final command result has:

| TLV | Type | Meaning |
| --- | ---: | --- |
| `TLV_COMMAND_ID` | `0x10` | `0x8000` |
| `TLV_COMMAND_STATUS` | `0x11` | `0=OK`, `1=unsupported`, `2=malformed`, `3=busy`, `4=denied`, `5=timeout`, `6=radio error`, `7=invalid state`, `8=internal error` |
| `TLV_REASON` | `0x1E` | Extra reason byte, currently `0` for this path |
| `TLV_EVENT_SEQ` | `0x06` | Present when a collection started |
| `TLV_SAMPLE_COUNT` | `0x0F` | Number of sample notifications emitted |

Match results to commands with packet `session_id` and `seq`. Match sample rows to a collection with `TLV_EVENT_SEQ`.

If no anchors reply during fresh UWB discovery, the clicker emits no `MSG_CLICK_REPORT` samples and then sends a `MSG_COMMAND_RESULT | FLAG_DIAGNOSTIC | FLAG_ERROR` with `TLV_COMMAND_STATUS = COMMAND_TIMEOUT` and `TLV_SAMPLE_COUNT = 0`.

If at least one anchor replies but fewer than the configured discovery slot count reply, the command continues with the discovered anchors. The schedule uses `min(discovered anchors, TLV_DISCOVERY_SLOT_COUNT or CONFIG_IMEC_ML_DISCOVERY_SLOT_COUNT)` and requests the configured samples per selected anchor. In that case the final status is `COMMAND_OK` when the scheduled exchanges complete and all sample notifications are delivered; `TLV_SAMPLE_COUNT` is the actual number of `MSG_CLICK_REPORT` notifications emitted.

## Practical GUI Behavior

1. Scan for `IMEC ML Clicker`, connect, discover the service, and subscribe to `Packet TX`.
2. Send one `CMD_ML_START_COLLECTION` command for full diagnostics / CIR, or `CMD_ML_START_FAST_RANGING` for range-data-only collection, with the desired sample count.
3. Buffer and decode every `Packet TX` notification until the matching command result arrives. Samples may arrive as a post-UWB burst rather than one-by-one during ranging.
4. Save every `MSG_CLICK_REPORT | FLAG_DIAGNOSTIC` packet. Rows with `TLV_SAMPLE_COUNT` are scheduled training samples, including failed rows with nonzero `TLV_RANGE_STATUS`; rows without it are post-burst diagnostics and are expected only in full diagnostics mode.
5. Treat `COMMAND_BUSY` as a retryable state; wait for the prior collection to finish before sending another command.
6. Keep the BLE connection open between captures. The GUI does not manage anchor identity or cache discovery results; fast ranging either uses the firmware's recent ranging cache or falls back to fresh discovery.

Source-of-truth implementation files are `firmware/include/protocol.h`, `firmware/include/report.h`, `firmware/src/serial_frame.c`, and `firmware/app/src/main.c`.
