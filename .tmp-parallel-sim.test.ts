// Zero-token test of the parallel-compaction algorithm.
//
// Instead of calling the API, this stubs runForkedAgent (via bun:test's
// mock.module) with a fake worker that SLEEPS according to a latency model:
//
//   latency = TTFT + input_tokens/PREFILL_RATE + output_tokens/DECODE_RATE
//
// Everything else is the REAL production code path — generateCompactSummaryParallel
// → partitionMessagesIntoBlocks → dispatchParallelCompaction (Promise.all over
// dispatchBlockSummary) → mergeBlockSummaries. So this validates the actual
// orchestration, not a reimplementation of it.
//
// What this CAN prove (exactly, no assumptions):
//   - the N dispatches genuinely overlap (wall ≈ slowest block, not the sum)
//   - every message lands in exactly one target block (no gaps/duplicates)
//   - the input-token amplification factor from re-sending the growing
//     shared prefix in every block's request
//
// What this CANNOT prove (needs the real API):
//   - the absolute latency constants below (TTFT/decode rate are estimates)
//   - real server-side behavior: concurrency throttling, queuing, and
//     whether cross-block prompt-cache hits actually land
// The RATIO results are robust to the constants — the decode-rate sweep at
// the bottom demonstrates that explicitly.

import { beforeAll, describe, expect, mock, test } from 'bun:test'
import { roughTokenCountEstimationForMessages } from './src/services/tokenEstimation.js'

// ---------------------------------------------------------------------------
// Latency model (assumptions — tweak freely; conclusions are ratio-based)
// ---------------------------------------------------------------------------
const TTFT_MS = 500 // time-to-first-token, per request
const PREFILL_TOKENS_PER_SEC = 8_000 // input processing: fast, parallel
const DECODE_TOKENS_PER_SEC = 60 // output generation: slow, autoregressive

// Real sleeps are scaled down so the suite finishes in seconds. Concurrency
// is still genuinely exercised; reported "modeled" times divide this out.
const TIME_SCALE = 0.02

// Output-size assumptions, mirroring how the two paths are configured in
// compact.ts / the timing script: one holistic summary vs. N smaller ones.
const SEQUENTIAL_OUTPUT_TOKENS = 2_000
const PARALLEL_OUTPUT_TOKENS_PER_BLOCK = 700

// Mutable so a scenario can switch output-budget policy: fixed-per-block
// (summary grows with N) vs. fixed-total (per-block = total/N).
let currentPerBlockOutput = PARALLEL_OUTPUT_TOKENS_PER_BLOCK

function modelLatencyMs(inputTokens: number, outputTokens: number): number {
  return (
    TTFT_MS +
    (inputTokens / PREFILL_TOKENS_PER_SEC) * 1000 +
    (outputTokens / DECODE_TOKENS_PER_SEC) * 1000
  )
}

// ---------------------------------------------------------------------------
// Fake worker: records what each block was sent, sleeps the modeled duration
// ---------------------------------------------------------------------------
type CallRecord = {
  forkLabel: string
  inputTokens: number
  outputTokens: number
  modeledMs: number
  startedAt: number
  finishedAt: number
}
const callLog: CallRecord[] = []

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

mock.module('./src/utils/forkedAgent.js', () => ({
  runForkedAgent: async (params: any) => {
    const inputTokens = roughTokenCountEstimationForMessages([
      ...params.cacheSafeParams.forkContextMessages,
      ...params.promptMessages,
    ])
    const outputTokens = currentPerBlockOutput
    const modeledMs = modelLatencyMs(inputTokens, outputTokens)

    const startedAt = performance.now()
    await sleep(modeledMs * TIME_SCALE)
    const finishedAt = performance.now()

    callLog.push({
      forkLabel: params.forkLabel,
      inputTokens,
      outputTokens,
      modeledMs,
      startedAt,
      finishedAt,
    })

    // Body sized to the modeled output so merge sees realistic text.
    const body = 'summary sentence. '.repeat(Math.round((outputTokens * 4) / 18))
    return {
      messages: [
        {
          type: 'assistant',
          uuid: `resp-${params.forkLabel}`,
          timestamp: new Date().toISOString(),
          isApiErrorMessage: false,
          message: {
            role: 'assistant',
            id: `msg-${params.forkLabel}`,
            model: 'claude-sonnet-4-6',
            content: [
              { type: 'text', text: `<summary>\n${body}\n</summary>` },
            ],
            usage: {
              input_tokens: inputTokens,
              output_tokens: outputTokens,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
          },
        },
      ],
      totalUsage: {
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
    }
  },
}))

// Imported AFTER the mock is registered so it picks up the stub.
let parallel: typeof import('./src/services/compact/parallelCompact.js')
beforeAll(async () => {
  parallel = await import('./src/services/compact/parallelCompact.js')
})

// ---------------------------------------------------------------------------
// Synthetic conversation builder
// ---------------------------------------------------------------------------
let uuidCounter = 0
const uuid = () => `uuid-${++uuidCounter}`

function userMsg(text: string): any {
  return {
    type: 'user',
    uuid: uuid(),
    timestamp: new Date().toISOString(),
    message: { role: 'user', content: text },
  }
}
function assistantToolUse(id: string, msgId: string): any {
  return {
    type: 'assistant',
    uuid: uuid(),
    timestamp: new Date().toISOString(),
    message: {
      role: 'assistant',
      id: msgId,
      content: [
        { type: 'tool_use', id, name: 'Read', input: { file_path: '/src/x.ts' } },
      ],
    },
  }
}
function userToolResult(id: string, content: string): any {
  return {
    type: 'user',
    uuid: uuid(),
    timestamp: new Date().toISOString(),
    message: {
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: id, content }],
    },
  }
}

/** One API round ≈ `roundTokens` tokens (4 chars/token). */
function buildConversation(numRounds: number, roundTokens: number): any[] {
  const body = 'x'.repeat(roundTokens * 4)
  const messages: any[] = [userMsg('Fix the race condition and add the panel.')]
  for (let i = 0; i < numRounds; i++) {
    const id = `tool_${i}`
    messages.push(assistantToolUse(id, `resp_${i}`))
    messages.push(userToolResult(id, body))
    messages.push(userMsg(`round ${i} looks good, continue`))
  }
  return messages
}

const fmt = (ms: number) => `${(ms / 1000).toFixed(2)}s`

// ---------------------------------------------------------------------------
describe('parallel compaction — zero-token algorithm test', () => {
  test('dispatches concurrently: wall ≈ slowest block, not the sum', async () => {
    callLog.length = 0
    const conversation = buildConversation(20, 2_500) // ~50K tokens
    const blockSize = 10_000

    const t0 = performance.now()
    const result = await parallel.generateCompactSummaryParallel(
      conversation,
      { abortController: new AbortController() } as any,
      { forkContextMessages: conversation } as any,
      undefined,
      blockSize,
    )
    const measuredWallMs = performance.now() - t0

    const modeledPerBlock = callLog.map(c => c.modeledMs)
    const modeledCriticalPath = Math.max(...modeledPerBlock)
    const modeledIfSerial = modeledPerBlock.reduce((a, b) => a + b, 0)
    const measuredModeled = measuredWallMs / TIME_SCALE

    console.log(`\n=== CONCURRENCY CHECK (${result.blockCount} blocks, B=${blockSize}) ===`)
    console.log(`  per-block modeled latency: ${modeledPerBlock.map(fmt).join(', ')}`)
    console.log(`  if run serially:      ${fmt(modeledIfSerial)}`)
    console.log(`  critical path (max):  ${fmt(modeledCriticalPath)}`)
    console.log(`  measured wall:        ${fmt(measuredModeled)} (modeled-equivalent)`)

    expect(result.blockCount).toBeGreaterThan(1)
    // The real proof of overlap: wall time tracks the slowest single block,
    // nowhere near the serial sum.
    expect(measuredModeled).toBeLessThan(modeledIfSerial * 0.75)
    expect(measuredModeled).toBeGreaterThan(modeledCriticalPath * 0.5)

    // All N requests must be in flight at the same instant.
    const latestStart = Math.max(...callLog.map(c => c.startedAt))
    const earliestFinish = Math.min(...callLog.map(c => c.finishedAt))
    expect(latestStart).toBeLessThan(earliestFinish)
  })

  test('every message lands in exactly one target block', () => {
    const conversation = buildConversation(12, 2_000)
    const blocks = parallel.partitionMessagesIntoBlocks(conversation, 8_000)

    const flat = blocks.flat()
    expect(flat.length).toBe(conversation.length)
    expect(flat.map(m => m.uuid)).toEqual(conversation.map(m => m.uuid))

    // No tool_use separated from its tool_result (would be an API 400).
    for (const block of blocks) {
      const uses = new Set<string>()
      const results = new Set<string>()
      for (const m of block as any[]) {
        const content = m.message.content
        if (!Array.isArray(content)) continue
        for (const b of content) {
          if (b.type === 'tool_use') uses.add(b.id)
          if (b.type === 'tool_result') results.add(b.tool_use_id)
        }
      }
      for (const id of uses) expect(results.has(id)).toBe(true)
    }
  })

  test('speedup vs. token cost across conversation sizes', async () => {
    const blockSize = 20_000
    const scenarios = [
      { rounds: 8, roundTokens: 2_500, label: '~20K' },
      { rounds: 20, roundTokens: 2_500, label: '~50K' },
      { rounds: 40, roundTokens: 2_500, label: '~100K' },
      { rounds: 80, roundTokens: 2_500, label: '~200K' },
    ]

    console.log(`\n=== SPEEDUP vs. TOKEN COST (B=${blockSize}) ===`)
    console.log(
      '  size    N   seq latency  par latency  speedup   input tokens (seq→par)  amplification',
    )

    for (const s of scenarios) {
      callLog.length = 0
      const conversation = buildConversation(s.rounds, s.roundTokens)
      const totalTokens = roughTokenCountEstimationForMessages(conversation)

      await parallel.generateCompactSummaryParallel(
        conversation,
        { abortController: new AbortController() } as any,
        { forkContextMessages: conversation } as any,
        undefined,
        blockSize,
      )

      const seqLatency = modelLatencyMs(totalTokens, SEQUENTIAL_OUTPUT_TOKENS)
      const parLatency = Math.max(...callLog.map(c => c.modeledMs))
      const parInput = callLog.reduce((sum, c) => sum + c.inputTokens, 0)

      console.log(
        `  ${s.label.padEnd(7)} ${String(callLog.length).padEnd(3)} ` +
          `${fmt(seqLatency).padEnd(12)} ${fmt(parLatency).padEnd(12)} ` +
          `${(seqLatency / parLatency).toFixed(2)}x`.padEnd(9) +
          ` ${totalTokens.toLocaleString()} → ${parInput.toLocaleString()}`.padEnd(24) +
          ` ${(parInput / totalTokens).toFixed(2)}x`,
      )
    }
  })

  test('32K context window: block-size sweep + summary-budget policy', async () => {
    // Compaction can only ever see what fits below the auto-compact
    // threshold, so at 32K the ENTIRE input to compaction is ~27K:
    //   effective window = 32,000 - summary reserve
    //   threshold        = effective window - autocompact buffer
    // Using the 32K-rescaled buffers (summary reserve 3,200 / buffer 2,000)
    // from prototype/sm-compact-tool-result-clearing. NOTE: master's stock
    // values (20,000 / 13,000) give 32,000-20,000-13,000 = -1,000 — a
    // NEGATIVE threshold, i.e. autocompact fires on every single turn. Those
    // rescaled constants are a prerequisite for any of this at 32K.
    const CONTEXT_WINDOW = 32_000
    const triggerAt = CONTEXT_WINDOW - 3_200 - 2_000 // ~26,800

    const conversation = buildConversation(20, 1_340) // ≈ triggerAt
    const totalTokens = roughTokenCountEstimationForMessages(conversation)
    const seqLatency = modelLatencyMs(totalTokens, SEQUENTIAL_OUTPUT_TOKENS)

    console.log(
      `\n=== 32K CONTEXT: conversation at compaction = ${totalTokens.toLocaleString()} tokens (threshold ≈ ${triggerAt.toLocaleString()}) ===`,
    )
    console.log(
      `  sequential baseline: ${fmt(seqLatency)}, ${SEQUENTIAL_OUTPUT_TOKENS} output tokens ` +
        `(${((SEQUENTIAL_OUTPUT_TOKENS / CONTEXT_WINDOW) * 100).toFixed(1)}% of window)`,
    )

    for (const policy of ['fixed-per-block', 'fixed-total-budget'] as const) {
      console.log(
        `\n  --- policy: ${policy} ${policy === 'fixed-per-block' ? `(${PARALLEL_OUTPUT_TOKENS_PER_BLOCK} tok/block)` : `(${SEQUENTIAL_OUTPUT_TOKENS} tok total, split N ways)`} ---`,
      )
      console.log(
        '     B        N   par latency  speedup   input ampl.  merged summary   % of 32K window',
      )

      for (const B of [4_000, 6_000, 9_000, 13_500, 20_000]) {
        const blocks = parallel.partitionMessagesIntoBlocks(conversation, B)
        currentPerBlockOutput =
          policy === 'fixed-per-block'
            ? PARALLEL_OUTPUT_TOKENS_PER_BLOCK
            : Math.round(SEQUENTIAL_OUTPUT_TOKENS / blocks.length)

        callLog.length = 0
        await parallel.generateCompactSummaryParallel(
          conversation,
          { abortController: new AbortController() } as any,
          { forkContextMessages: conversation } as any,
          undefined,
          B,
        )

        const N = callLog.length
        const parLatency = Math.max(...callLog.map(c => c.modeledMs))
        const parInput = callLog.reduce((s, c) => s + c.inputTokens, 0)
        const mergedSummary = N * currentPerBlockOutput
        const pctWindow = (mergedSummary / CONTEXT_WINDOW) * 100

        console.log(
          `     ${String(B).padEnd(8)} ${String(N).padEnd(3)} ` +
            `${fmt(parLatency).padEnd(12)} ` +
            `${(seqLatency / parLatency).toFixed(2)}x`.padEnd(9) +
            ` ${(parInput / totalTokens).toFixed(2)}x`.padEnd(13) +
            ` ${mergedSummary.toLocaleString()} tok`.padEnd(17) +
            ` ${pctWindow.toFixed(1)}%`,
        )
      }
    }
    currentPerBlockOutput = PARALLEL_OUTPUT_TOKENS_PER_BLOCK
  })

  test('speedup is robust to the decode-rate assumption', async () => {
    const conversation = buildConversation(40, 2_500) // ~100K
    const totalTokens = roughTokenCountEstimationForMessages(conversation)

    callLog.length = 0
    await parallel.generateCompactSummaryParallel(
      conversation,
      { abortController: new AbortController() } as any,
      { forkContextMessages: conversation } as any,
      undefined,
      20_000,
    )
    const maxBlockInput = Math.max(...callLog.map(c => c.inputTokens))

    console.log('\n=== SENSITIVITY: decode rate (tok/s) → speedup ===')
    for (const rate of [20, 40, 60, 100, 200]) {
      const seq =
        TTFT_MS +
        (totalTokens / PREFILL_TOKENS_PER_SEC) * 1000 +
        (SEQUENTIAL_OUTPUT_TOKENS / rate) * 1000
      const par =
        TTFT_MS +
        (maxBlockInput / PREFILL_TOKENS_PER_SEC) * 1000 +
        (PARALLEL_OUTPUT_TOKENS_PER_BLOCK / rate) * 1000
      console.log(
        `  ${String(rate).padStart(3)} tok/s: seq=${fmt(seq)} par=${fmt(par)} → ${(seq / par).toFixed(2)}x`,
      )
      expect(seq / par).toBeGreaterThan(1)
    }
  })
})
