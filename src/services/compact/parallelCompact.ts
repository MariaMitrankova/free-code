/**
 * Parallel compaction — an alternative to the sequential single-call summary
 * in compact.ts's streamCompactSummary, structured analogously to the
 * three-phase scheme in the Parallel Compaction paper
 * (https://arxiv.org/html/2605.23296):
 *
 *   1. Snapshot & partition: copy the conversation and split it into N
 *      contiguous blocks of ~B tokens each (N = ceil(|X|/B), B fixed).
 *   2. Dispatch: for each block k, build a prompt of blocks 1..k in order
 *      with block k wrapped in <TARGET_BLOCK>...</TARGET_BLOCK>. All N
 *      prompts are dispatched concurrently.
 *   3. Merge: concatenate the N per-block summaries in block order to form
 *      the compacted history.
 *
 * The paper targets a self-hosted vLLM server and leans on its automatic
 * prefix-KV-cache reuse across the N concurrent requests (block k's prompt
 * is block (k-1)'s prompt plus one more block, so the prefixes nest). This
 * codebase talks to the Anthropic API instead, so the analogous mechanism is
 * Claude's own prompt caching: each dispatch below is built as
 * runForkedAgent's forkContextMessages, the exact mechanism this repo
 * already uses (see forkedAgent.ts, compact.ts's streamCompactSummary) to
 * let a forked call share the parent conversation's cached prefix. Blocks
 * 1..k-1 are byte-identical across every dispatch from k' >= k onward, so
 * whichever dispatch's cache write lands first is available to the rest for
 * the remainder of their (longer) prompt.
 *
 * This module is intentionally self-contained (no imports from compact.ts)
 * to avoid a circular import — compact.ts imports from here, not the other
 * way around.
 */

import type { ToolUseContext } from '../../Tool.js'
import type { AssistantMessage, Message, UserMessage } from '../../types/message.js'
import { logForDebugging } from '../../utils/debug.js'
import { isEnvTruthy } from '../../utils/envUtils.js'
import {
  type CacheSafeParams,
  runForkedAgent,
} from '../../utils/forkedAgent.js'
import {
  createUserMessage,
  getAssistantMessageText,
  getLastAssistantMessage,
} from '../../utils/messages.js'
import { logEvent } from '../analytics/index.js'
import { PROMPT_TOO_LONG_ERROR_MESSAGE, startsWithApiErrorPrefix } from '../api/errors.js'
import { roughTokenCountEstimationForMessages } from '../tokenEstimation.js'
import { groupMessagesByApiRound } from './grouping.js'
import { getParallelBlockCompactPrompt } from './parallelPrompt.js'

/**
 * B — the fixed block-size configuration knob from the paper, in tokens.
 * Overridable via PARALLEL_COMPACT_BLOCK_SIZE_TOKENS for testing/tuning.
 */
export const PARALLEL_COMPACT_BLOCK_SIZE_TOKENS = (() => {
  const override = process.env.PARALLEL_COMPACT_BLOCK_SIZE_TOKENS
  const parsed = override ? parseInt(override, 10) : NaN
  return !isNaN(parsed) && parsed > 0 ? parsed : 20_000
})()

/**
 * Opt-in only: this is new and unproven relative to the sequential path.
 * No GrowthBook gate yet — env-only, same pattern as
 * ENABLE_CLAUDE_CODE_PRECOMPUTED_COMPACT.
 */
export function shouldUseParallelCompaction(): boolean {
  if (isEnvTruthy(process.env.ENABLE_CLAUDE_CODE_PARALLEL_COMPACT)) {
    return true
  }
  if (isEnvTruthy(process.env.DISABLE_CLAUDE_CODE_PARALLEL_COMPACT)) {
    return false
  }
  return false
}

/**
 * Phase 1: Snapshot & partition.
 *
 * Splits `messages` into N contiguous blocks of roughly `blockSizeTokens`
 * tokens each. Never splits an API round (see groupMessagesByApiRound) —
 * the paper's flat-text blocks have no such constraint, but this codebase's
 * multi-turn message arrays must keep every tool_use paired with its
 * tool_result or the API rejects the request. A single API round larger
 * than blockSizeTokens still becomes its own (oversized) block rather than
 * being split, since splitting it isn't possible.
 */
export function partitionMessagesIntoBlocks(
  messages: Message[],
  blockSizeTokens: number = PARALLEL_COMPACT_BLOCK_SIZE_TOKENS,
): Message[][] {
  const rounds = groupMessagesByApiRound(messages)
  const blocks: Message[][] = []
  let current: Message[] = []
  let currentTokens = 0

  for (const round of rounds) {
    const roundTokens = roughTokenCountEstimationForMessages(round)
    if (current.length > 0 && currentTokens + roundTokens > blockSizeTokens) {
      blocks.push(current)
      current = []
      currentTokens = 0
    }
    current.push(...round)
    currentTokens += roundTokens
  }

  if (current.length > 0) {
    blocks.push(current)
  }

  return blocks
}

const TARGET_BLOCK_OPEN = '<TARGET_BLOCK>'
const TARGET_BLOCK_CLOSE = '</TARGET_BLOCK>'

/**
 * Phase 2 (prompt construction): builds the forkContextMessages for worker
 * k — blocks[0..k] in order, with blocks[k] (the target) wrapped in
 * <TARGET_BLOCK> markers. Blocks before the target are real conversation
 * messages, included verbatim as context (so the target block's cross-
 * references resolve) but not marked for summarization.
 *
 * The markers are synthetic isMeta user messages (same convention used
 * elsewhere for injected instructions, e.g. plan-mode reminders) rather
 * than literal text spliced into a message's content — this keeps every
 * original message byte-identical across dispatches, which is what makes
 * the growing-prefix cache sharing described above possible.
 */
export function buildTargetBlockForkContext(
  blocks: Message[][],
  targetIndex: number,
): Message[] {
  const precedingMessages = blocks.slice(0, targetIndex).flat()
  const targetBlockMessages = blocks[targetIndex] ?? []

  return [
    ...precedingMessages,
    createUserMessage({ content: TARGET_BLOCK_OPEN, isMeta: true }),
    ...targetBlockMessages,
    createUserMessage({ content: TARGET_BLOCK_CLOSE, isMeta: true }),
  ]
}

export type ParallelBlockResult = {
  blockIndex: number
  summary: string
  response: AssistantMessage
  /** Wall-clock time for this block's dispatch, for speedup measurement. */
  durationMs: number
}

async function dispatchBlockSummary(
  blocks: Message[][],
  blockIndex: number,
  cacheSafeParams: CacheSafeParams,
  context: ToolUseContext,
  customInstructions: string | undefined,
): Promise<ParallelBlockResult> {
  const forkContextMessages = buildTargetBlockForkContext(blocks, blockIndex)
  const summaryRequest: UserMessage = createUserMessage({
    content: getParallelBlockCompactPrompt(
      blockIndex + 1,
      blocks.length,
      customInstructions,
    ),
  })

  const blockStartedAt = Date.now()
  const result = await runForkedAgent({
    promptMessages: [summaryRequest],
    cacheSafeParams: { ...cacheSafeParams, forkContextMessages },
    canUseTool: async () => ({
      behavior: 'deny' as const,
      message: 'Tool use is not allowed during compaction',
      decisionReason: {
        type: 'other' as const,
        reason: 'compaction agent should only produce text summary',
      },
    }),
    querySource: 'compact',
    forkLabel: `parallel_compact_block_${blockIndex + 1}_of_${blocks.length}`,
    maxTurns: 1,
    skipCacheWrite: true,
    overrides: { abortController: context.abortController },
  })

  const durationMs = Date.now() - blockStartedAt
  const response = getLastAssistantMessage(result.messages)
  const summary = response ? getAssistantMessageText(response) : null

  logForDebugging(
    `[parallel-compact] block ${blockIndex + 1}/${blocks.length} finished in ${(durationMs / 1000).toFixed(2)}s ` +
      `(input=${result.totalUsage.input_tokens} output=${result.totalUsage.output_tokens} ` +
      `cacheRead=${result.totalUsage.cache_read_input_tokens} cacheCreate=${result.totalUsage.cache_creation_input_tokens})`,
  )

  if (!response || !summary || response.isApiErrorMessage) {
    throw new Error(
      `Parallel compaction block ${blockIndex + 1}/${blocks.length} failed to produce a summary`,
    )
  }
  if (startsWithApiErrorPrefix(summary)) {
    throw new Error(summary)
  }
  if (summary.startsWith(PROMPT_TOO_LONG_ERROR_MESSAGE)) {
    // Unlike the sequential path, there's no cross-block retry here: a
    // single block hitting prompt-too-long means blockSizeTokens itself is
    // too large for this conversation's content density (e.g. one round
    // with a huge tool result) and should be reconfigured, not retried.
    throw new Error(
      `Parallel compaction block ${blockIndex + 1}/${blocks.length} exceeded the API's prompt-too-long limit — ` +
        `PARALLEL_COMPACT_BLOCK_SIZE_TOKENS (currently ${roughTokenCountEstimationForMessages(blocks[blockIndex] ?? [])} tokens for this block) may need to be lowered`,
    )
  }

  return { blockIndex, summary, response, durationMs }
}

/**
 * Phase 2: Dispatch. Fires all N block-summary requests concurrently and
 * waits for all of them. Promise.all rejects (aborting the whole compaction
 * attempt) if any single block fails — same failure semantics as the
 * sequential path's single request failing.
 */
export async function dispatchParallelCompaction(
  blocks: Message[][],
  cacheSafeParams: CacheSafeParams,
  context: ToolUseContext,
  customInstructions?: string,
): Promise<ParallelBlockResult[]> {
  const results = await Promise.all(
    blocks.map((_, k) =>
      dispatchBlockSummary(blocks, k, cacheSafeParams, context, customInstructions),
    ),
  )
  // Promise.all already preserves input order, but sort defensively in case
  // that guarantee is ever relied on across a refactor.
  return results.sort((a, b) => a.blockIndex - b.blockIndex)
}

/**
 * Phase 3: Merge. Concatenates the N per-block summaries in block order
 * into the single compacted-history text that replaces the original
 * snapshot, matching what compact.ts's getCompactUserSummaryMessage expects
 * as input (it wraps this in <analysis>-stripped/<summary>-formatted
 * framing the same way for both the sequential and parallel paths).
 *
 * IMPORTANT: formatCompactSummary's <summary> extraction regex
 * (/<summary>([\s\S]*?)<\/summary>/) is non-global — it matches only the
 * FIRST <summary>...</summary> pair in the string. Emitting one such pair
 * per block here would silently drop every part after the first (its raw
 * <summary> tags would survive unconverted in the final text). So the N
 * per-block bodies are concatenated INSIDE a single outer <summary> pair
 * instead of N separate ones.
 */
export function mergeBlockSummaries(summaries: string[]): string {
  if (summaries.length === 1) {
    return summaries[0]!
  }
  const bodies = summaries.map(
    (summary, i) =>
      `## Part ${i + 1}/${summaries.length} of the conversation\n\n${extractSummaryBody(summary)}`,
  )
  return `<summary>\n${bodies.join('\n\n')}\n</summary>`
}

/**
 * Each block worker still emits the <analysis>/<summary> structure required
 * by the NO_TOOLS_PREAMBLE/TRAILER framing (formatCompactSummary strips
 * <analysis> and unwraps <summary> downstream, same as the sequential
 * path) — pull just the <summary> body out here before re-wrapping N of
 * them into one merged <summary> block, so formatCompactSummary's single
 * regex match against the merged text still works.
 */
function extractSummaryBody(blockSummary: string): string {
  const match = blockSummary.match(/<summary>([\s\S]*?)<\/summary>/)
  return match ? match[1]!.trim() : blockSummary.trim()
}

/**
 * Full parallel-compaction summary generation: partition + dispatch + merge.
 * Mirrors the shape compact.ts's streamCompactSummary/PTL-retry loop
 * produces (a summary string plus the underlying AssistantMessage
 * responses, so the caller can aggregate usage/token metrics) so
 * compactConversation can swap strategies without touching the surrounding
 * hooks/attachments/event-logging pipeline.
 */
export async function generateCompactSummaryParallel(
  messages: Message[],
  context: ToolUseContext,
  cacheSafeParams: CacheSafeParams,
  customInstructions: string | undefined,
  blockSizeTokens: number = PARALLEL_COMPACT_BLOCK_SIZE_TOKENS,
): Promise<{
  summary: string
  responses: AssistantMessage[]
  blockCount: number
  durationMs: number
}> {
  const startedAt = Date.now()
  const blocks = partitionMessagesIntoBlocks(messages, blockSizeTokens)
  const results = await dispatchParallelCompaction(
    blocks,
    cacheSafeParams,
    context,
    customInstructions,
  )
  const summary = mergeBlockSummaries(results.map(r => r.summary))
  const durationMs = Date.now() - startedAt

  // The concurrency payoff, stated directly: wall-clock time should track
  // the SLOWEST block, not the sum of all of them. serialMs/durationMs is
  // the speedup this path bought over dispatching the same blocks one at a
  // time (it is NOT the speedup vs. the sequential single-call path —
  // compare against tengu_compact's compactionDurationMs for that).
  const perBlockMs = results.map(r => r.durationMs)
  const serialMs = perBlockMs.reduce((sum, ms) => sum + ms, 0)
  const criticalPathMs = Math.max(...perBlockMs, 0)
  logForDebugging(
    `[parallel-compact] ${blocks.length} blocks in ${(durationMs / 1000).toFixed(2)}s ` +
      `(critical path ${(criticalPathMs / 1000).toFixed(2)}s, would be ${(serialMs / 1000).toFixed(2)}s serially — ` +
      `${serialMs > 0 ? (serialMs / durationMs).toFixed(2) : '1.00'}x from concurrency); ` +
      `per-block: ${perBlockMs.map(ms => (ms / 1000).toFixed(2) + 's').join(', ')}`,
  )
  logEvent('tengu_compact_parallel_dispatch', {
    blockCount: blocks.length,
    durationMs,
    criticalPathMs,
    serialEquivalentMs: serialMs,
    blockSizeTokens,
  })

  return {
    summary,
    responses: results.map(r => r.response),
    blockCount: blocks.length,
    durationMs,
  }
}
