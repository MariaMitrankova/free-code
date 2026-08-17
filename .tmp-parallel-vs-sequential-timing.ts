// Real-API wall-clock comparison: sequential compaction (one call
// summarizing the whole conversation) vs. parallel compaction (N concurrent
// calls, each summarizing one block — see parallelCompact.ts).
//
// This tests the actual mechanism the Parallel Compaction paper
// (https://arxiv.org/html/2605.23296) claims speeds things up: decode
// (output-token generation) is autoregressive and dominates request
// latency far more than prefill (input processing) does, so replacing one
// big call with a large output by N concurrent calls each with a SMALL
// output should reduce wall-clock time — bounded by the slowest single
// block's decode time, not the sum of all of them.
//
// Deliberately NOT tested here: cross-block prompt-cache sharing (the
// paper's vLLM-KV-cache-reuse analog). Getting real cache_control
// breakpoints right requires the production runForkedAgent/query()
// machinery and a warm parent cache — out of scope for a standalone
// script hitting the API directly. This test isolates and measures ONLY
// the output-length/concurrency mechanism.
//
// Reuses REAL production code: partitionMessagesIntoBlocks,
// buildTargetBlockForkContext, mergeBlockSummaries (parallelCompact.ts),
// getParallelBlockCompactPrompt (parallelPrompt.ts), getCompactPrompt
// (prompt.ts) — only the "call the Anthropic SDK directly instead of
// through runForkedAgent/app context" part is test-only scaffolding.
//
// Uses the real Claude Code OAuth credential from ~/.claude/.credentials.json
// (Windows-side) — never printed, only read into memory and used for auth.

import { readFile } from 'fs/promises'
import Anthropic from '@anthropic-ai/sdk'
import {
  buildTargetBlockForkContext,
  mergeBlockSummaries,
  partitionMessagesIntoBlocks,
} from './src/services/compact/parallelCompact.js'
import { getParallelBlockCompactPrompt } from './src/services/compact/parallelPrompt.js'
import { getCompactPrompt } from './src/services/compact/prompt.js'

const MODEL = 'claude-sonnet-4-6'
// Shrunk well below the earlier 8-round/9.5K-input attempt, which hit a
// 429 rate_limit_error — per user instruction, keep prompts small rather
// than retry at the same size. Small enough that even the ~3-4K-token
// synthetic conversation below still splits into multiple blocks (at
// 1500 it collapsed to a single block, which defeats the point of the test).
const BLOCK_SIZE_TOKENS = 500
// Sequential must write one full 9-section summary of everything.
const SEQUENTIAL_MAX_OUTPUT_TOKENS = 500
// Each parallel worker only summarizes its own block — a fraction of the
// sequential output, matching how a real deployment would configure it.
const PARALLEL_MAX_OUTPUT_TOKENS_PER_BLOCK = 200

const SYSTEM_PROMPT =
  'You are a helpful AI assistant tasked with summarizing conversations.'

async function loadAccessToken(): Promise<string> {
  const raw = await readFile(
    '/mnt/c/Users/MARIAM~1/.claude/.credentials.json',
    'utf-8',
  )
  const parsed = JSON.parse(raw)
  const token = parsed?.claudeAiOauth?.accessToken
  if (!token) throw new Error('No claudeAiOauth.accessToken found in credentials file')
  return token
}

// --- Synthetic conversation, built as internal Message[] objects (same
// shape used by the real partition/prompt functions) so partitioning and
// block-prompt construction run through the REAL production code. ---
let uuidCounter = 0
function uuid(): string {
  uuidCounter += 1
  return `uuid-${uuidCounter}`
}
function userMsg(text: string): any {
  return {
    type: 'user',
    uuid: uuid(),
    timestamp: new Date().toISOString(),
    isMeta: undefined,
    message: { role: 'user', content: text },
  }
}
function assistantMsg(text: string): any {
  return {
    type: 'assistant',
    uuid: uuid(),
    timestamp: new Date().toISOString(),
    message: { role: 'assistant', id: `msg-${uuidCounter}`, content: text },
  }
}

const FILE_BODIES = [
  'export function handleRequest(input: RequestInput): Promise<Response> {\n  const result = processInternal(input)\n  return result\n}\n'.repeat(4),
  'export class TokenRefresher {\n  private inFlight: Promise<Token> | null = null\n  async refresh(): Promise<Token> {\n    if (this.inFlight) return this.inFlight\n    this.inFlight = this.doRefresh()\n    try { return await this.inFlight } finally { this.inFlight = null }\n  }\n}\n'.repeat(4),
  'function calculateConversionRate(starts: number, completions: number): number {\n  if (starts === 0) return 0\n  return completions / starts\n}\n'.repeat(4),
]

function buildSyntheticConversation(numRounds: number): any[] {
  const messages: any[] = []
  messages.push(
    userMsg(
      'We need to fix a race condition in the token refresh logic and add a checkout conversion-rate dashboard panel. Please investigate and implement both.',
    ),
  )
  for (let i = 0; i < numRounds; i++) {
    const body = FILE_BODIES[i % FILE_BODIES.length]
    messages.push(
      assistantMsg(
        `Looking into round ${i}. Reading the relevant file and analyzing the current implementation.\n\n[File content]\n${body}\n\nI found an issue here: ${i % 2 === 0 ? 'a race condition between concurrent refresh calls' : 'the conversion rate calculation divides by zero when there are no checkout starts'}. Fixing it now by ${i % 2 === 0 ? 'adding an in-flight promise cache' : 'adding a zero-guard'}.`,
      ),
    )
    messages.push(
      userMsg(
        i === numRounds - 1
          ? 'Great, that all looks correct. Can you also add loading and error states to the dashboard panel?'
          : `Thanks, that fix for round ${i} looks good. Please continue to the next part.`,
      ),
    )
  }
  messages.push(
    assistantMsg(
      'Added a loading skeleton and an error boundary with a retry button to the dashboard panel. All changes are complete and tests pass.',
    ),
  )
  return messages
}

// --- Message[] -> Anthropic.MessageParam[], merging adjacent same-role
// turns (required for strict user/assistant alternation — the TARGET_BLOCK
// markers are synthetic user-role messages that can land next to other
// user-role messages). ---
function toApiMessages(messages: any[]): Anthropic.MessageParam[] {
  const result: Anthropic.MessageParam[] = []
  for (const m of messages) {
    const role = m.message.role as 'user' | 'assistant'
    const text = typeof m.message.content === 'string' ? m.message.content : ''
    const last = result[result.length - 1]
    if (last && last.role === role && typeof last.content === 'string') {
      last.content = `${last.content}\n\n${text}`
    } else {
      result.push({ role, content: text })
    }
  }
  return result
}

function estimateTokens(messages: Anthropic.MessageParam[]): number {
  const chars = messages.reduce(
    (sum, m) => sum + (typeof m.content === 'string' ? m.content.length : 0),
    0,
  )
  return Math.round(chars / 4)
}

async function runSequential(
  client: Anthropic,
  conversation: any[],
): Promise<{ wallMs: number; inputTokens: number; outputTokens: number }> {
  const compactPrompt = getCompactPrompt()
  const apiMessages = toApiMessages([...conversation, userMsg(compactPrompt)])

  console.log(
    `\n--- SEQUENTIAL: 1 call, ~${estimateTokens(apiMessages)} input tokens, max_output=${SEQUENTIAL_MAX_OUTPUT_TOKENS} ---`,
  )
  const t0 = performance.now()
  const response = await client.messages.create({
    model: MODEL,
    max_tokens: SEQUENTIAL_MAX_OUTPUT_TOKENS,
    system: SYSTEM_PROMPT,
    messages: apiMessages,
  })
  const wallMs = performance.now() - t0

  console.log(
    `  wall=${wallMs.toFixed(0)}ms input_tokens=${response.usage.input_tokens} output_tokens=${response.usage.output_tokens} stop_reason=${response.stop_reason}`,
  )
  return {
    wallMs,
    inputTokens: response.usage.input_tokens,
    outputTokens: response.usage.output_tokens,
  }
}

async function runParallel(
  client: Anthropic,
  conversation: any[],
): Promise<{
  wallMs: number
  blockWallMsList: number[]
  totalInputTokens: number
  totalOutputTokens: number
  blockCount: number
  merged: string
}> {
  const blocks = partitionMessagesIntoBlocks(conversation, BLOCK_SIZE_TOKENS)
  console.log(
    `\n--- PARALLEL: ${blocks.length} blocks (block_size=${BLOCK_SIZE_TOKENS} tokens), dispatched concurrently ---`,
  )

  const t0 = performance.now()
  const results = await Promise.all(
    blocks.map(async (_, k) => {
      const forkContext = buildTargetBlockForkContext(blocks, k)
      const prompt = getParallelBlockCompactPrompt(k + 1, blocks.length)
      const apiMessages = toApiMessages([...forkContext, userMsg(prompt)])

      const blockT0 = performance.now()
      const response = await client.messages.create({
        model: MODEL,
        max_tokens: PARALLEL_MAX_OUTPUT_TOKENS_PER_BLOCK,
        system: SYSTEM_PROMPT,
        messages: apiMessages,
      })
      const blockWallMs = performance.now() - blockT0

      const summary = response.content
        .filter((b): b is Anthropic.TextBlock => b.type === 'text')
        .map(b => b.text)
        .join('')

      console.log(
        `  block ${k + 1}/${blocks.length}: wall=${blockWallMs.toFixed(0)}ms input_tokens=${response.usage.input_tokens} output_tokens=${response.usage.output_tokens} (~${estimateTokens(apiMessages)} est. input tokens sent)`,
      )

      return {
        blockIndex: k,
        summary,
        wallMs: blockWallMs,
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      }
    }),
  )
  const wallMs = performance.now() - t0

  const merged = mergeBlockSummaries(results.map(r => r.summary))

  return {
    wallMs,
    blockWallMsList: results.map(r => r.wallMs),
    totalInputTokens: results.reduce((s, r) => s + r.inputTokens, 0),
    totalOutputTokens: results.reduce((s, r) => s + r.outputTokens, 0),
    blockCount: blocks.length,
    merged,
  }
}

async function main(): Promise<void> {
  const accessToken = await loadAccessToken()
  const client = new Anthropic({ apiKey: null, authToken: accessToken })

  // 4 short rounds lands around 3-4K tokens with these synthetic bodies —
  // small enough to (hopefully) avoid the 429 the 8-round/9.5K-token
  // version hit, while BLOCK_SIZE_TOKENS=1500 still yields 2-3 blocks.
  const conversation = buildSyntheticConversation(4)
  console.log(
    `Synthetic conversation: ${conversation.length} messages, model=${MODEL}`,
  )

  let sequential: Awaited<ReturnType<typeof runSequential>> | null = null
  let parallel: Awaited<ReturnType<typeof runParallel>> | null = null

  try {
    sequential = await runSequential(client, conversation)
  } catch (error) {
    console.error(`\nSEQUENTIAL call failed: ${error instanceof Error ? error.message : String(error)}`)
  }

  try {
    parallel = await runParallel(client, conversation)
  } catch (error) {
    console.error(`\nPARALLEL dispatch failed: ${error instanceof Error ? error.message : String(error)}`)
  }

  console.log(`\n=== RESULT ===`)
  if (sequential) {
    console.log(
      `Sequential : wall=${sequential.wallMs.toFixed(0)}ms  input_tokens=${sequential.inputTokens}  output_tokens=${sequential.outputTokens}`,
    )
  } else {
    console.log('Sequential : FAILED (see error above)')
  }
  if (parallel) {
    const criticalPathMs = Math.max(...parallel.blockWallMsList)
    console.log(
      `Parallel   : wall=${parallel.wallMs.toFixed(0)}ms  (critical-path single block=${criticalPathMs.toFixed(0)}ms)  blocks=${parallel.blockCount}  total_input_tokens=${parallel.totalInputTokens}  total_output_tokens=${parallel.totalOutputTokens}`,
    )
  } else {
    console.log('Parallel   : FAILED (see error above)')
  }

  if (sequential && parallel) {
    const speedup = sequential.wallMs / parallel.wallMs
    console.log(
      `\nWall-clock speedup (sequential/parallel): ${speedup.toFixed(2)}x`,
    )
    console.log(
      `Token cost overhead (parallel total input tokens / sequential input tokens): ${(parallel.totalInputTokens / sequential.inputTokens).toFixed(2)}x` +
        ` — parallel reprocesses the growing shared prefix in every block's request (no cache-sharing measured here), so it trades extra billed input tokens for wall-clock time.`,
    )
    console.log(
      `Output token overhead (parallel total output / sequential output): ${(parallel.totalOutputTokens / sequential.outputTokens).toFixed(2)}x` +
        ` — N per-block summaries typically produce somewhat more combined text than one holistic summary (per-block section headers, some cross-block redundancy).`,
    )
  }
}

await main()
