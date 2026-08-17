// Synthetic test for the parallel-compaction module (parallelCompact.ts,
// parallelPrompt.ts) — exercises the pure, non-API pieces of the 3-phase
// algorithm (partition / prompt-construction / merge) described in the
// Parallel Compaction paper (https://arxiv.org/html/2605.23296):
//   1. Snapshot & partition — partitionMessagesIntoBlocks
//   2. Dispatch (prompt construction only, no live API call) —
//      buildTargetBlockForkContext + getParallelBlockCompactPrompt
//   3. Merge — mergeBlockSummaries
// No live API calls are made — dispatchParallelCompaction/
// generateCompactSummaryParallel (which DO call the API via
// runForkedAgent) are intentionally not exercised here.

import {
  buildTargetBlockForkContext,
  mergeBlockSummaries,
  partitionMessagesIntoBlocks,
} from './src/services/compact/parallelCompact.js'
import { getParallelBlockCompactPrompt } from './src/services/compact/parallelPrompt.js'

let failures = 0
function check(label: string, condition: boolean, detail?: string): void {
  const status = condition ? 'PASS' : 'FAIL'
  console.log(`[${status}] ${label}${detail ? ' — ' + detail : ''}`)
  if (!condition) failures++
}

let uuidCounter = 0
function uuid(): string {
  uuidCounter += 1
  return `uuid-${uuidCounter}`
}
function userTextMsg(text: string) {
  return {
    type: 'user',
    uuid: uuid(),
    timestamp: new Date().toISOString(),
    message: { role: 'user', content: text },
  }
}
function assistantTextMsg(text: string) {
  return {
    type: 'assistant',
    uuid: uuid(),
    timestamp: new Date().toISOString(),
    message: {
      role: 'assistant',
      id: `msg-${uuidCounter}`,
      content: [{ type: 'text', text }],
    },
  }
}
function assistantToolUseMsg(id: string, msgId: string) {
  return {
    type: 'assistant',
    uuid: uuid(),
    timestamp: new Date().toISOString(),
    message: {
      role: 'assistant',
      id: msgId,
      content: [{ type: 'tool_use', id, name: 'Read', input: { file_path: '/x' } }],
    },
  }
}
function userToolResultMsg(id: string, content: string) {
  return {
    type: 'user',
    uuid: uuid(),
    timestamp: new Date().toISOString(),
    message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: id, content }] },
  }
}

// --- Build a synthetic conversation: 6 API rounds, each ~2500 tokens
// (10,000 chars of tool_result content ≈ 2500 tokens at 4 chars/token). ---
const ROUND_TOOL_RESULT = 'x'.repeat(10_000) // ~2500 tokens
function buildRound(i: number): any[] {
  const toolId = `tool_${i}`
  const msgId = `resp_${i}`
  return [
    userTextMsg(`round ${i} request`),
    assistantToolUseMsg(toolId, msgId),
    userToolResultMsg(toolId, ROUND_TOOL_RESULT),
    assistantTextMsg(`round ${i} reply`),
  ]
}
const NUM_ROUNDS = 6
const messages: any[] = []
for (let i = 0; i < NUM_ROUNDS; i++) {
  messages.push(...buildRound(i))
}
console.log(`Synthetic conversation: ${messages.length} messages across ${NUM_ROUNDS} API rounds (~2500 tokens/round)`)

// --- Phase 1: Snapshot & partition ---
// blockSizeTokens=6000 should fit ~2 rounds/block (2*2500=5000 <= 6000,
// 3*2500=7500 > 6000) → 3 blocks for 6 rounds.
const BLOCK_SIZE = 6_000
const blocks = partitionMessagesIntoBlocks(messages as any, BLOCK_SIZE)

check(
  `partition produces multiple blocks (got ${blocks.length})`,
  blocks.length > 1 && blocks.length < NUM_ROUNDS,
  `expected somewhere between 2 and ${NUM_ROUNDS - 1}`,
)

const allBlockMessages = blocks.flat()
check(
  'partition preserves every message exactly once, in order',
  allBlockMessages.length === messages.length &&
    allBlockMessages.every((m, i) => m.uuid === messages[i]!.uuid),
)

// No API round should be split across two blocks: every tool_use's matching
// tool_result must land in the same block.
let noSplitRounds = true
for (const block of blocks) {
  const toolUseIds = new Set<string>()
  const toolResultIds = new Set<string>()
  for (const m of block) {
    if (m.type === 'assistant') {
      for (const b of m.message.content) {
        if (b.type === 'tool_use') toolUseIds.add(b.id)
      }
    }
    if (m.type === 'user' && Array.isArray(m.message.content)) {
      for (const b of m.message.content) {
        if (b.type === 'tool_result') toolResultIds.add(b.tool_use_id)
      }
    }
  }
  for (const id of toolUseIds) {
    if (!toolResultIds.has(id)) noSplitRounds = false
  }
}
check('no API round (tool_use/tool_result pair) is split across blocks', noSplitRounds)

// --- Phase 2: prompt construction (buildTargetBlockForkContext) ---
const targetIndex = Math.min(1, blocks.length - 1)
const forkContext = buildTargetBlockForkContext(blocks as any, targetIndex)

const precedingCount = blocks.slice(0, targetIndex).flat().length
const targetCount = blocks[targetIndex]!.length
check(
  'forkContext length = preceding blocks + 2 markers + target block',
  forkContext.length === precedingCount + 2 + targetCount,
  `got ${forkContext.length}, expected ${precedingCount + 2 + targetCount}`,
)

const openMarker = forkContext[precedingCount]
const closeMarker = forkContext[forkContext.length - 1]
check(
  'open marker is <TARGET_BLOCK> and isMeta',
  openMarker?.message?.content === '<TARGET_BLOCK>' && openMarker?.isMeta === true,
)
check(
  'close marker is </TARGET_BLOCK> and isMeta',
  closeMarker?.message?.content === '</TARGET_BLOCK>' && closeMarker?.isMeta === true,
)
check(
  'messages between markers are exactly the target block, untouched',
  blocks[targetIndex]!.every(
    (m, i) => forkContext[precedingCount + 1 + i]!.uuid === m.uuid,
  ),
)
check(
  'preceding messages are byte-identical to the original blocks (prefix-cache-sharing precondition)',
  blocks
    .slice(0, targetIndex)
    .flat()
    .every((m, i) => forkContext[i] === m), // same object reference, not just equal uuid
)

// --- Phase 2b: prompt text (getParallelBlockCompactPrompt) ---
const middlePrompt = getParallelBlockCompactPrompt(1, 3)
const finalPrompt = getParallelBlockCompactPrompt(3, 3)
check('middle-block prompt mentions block 1 of 3', middlePrompt.includes('block 1 of 3'))
check('middle-block prompt omits "Current Work" section', !middlePrompt.includes('8. Current Work'))
check('final-block prompt includes "Current Work" section', finalPrompt.includes('8. Current Work'))
check('final-block prompt includes "Optional Next Step" section', finalPrompt.includes('9. Optional Next Step'))
check('prompt references TARGET_BLOCK marker', middlePrompt.includes('<TARGET_BLOCK>'))

const customPrompt = getParallelBlockCompactPrompt(1, 1, 'Focus on TypeScript changes only.')
check(
  'custom instructions are appended',
  customPrompt.includes('Focus on TypeScript changes only.'),
)

// --- Phase 3: Merge ---
const singleSummary = '<summary>\nonly block\n</summary>'
check(
  'single-block merge returns the summary unchanged',
  mergeBlockSummaries([singleSummary]) === singleSummary,
)

const threeSummaries = [
  '<analysis>ignore me</analysis>\n<summary>\nfirst block content\n</summary>',
  '<summary>\nsecond block content\n</summary>',
  '<summary>\nthird block content\n</summary>',
]
const merged = mergeBlockSummaries(threeSummaries)
check('merged output contains block 1 content', merged.includes('first block content'))
check('merged output contains block 2 content', merged.includes('second block content'))
check('merged output contains block 3 content', merged.includes('third block content'))
check(
  'merged output preserves block order (1 before 2 before 3)',
  merged.indexOf('first block content') <
    merged.indexOf('second block content') &&
    merged.indexOf('second block content') < merged.indexOf('third block content'),
)
check(
  'merged output is wrapped as exactly ONE outer <summary> block (formatCompactSummary uses a non-global regex, so N separate <summary> tags would silently drop parts 2..N)',
  (merged.match(/<summary>/g) ?? []).length === 1 &&
    (merged.match(/<\/summary>/g) ?? []).length === 1 &&
    /^<summary>[\s\S]*<\/summary>$/.test(merged.trim()),
)

console.log(`\n${failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`}`)
if (failures > 0) process.exit(1)
