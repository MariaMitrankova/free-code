# Compaction

Compaction is how the CLI keeps a long-running conversation inside a model's context
window: instead of truncating history, it asks the model to write a structured summary
of the older messages, discards the originals, and reconstructs just enough state
(recently read files, active plan, active skills, tool listings) for the conversation to
continue coherently. All of it lives under `src/services/compact/`, with the `/compact`
slash command and a few UI components as the outer layers.

## Entry points

- **`src/commands/compact/compact.ts`** — the `/compact` slash command. Order of
  attempts: session-memory compaction → reactive-only mode (if that experiment is on) →
  traditional compaction (microcompact first, then full summarization).
- **`src/services/compact/autoCompact.ts`** — automatic compaction, checked every turn
  once token usage crosses a threshold. Same fallback order minus the reactive branch.
- Both funnel into `compactConversation()` / `partialCompactConversation()` in
  **`src/services/compact/compact.ts`**, the shared implementation.

## Triggering (`autoCompact.ts`)

- `getAutoCompactThreshold(model)` = effective context window (model's context window,
  minus ~20K reserved for the summary's own output, optionally capped by
  `CLAUDE_CODE_AUTO_COMPACT_WINDOW`) minus `AUTOCOMPACT_BUFFER_TOKENS` (13,000).
- `shouldAutoCompact()` gates on this threshold, and bails out early for:
  - forked/recursive query sources (`session_memory`, `compact` itself — would deadlock),
  - the `CONTEXT_COLLAPSE` experiment's own agent,
  - `DISABLE_COMPACT` / `DISABLE_AUTO_COMPACT` env vars or the user's
    `autoCompactEnabled` setting,
  - `REACTIVE_COMPACT` / `CONTEXT_COLLAPSE` experiments when they're active (they own
    context management instead).
- `autoCompactIfNeeded()` has a **circuit breaker**: after
  `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES` (3) failed attempts in a row it stops retrying
  for the rest of the session, so a conversation that's irrecoverably over the limit
  doesn't hammer the API every turn.
- Manual `/compact` has its own, tighter blocking threshold
  (`MANUAL_COMPACT_BUFFER_TOKENS` = 3,000) enforced via `calculateTokenWarningState()`
  in the same file, which also drives the UI's "context left" warning
  (`compactWarningHook.ts` / `compactWarningState.ts`, `TokenWarning.tsx`).

## The compaction ladder

Cheapest technique first; each stage only runs if the previous one wasn't enough or
doesn't apply. All are tried before falling back to full LLM summarization.

1. **Time-based microcompact** (`microCompact.ts` → `maybeTimeBasedMicrocompact`) —
   if the gap since the last assistant message exceeds a configured threshold (the
   server-side prompt cache has gone cold anyway), content-clear old tool results
   in place, keeping only the most recent N. Cheapest: no API call, no message removal,
   just replaces tool_result content with `[Old tool result content cleared]`.
2. **Cached microcompact** (`cachedMicrocompact.ts`, gated by feature flag
   `CACHED_MICROCOMPACT`) — uses the API's cache-editing mechanism to delete old tool
   results without invalidating the cached prompt prefix. Tracks state across turns
   (`cachedMCState`) and only runs on the main thread (not forked/sub-agents).
3. **Session memory compaction** (`sessionMemoryCompact.ts`) — an experimental path
   that, instead of asking the model to summarize, reuses a `SessionMemory` file that's
   already being incrementally extracted in the background. No compaction API call
   needed when it works. See [Session memory compaction](#session-memory-compaction)
   below for the full mechanics, gating, and a known failure mode. Falls back to
   `null` (→ next stage) if the memory file doesn't exist yet, is still the empty
   template, or the resulting context is still too large.
4. **Full summarization** (`compact.ts` → `compactConversation`) — the traditional
   path: send the whole conversation to the model with a dedicated summarization
   prompt (see below), replace all prior messages with a compact boundary marker +
   the summary, and reconstruct minimal follow-up state.
5. **Partial compaction** (`compact.ts` → `partialCompactConversation`) — a
   user-driven variant (via the message selector UI) that summarizes only messages
   *before* or *after* a chosen pivot message (`direction: 'from' | 'up_to'`), keeping
   the other half verbatim. Used when the user wants to compact "everything up to here"
   or "everything after here" rather than the whole conversation.
6. **Reactive compact** (`reactiveCompact.ts`, gated by `REACTIVE_COMPACT`, an
   experimental "compact only when the API actually rejects a too-long prompt" mode) —
   not fully traced here, but wired into both `/compact` and the general query path as
   an alternative to proactive threshold-based triggering.

## Session memory compaction

Two separate systems are involved, gated independently — that independence is the
source of the main gotcha below.

- **`src/services/SessionMemory/`** (`sessionMemory.ts`, `sessionMemoryUtils.ts`,
  `prompts.ts`) — background extraction that maintains a running markdown notes file
  for the session.
- **`src/services/compact/sessionMemoryCompact.ts`** — the compaction-time consumer
  that splices that file in as the "summary" instead of calling the model.

### Background extraction (`SessionMemory/sessionMemory.ts`)

- Registered as a post-sampling hook (`registerPostSamplingHook(extractSessionMemory)`)
  — runs after each assistant turn completes, via a forked subagent that shares the
  main thread's prompt cache, without blocking the conversation.
- **Gate**: `isSessionMemoryGateEnabled()` reads only the GrowthBook flag
  `tengu_session_memory` (default `false`). **No env var override exists for this
  gate** — unlike the compaction-side gate below.
- **Trigger logic** (`shouldExtractMemory`):
  - Must first cross `minimumMessageTokensToInit` (10,000 tokens) to initialize.
  - Then re-extracts once `minimumTokensBetweenUpdate` (5,000 tokens of context
    growth) is met **and either** `toolCallsBetweenUpdates` (3) tool calls have
    happened since the last extraction, **or** the last assistant turn had no tool
    calls (a natural conversation break). The token threshold is always required
    even if the tool-call threshold is met, to cap extraction frequency.
- The file is capped at `MAX_TOTAL_SESSION_MEMORY_TOKENS` (12,000 tokens,
  `prompts.ts`) — the extraction agent is instructed to self-condense as it grows,
  prioritizing "Current State" and "Errors & Corrections" over older detail.
- Extraction runs are coalesced: only one in flight at a time (`inProgress`); a turn
  that arrives mid-extraction is stashed and run as a trailing extraction afterward
  rather than run concurrently.
- `waitForSessionMemoryExtraction()` lets compaction wait (up to 15s) for an
  in-flight extraction to finish before reading the file, unless it's stale (>60s
  old), in which case it proceeds without waiting.

### Compaction-time consumption (`compact/sessionMemoryCompact.ts`)

- **Gate**: `shouldUseSessionMemoryCompaction()` — the `ENABLE_CLAUDE_CODE_SM_COMPACT`
  / `DISABLE_CLAUDE_CODE_SM_COMPACT` env vars override; otherwise it requires *both*
  `tengu_session_memory` and `tengu_sm_compact` GrowthBook flags to be true.
- ⚠️ **Gotcha**: this gate is independent of the extraction gate above. Setting
  `ENABLE_CLAUDE_CODE_SM_COMPACT=1` alone does **not** turn on extraction — if
  `tengu_session_memory` is false (e.g. no GrowthBook connectivity on a self-hosted
  deployment), the memory file is never written, and `trySessionMemoryCompaction()`
  always bails at its first check (`tengu_sm_compact_no_session_memory`), falling
  through to full legacy compaction every time. Both gates need to be open for this
  path to have any effect.
- `trySessionMemoryCompaction(messages, agentId, autoCompactThreshold?)`:
  1. Waits for in-flight extraction, reads the file; bails (`null`) if missing or
     still the empty template (`isSessionMemoryEmpty`).
  2. Finds the boundary via `getLastSummarizedMessageId()`; bails if that message ID
     no longer exists in `messages` (e.g. pruned by an earlier compaction).
  3. `calculateMessagesToKeepIndex()` decides how much of the *recent* conversation
     to keep verbatim (not summarized) — expands backward from the boundary until
     `minTokens` (10,000) and `minTextBlockMessages` (5) are satisfied, capped at
     `maxTokens` (40,000).
  4. `adjustIndexToPreserveAPIInvariants()` nudges that cut point so it never splits
     a tool_use/tool_result pair or a streamed multi-block assistant turn.
  5. Splices the memory file in as the summary message (same wrapper format as
     legacy compaction's), attaches the plan file if any, builds the boundary marker.
  6. For autocompact: if the resulting token estimate still exceeds
     `autoCompactThreshold`, gives up (`tengu_sm_compact_threshold_exceeded`) rather
     than compacting to something that will immediately retrigger.
- The payoff when it works: this is a synchronous local operation (index math + file
  read) — no summarization API call at compaction time, unlike every other stage in
  the ladder.

### Known failure mode: large tool outputs defeat the "kept tail" logic

- `calculateMessagesToKeepIndex()`'s walk-back loop is driven by `hasTextBlocks()`,
  which only counts `type: 'text'` blocks toward `minTextBlockMessages` —
  `tool_result` blocks never count toward that minimum.
- But every message the walk passes over, text or not, is added to `totalTokens` via
  `estimateMessageTokens()` (full-size counting — no truncation happens at this
  stage; the 5,000-token-per-file cap only applies to *post-compaction* file
  re-attachment, not to what's already sitting live in history).
- So a handful of large tool outputs (big file reads, large grep/bash output) sitting
  between the "real" text messages the loop is hunting for get swept into the kept
  window in full, inflating `totalTokens` without helping satisfy the text-message
  minimum — easily pushing the kept tail from the ~10K target toward the 40K hard
  cap purely from tool-output bulk.
- This can trip the `autoCompactThreshold` bail-out above, or — even when it
  doesn't bail — produce a "compacted" result that isn't meaningfully smaller than
  before, defeating the purpose.
- No earlier stage protects against this: `microCompact.ts`'s time-based and cached
  clearing both deliberately leave the *most recent* tool results untouched
  (`keepRecent`) so the model doesn't lose fresh working context — which is exactly
  the window `calculateMessagesToKeepIndex` is then forced to keep raw.

## The summarization prompt (`prompt.ts`)

- `getCompactPrompt()` / `getPartialCompactPrompt()` build a prompt that forces a
  **text-only, no-tool-calls** response (`NO_TOOLS_PREAMBLE` + `NO_TOOLS_TRAILER` —
  belt-and-suspenders because the model occasionally tries to call a tool anyway, which
  would waste the single allowed turn).
- The model is asked to think in a scratch `<analysis>` block, then produce a
  `<summary>` with fixed sections: Primary Request and Intent, Key Technical Concepts,
  Files and Code Sections (with code snippets), Errors and Fixes, Problem Solving, All
  User Messages, Pending Tasks, Current Work, Optional Next Step.
- `formatCompactSummary()` strips the `<analysis>` scratchpad and turns `<summary>`
  into a plain "Summary:" section before it's stored.
- `getCompactUserSummaryMessage()` wraps the formatted summary into the synthetic user
  message that replaces history, adds a pointer to the full transcript file, and (for
  autocompact) an instruction to resume work silently without recapping.
- User-supplied custom instructions (`/compact <instructions>`) and hook-provided
  instructions (`PreCompact` hook) are merged in and appended to the prompt.

## Executing the summarization call (`compact.ts`)

- Images/documents are stripped from messages before sending for summarization
  (`stripImagesFromMessages`) — not needed for a text summary and can themselves blow
  the prompt-too-long limit.
- Certain attachment types that get re-injected after compaction anyway are stripped
  from the summarizer's input (`stripReinjectedAttachments`).
- **Prompt-cache sharing**: by default the summarization call runs as a *forked agent*
  that reuses the main conversation's cached prompt prefix (system prompt, tools,
  message prefix) rather than paying for a cold cache — falls back to a plain streaming
  call if that fails.
- **Prompt-too-long retry loop**: if even the compaction request itself is too long for
  the model, `truncateHeadForPTLRetry()` drops the oldest API-round message groups
  (grouped via `grouping.ts`'s `groupMessagesByApiRound`) and retries, up to
  `MAX_PTL_RETRIES` (3) times — a last-resort escape hatch so the user isn't stuck.
- A streaming-failure retry (`MAX_COMPACT_STREAMING_RETRIES`) also exists, gated behind
  a feature flag.

## Post-compaction reconstruction

After the summary is generated, the new context is rebuilt (in
`buildPostCompactMessages()` order: boundary marker → summary → any preserved messages
→ attachments → hook results):

- **Compact boundary marker** — a `SystemCompactBoundaryMessage` recording
  pre-compaction token count, trigger (`auto`/`manual`), and metadata used to relink
  preserved message chains on disk.
- **File attachments** (`createPostCompactFileAttachments`) — re-reads up to 5 most
  recently accessed files (skipping any already visible in preserved messages) within a
  50,000-token budget, so the model doesn't have to re-`Read` them.
- **Plan / plan-mode attachments** — preserves the active plan file and, if the user was
  in plan mode, an instruction to remain in plan mode.
- **Skill attachments** (`createSkillAttachmentIfNeeded`) — re-injects content of any
  skills invoked this session, truncated per-skill (5,000 tokens) within an overall
  25,000-token budget, most-recently-invoked first.
- **Tool/agent/MCP "delta" attachments** — re-announces tool and MCP listings that were
  previously communicated via now-discarded messages.
- **SessionStart hooks** re-run (as after a fresh session) to restore CLAUDE.md/memory
  context.
- **`postCompactCleanup.ts`** clears a grab-bag of module-level caches and tracking
  state that's now stale: microcompact/cache-edit state, system-prompt-section cache,
  classifier approvals, speculative permission checks, `getUserContext`/memory-file
  caches (main thread only — subagents share process state and must not clobber it),
  telemetry tracing state, file-content cache for commit attribution. Explicitly does
  *not* clear invoked-skill tracking (needed for future compactions' skill attachments)
  or `sentSkillNames` (re-sending the full skill listing is expensive and low-value).

## Everything is logged

Every path logs analytics events (`tengu_compact`, `tengu_partial_compact`,
`tengu_cached_microcompact`, `tengu_time_based_microcompact`, `tengu_sm_compact_*`,
`tengu_compact_failed`, `tengu_compact_ptl_retry`, etc.) with token counts, cache-sharing
success/failure, and failure reasons — used to tune the thresholds and diagnose
recompaction loops (`RecompactionInfo` tracks whether this is a repeat compaction in the
same turn chain).

## Key files

| File | Role |
|---|---|
| `src/commands/compact/compact.ts` | `/compact` slash command entry point |
| `src/services/compact/autoCompact.ts` | Threshold calculation + automatic triggering |
| `src/services/compact/compact.ts` | Core summarization + full/partial compaction logic |
| `src/services/compact/prompt.ts` | Summarization prompt templates + formatting |
| `src/services/compact/microCompact.ts` | Time-based and cached tool-result clearing |
| `src/services/compact/cachedMicrocompact.ts` | Cache-editing state for cached microcompact |
| `src/services/compact/sessionMemoryCompact.ts` | Session-memory-based compaction path (consumer) |
| `src/services/SessionMemory/sessionMemory.ts` | Background extraction that writes the session-memory file (producer) |
| `src/services/SessionMemory/sessionMemoryUtils.ts` | Shared thresholds/config/cursor state for session memory |
| `src/services/SessionMemory/prompts.ts` | Extraction prompt + template + token-budget truncation for the memory file |
| `src/services/compact/grouping.ts` | Groups messages by API round (for PTL-retry truncation) |
| `src/services/compact/postCompactCleanup.ts` | Clears stale caches/state after compaction |
| `src/services/compact/timeBasedMCConfig.ts` | Config for time-based microcompact |
| `src/services/compact/compactWarningHook.ts` / `compactWarningState.ts` | Drives the "context left" UI warning |
| `src/services/compact/reactiveCompact.ts` | Experimental reactive (on-error) compaction |
| `src/components/CompactBoundaryMessage.tsx` | Renders the boundary marker in the transcript |
| `src/components/CompactSummary.tsx` | Renders the summary in the transcript |
| `src/components/TokenWarning.tsx` | "Context left until auto-compact" indicator |

## Notes / open questions

- Several mechanisms (`CACHED_MICROCOMPACT`, `REACTIVE_COMPACT`, `CONTEXT_COLLAPSE`)
  are gated behind feature flags or GrowthBook experiment values and may not be
  active in a given build — see `FEATURES.md` for flag status.
- Session memory compaction has two *independently* gated halves (background
  extraction vs. compaction-time consumption) — see
  [Session memory compaction](#session-memory-compaction). Enabling one without the
  other silently no-ops; this is easy to get wrong on a self-hosted deployment
  without GrowthBook connectivity.
- `reactiveCompact.ts` itself wasn't read in detail for this doc — worth a follow-up
  pass if the reactive-only path needs to be understood precisely.
