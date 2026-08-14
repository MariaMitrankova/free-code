/**
 * Prompt for the parallel-compaction path (see parallelCompact.ts).
 *
 * Rewrite of the standard compact prompt (prompt.ts's BASE_COMPACT_PROMPT)
 * analogous to the block-partitioned scheme in the Parallel Compaction paper
 * (https://arxiv.org/html/2605.23296): instead of one call summarizing the
 * whole conversation, N independent workers each summarize ONE contiguous
 * block, dispatched concurrently. Each worker sees blocks 1..k as context
 * (so it can resolve references like "that file" or "the bug from earlier")
 * but is instructed to summarize ONLY the block wrapped in
 * <TARGET_BLOCK>...</TARGET_BLOCK> — everything before that marker is being
 * summarized independently by other workers and must not be re-summarized
 * here, or the merged result duplicates content across blocks.
 */

import { NO_TOOLS_PREAMBLE, NO_TOOLS_TRAILER } from './prompt.js'

const TARGET_BLOCK_ANALYSIS_INSTRUCTION = `Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message INSIDE the <TARGET_BLOCK>...</TARGET_BLOCK> markers only. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Everything BEFORE the <TARGET_BLOCK> marker is earlier conversation history, included only so you can resolve references (e.g. "that file", "the bug we found earlier", "as I mentioned before"). It is being summarized independently by other workers running in parallel — do NOT summarize it yourself, and do NOT repeat information from it in your output unless it's necessary to make your summary of the target block self-contained.
3. Double-check for technical accuracy and completeness, addressing each required element thoroughly.`

const MIDDLE_BLOCK_SECTIONS = `1. Primary Request and Intent: Capture the user's explicit requests and intents that occur within the target block.
2. Key Technical Concepts: List technical concepts, technologies, and frameworks discussed within the target block.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created within the target block. Include full code snippets where applicable and a summary of why each read or edit is important.
4. Errors and fixes: List errors encountered within the target block and how they were fixed.
5. Problem Solving: Document problems solved and any ongoing troubleshooting within the target block.
6. All user messages: List ALL user messages within the target block that are not tool results.
7. Pending Tasks: Outline any pending tasks mentioned within the target block.`

const FINAL_BLOCK_SECTIONS = `1. Primary Request and Intent: Capture the user's explicit requests and intents that occur within the target block.
2. Key Technical Concepts: List technical concepts, technologies, and frameworks discussed within the target block.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created within the target block. Include full code snippets where applicable and a summary of why each read or edit is important.
4. Errors and fixes: List errors encountered within the target block and how they were fixed.
5. Problem Solving: Document problems solved and any ongoing troubleshooting within the target block.
6. All user messages: List ALL user messages within the target block that are not tool results.
7. Pending Tasks: Outline any pending tasks mentioned within the target block.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request — this is the LAST block, so this describes the frontier of the whole conversation. Pay special attention to the most recent messages. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure this step is DIRECTLY in line with the user's most recent explicit requests. Include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off.`

function buildParallelBlockCompactPrompt(
  blockIndex: number,
  totalBlocks: number,
  isFinalBlock: boolean,
): string {
  const sections = isFinalBlock ? FINAL_BLOCK_SECTIONS : MIDDLE_BLOCK_SECTIONS

  return `Your task is to create a detailed summary of ONE block of a larger conversation that has been split into ${totalBlocks} contiguous block(s) for parallel processing. You are processing block ${blockIndex} of ${totalBlocks}.

The block you must summarize is wrapped in <TARGET_BLOCK>...</TARGET_BLOCK> markers below, in the conversation above this instruction. Any conversation content before the <TARGET_BLOCK> marker is earlier history provided only as context to help you understand references made inside the target block — it is being summarized by other workers running in parallel right now, so do NOT summarize it, and do not worry about continuity with blocks after this one (a later worker handles that).

${TARGET_BLOCK_ANALYSIS_INSTRUCTION}

Your summary should include the following sections, covering ONLY what happens inside the target block:

${sections}

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Important Code Snippet]

4. Errors and fixes:
    - [Error description]:
      - [How you fixed it]

5. Problem Solving:
   [Description]

6. All user messages:
    - [Detailed non tool use user message]

7. Pending Tasks:
   - [Task 1]
${
  isFinalBlock
    ? `
8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]
`
    : ''
}</summary>
</example>

Please provide your summary of ONLY the <TARGET_BLOCK> content, following this structure and ensuring precision and thoroughness in your response.
`
}

/**
 * Builds the compact-instruction prompt sent to a single parallel-compaction
 * worker summarizing block `blockIndex` (1-indexed) of `totalBlocks`.
 */
export function getParallelBlockCompactPrompt(
  blockIndex: number,
  totalBlocks: number,
  customInstructions?: string,
): string {
  const isFinalBlock = blockIndex === totalBlocks
  let prompt =
    NO_TOOLS_PREAMBLE +
    buildParallelBlockCompactPrompt(blockIndex, totalBlocks, isFinalBlock)

  if (customInstructions && customInstructions.trim() !== '') {
    prompt += `\n\nAdditional Instructions:\n${customInstructions}`
  }

  prompt += NO_TOOLS_TRAILER

  return prompt
}
