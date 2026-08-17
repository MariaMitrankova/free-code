// Import-only smoke test for the new parallel-compaction modules — surfaces
// bun-resolver-specific issues that tsc wouldn't catch (this repo has a
// known history of bun vs tsc module-resolution divergence).
import * as parallelCompact from './src/services/compact/parallelCompact.js'
import * as parallelPrompt from './src/services/compact/parallelPrompt.js'
import * as compact from './src/services/compact/compact.js'

console.log('parallelCompact exports:', Object.keys(parallelCompact))
console.log('parallelPrompt exports:', Object.keys(parallelPrompt))
console.log(
  'compact.js still exports compactConversation:',
  typeof compact.compactConversation === 'function',
)
