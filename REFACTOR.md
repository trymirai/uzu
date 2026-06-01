Required:
- suffix capacity hardcode in attention and short conv trie
- port tests

Verification:
- speculation, grammar, speculation+grammar examples
- bench against main

Improvements:
- api misuse hardening (mostly asserts)
  - mixers acceptance/context overflow
  - public api should return explicit errors for api misuse, not internal/panic
- consistent u32 token ids to get rid of token copy kernel

Followup:
- save/restore model state
- new tracing
- new tts
