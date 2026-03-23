cd /Users/eugenebokhan/Developer/trymirai/uzu-worktrees/rht
cargo build --release -p cli
cat > /tmp/rht_trace_task.json << 'EOF'
{
    "greedy": true,
    "identifier": "rht-trace",
    "messages": [
        {"content": "Summarize", "role": "system"},
        {"content": "Large language models are AI systems designed to understand and generate human language at scale.", "role": "user"}
    ],
    "number_of_runs": 1,
    "repo_id": "LiquidAI/LFM2.5-1.2B-Instruct-RHT",
    "tokens_limit": 64
}
EOF
UZU_CAPTURE_FIRST_DECODE=1 \
  ./target/release/cli bench \
  models/experimental \
  /tmp/rht_trace_task.json \
  /tmp/rht_trace_result.json
# The .gputrace file will be created in the current directory
# Open it in Xcode:
open *.gputrace