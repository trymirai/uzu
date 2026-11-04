"""One fail-closed Metal acceptance of the existing SystemOne endpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from fixture import digest, validate

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def run(command, env, output, cwd=ROOT):
    with output.open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
            require(code == 0, f"command exited {code}: {command}; see {output}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    return output.read_text()


def request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    call = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(call, timeout=45)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response), {name.lower(): value for name, value in response.headers.items()}


def check_result(status, result, model, questions):
    require(status == 200, (status, result))
    require(result["model"] == model, result)
    require(list(result["answers"]) == list(questions), result)
    require(result["usage"]["output_tokens"] == 0 and result["usage"]["input_tokens"] > 0, result)
    for name, question in questions.items():
        answer = result["answers"][name]
        require(answer["type"] == question["type"], answer)
        if answer["type"] == "noul":
            require(math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1, answer)
            continue
        probabilities = answer["probabilities"]
        values = list(probabilities.values())
        require(all(math.isfinite(p) and 0 <= p <= 1 for p in values), answer)
        require(abs(sum(values) - 1) < 1e-9, answer)
        if answer["type"] == "choice":
            require(list(probabilities) == list(question["criteria"]), answer)
            require(answer["choice"] == max(probabilities, key=probabilities.get), answer)
        else:
            require(list(probabilities) == [str(i) for i in range(len(question["criteria"]))], answer)
            require(abs(answer["score"] - sum(i * p for i, p in enumerate(values))) < 1e-9, answer)
        entropy = -sum(p * math.log(p) for p in values if p > 0)
        expected = 1 if len(values) == 1 else max(0, min(1, 1 - entropy / math.log(len(values))))
        require(abs(answer["confidence"] - expected) < 1e-9, answer)


def exercise(base, model, observations):
    questions = {
        "urgent": {"type": "noul", "instructions": "Does the customer ask for action today?"},
        "category": {"type": "choice", "instructions": "Classify this ticket.",
                     "criteria": {"billing": "Payments and refunds", "technical": "Technical support", "other": None}},
        "priority": {"type": "score", "instructions": "Rate urgency.",
                     "criteria": ["Routine", "This week", "Today"]},
    }
    sample = {"model": model, "state": "I was charged twice. Please refund me today.", "questions": questions}
    chat = {"model": model, "messages": [{"role": "user", "content": "Reply with one short sentence about the moon."}],
            "temperature": 0, "reasoning_effort": "disabled", "max_tokens": 64, "stream": False}
    status, before, _ = request(base, "/v1/chat/completions", chat)
    require(status == 200 and before["choices"][0]["finish_reason"] == "stop", (status, before))
    continuation = dict(chat, messages=chat["messages"] + [
        {"role": "assistant", "content": before["choices"][0]["message"]["content"]},
        {"role": "user", "content": "Now reply with one short sentence about the sun."},
    ])
    status, baseline, _ = request(base, "/v1/chat/completions", continuation)
    require(status == 200 and baseline["choices"], (status, baseline))
    baseline_cached = baseline["usage"]["prompt_tokens_details"]["cached_tokens"]
    require(baseline_cached > 0, "control continuation did not exercise the cached chat state")
    # Repeating C resets the completed C-D chain. Its next request strictly extends C.
    status, replay, _ = request(base, "/v1/chat/completions", chat)
    require(status == 200 and before["choices"] == replay["choices"], (before, replay))
    status, first, _ = request(base, "/v1/systemone", sample)
    check_result(status, first, model, questions)
    other = {"state": "A router cannot connect to the network.", "questions": {"q": questions["category"]}}
    status, second, _ = request(base, "/v1/systemone", other)
    check_result(status, second, model, other["questions"])
    status, again, _ = request(base, "/v1/systemone", sample)
    check_result(status, again, model, questions)
    require(first["answers"] == again["answers"], "SystemOne A-B-A changed A")
    status, after, _ = request(base, "/v1/chat/completions", continuation)
    require(status == 200 and baseline["choices"] == after["choices"], (baseline, after))
    after_cached = after["usage"]["prompt_tokens_details"]["cached_tokens"]
    require(after_cached == baseline_cached, "scoring changed cached chat continuation")
    observations.append({"name": "mixed-types-ABA-and-greedy-chat-isolation", "status": "PASS",
                         "cached_tokens": after_cached, "response": first})

    boundaries = {
        "letters": {"type": "choice", "instructions": "Choose a label.",
                    "criteria": {f"option{i}": None for i in range(26)}},
        "digits": {"type": "score", "instructions": "Rate the input.", "criteria": [str(i) for i in range(10)]},
        "binary": questions["urgent"],
    }
    status, result, _ = request(base, "/v1/systemone", {"state": "A test input.", "questions": boundaries})
    check_result(status, result, model, boundaries)
    observations.append({"name": "all-contextual-proxies-A-Z-0-9-yes-no", "status": "PASS", "response": result})

    active = dict(chat, stream=True, max_tokens=256)
    active["messages"] = [{"role": "user", "content": "Write a long numbered list of positive integers."}]
    call = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(active).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(call, timeout=45) as stream:
        require(stream.status == 200, stream.status)
        status, result, _ = request(base, "/v1/systemone", sample)
        check_result(status, result, model, questions)
        rest = stream.read()
        require(b"[DONE]" in rest and b'"error"' not in rest, rest)
    observations.append({"name": "chat-and-decisions-share-instance", "status": 200})


def terminate(_signal, _frame):
    raise SystemExit("terminated")


def source_fingerprint():
    paths = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                                     "--", "Cargo.toml", "Cargo.lock", "crates", "scripts", "docs"], cwd=ROOT).decode().split("\0")
    return {path: digest(ROOT / path) for path in sorted(set(paths))
            if path and (ROOT / path).is_file() and "__pycache__" not in path and "node_modules" not in path}


def main():
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=["metal"], default="metal")
    parser.add_argument("--output", type=Path, required=True, help="new directory for logs and result.json")
    args = parser.parse_args()
    model_path = args.model.resolve()
    manifest = validate(model_path)
    require(platform.system() == "Darwin" and platform.machine() == "arm64", "acceptance requires Apple Silicon Metal")
    for command in ("cargo", "node", "npm"):
        require(shutil.which(command), f"required tool missing: {command}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "FAIL", "model": manifest["repository"], "revision": manifest["revision"],
               "backend": "metal", "observations": [],
               "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
               "source_files_sha256": source_fingerprint()}
    try:
        env = dict(os.environ, UZU_SCORE_MODEL=str(model_path))
        log = run(["cargo", "+nightly", "test", "--locked", "-p", "uzu-engine", "--lib",
                   "native_metal_model_scores_match_independent_full_decoder_readout", "--", "--ignored", "--nocapture"],
                  env, output / "native.log")
        require(re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", log), "native test did not run exactly once")
        run(["cargo", "+nightly", "test", "--locked", "-p", "cli", "-p", "nagare"], env, output / "application.log")
        run(["cargo", "+nightly", "build", "--locked", "-p", "cli"], env, output / "build.log")
        metadata = subprocess.check_output(["cargo", "+nightly", "metadata", "--locked", "--no-deps", "--format-version=1"], cwd=ROOT, env=env)
        binary = Path(json.loads(metadata)["target_directory"]) / "debug" / "cli"
        summary["binary_sha256"] = digest(binary)
        with tempfile.TemporaryDirectory(prefix="uzu-systemone-") as directory:
            temporary = Path(directory)
            sdk = temporary / "sdk"
            sdk.mkdir()
            for name in ("package.json", "package-lock.json"):
                shutil.copyfile(HERE / name, sdk / name)
            shutil.copyfile(ROOT / "scripts/systemone-smoke.mjs", sdk / "smoke.mjs")
            run(["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"], env, output / "sdk-install.log", cwd=sdk)
            with socket.socket() as reserve:
                reserve.bind(("127.0.0.1", 0))
                port = reserve.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            for name in ("config", "cache", "storage"):
                (temporary / name).mkdir()
            env.update({"XDG_CONFIG_HOME": str(temporary / "config"), "XDG_CACHE_HOME": str(temporary / "cache"),
                        "LOCAL_PATH": str(temporary / "storage"), "UZU_URL": base})
            with (output / "server.log").open("w") as server_log:
                server = subprocess.Popen([str(binary), "server", "--model", str(model_path), "--host", "127.0.0.1", "--port", str(port), "--prefix-cache"],
                                          cwd=ROOT, env=env, stdout=server_log, stderr=subprocess.STDOUT)
                try:
                    until = time.monotonic() + 120
                    while True:
                        require(server.poll() is None, "server exited before readiness; see server.log")
                        try:
                            status, models, _ = request(base, "/v1/models")
                            require(status == 200 and len(models["data"]) == 1, models)
                            model = models["data"][0]["id"]
                            break
                        except (urllib.error.URLError, ConnectionError):
                            require(time.monotonic() < until, "server readiness timeout")
                            time.sleep(0.2)
                    exercise(base, model, summary["observations"])
                    env["UZU_MODEL"] = model
                    run(["node", str(sdk / "smoke.mjs")], env, output / "sdk.log")
                    require(digest(binary) == summary["binary_sha256"], "binary changed during acceptance")
                    require(source_fingerprint() == summary["source_files_sha256"], "source changed during acceptance")
                finally:
                    if server.poll() is None:
                        server.terminate()
                    try:
                        server.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
                    summary["server_cleanup"] = {"pid": server.pid, "returncode": server.returncode}
        summary["status"] = "PASS"
    except BaseException as error:
        summary["error"] = str(error)
        raise
    finally:
        destination = output / "result.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(destination)
    print(json.dumps({"status": "PASS", "evidence": str(output)}))


if __name__ == "__main__":
    main()
