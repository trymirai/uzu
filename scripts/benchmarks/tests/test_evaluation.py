"""Exercise the completion client against a local HTTP server without inference."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx2
import pytest
from openai import OpenAI
from typer.testing import CliRunner

from evaluation.main import app, send_request

runner = CliRunner()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = {
        "status": 200,
        "content_type": "application/json",
        "body": json.dumps({"choices": [{"message": {"content": "Hello, 世界"}}]}, ensure_ascii=False).encode(),
        "requests": [],
        "authorization": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, self.headers["Content-Type"], body))
            state["authorization"].append(self.headers.get("Authorization"))
            self.send_response(state["status"])
            self.send_header("Content-Type", state["content_type"])
            self.send_header("Content-Length", str(len(state["body"])))
            self.end_headers()
            self.wfile.write(state["body"])

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield ["--host", "127.0.0.1", "--port", str(httpd.server_port)], state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.fixture
def input_file(tmp_path):
    def write(payload=None):
        if payload is None:
            payload = {"messages": [{"role": "user", "content": "hello"}]}
        path = tmp_path / "request.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(path)

    return write


def streaming_body(chunks):
    metadata = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1, "model": "test-model"}
    events = [f"data: {json.dumps(metadata | chunk, ensure_ascii=False)}\n\n" for chunk in chunks]
    return (": keep-alive\n\n" + "".join(events) + "data: [DONE]\n\n").encode()


def test_chat_request(server, input_file):
    options, state = server
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.5,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result = runner.invoke(app, [*options, "--input", input_file(payload)])
    assert result.exit_code == 0, result.output
    assert state["requests"] == [("/v1/chat/completions", "application/json", payload)]
    assert json.loads(result.stdout) == json.loads(state["body"])
    assert "Hello, 世界" in result.stdout
    assert state["authorization"] == ["Bearer not-needed"]


def test_api_key(server, monkeypatch, input_file):
    options, state = server
    monkeypatch.setenv("OPENAI_API_KEY", "test-server-key")
    result = runner.invoke(app, [*options, "--input", input_file()])
    assert result.exit_code == 0, result.output
    assert state["authorization"] == ["Bearer test-server-key"]
    assert "test-server-key" not in result.output


def test_json_file(server, tmp_path):
    options, state = server
    payload = {"model": "test-model", "messages": [{"role": "user", "content": "世界"}]}
    path = tmp_path / "request with spaces.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    result = runner.invoke(app, [*options, "--input", str(path)])
    assert result.exit_code == 0, result.output
    assert state["requests"] == [("/v1/chat/completions", "application/json", payload)]


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_streaming_response(server, input_file, finish_reason):
    options, state = server
    state["content_type"] = "text/event-stream; charset=utf-8"
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    state["body"] = streaming_body(
        [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello, "}}]},
            {"choices": [{"index": 0, "delta": {"content": "世界"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]},
            {"choices": [], "usage": usage},
        ]
    )
    payload = {"messages": [{"role": "user", "content": "Hello"}], "stream": True}
    result = runner.invoke(app, [*options, "--input", input_file(payload)])
    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response["id"] == "chatcmpl-test"
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"] == "Hello, 世界"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert response["choices"][0]["finish_reason"] == finish_reason
    assert response["usage"] == usage
    assert state["requests"][0][2] == payload


def test_send_request_returns_json(server, capsys):
    options, state = server
    with OpenAI(base_url=f"http://127.0.0.1:{options[-1]}/v1", api_key="test-key") as client:
        response = send_request(client, {"messages": []})
    assert response == json.loads(state["body"])
    assert capsys.readouterr().out == ""


def test_send_request_collects_tool_calls(server, capsys):
    options, state = server
    state["content_type"] = "text/event-stream"
    tool_call = {
        "index": 0,
        "id": "call-1",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city":'},
    }
    state["body"] = streaming_body(
        [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [tool_call]}}]},
            {
                "choices": [
                    {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"London"}'}}]}}
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
    )
    with OpenAI(base_url=f"http://127.0.0.1:{options[-1]}/v1", api_key="test-key") as client:
        response = send_request(client, {"messages": [], "stream": True})
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["id"] == "call-1"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "weather"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"city":"London"}'
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("status", [302, 400, 503])
def test_http_error(server, status, input_file):
    options, state = server
    state["status"] = status
    state["body"] = b'{"error":{"message":"request failed"}}'
    result = runner.invoke(app, [*options, "--input", input_file()])
    assert result.exit_code == 1
    assert f"HTTP {status}" in result.stderr
    assert "request failed" in result.stderr
    assert not result.stdout
    assert len(state["requests"]) == 1


@pytest.mark.parametrize(
    "content_type, body",
    [
        ("application/json", b"invalid JSON"),
        ("application/json", b"[]"),
        ("application/json", b"null"),
        ("text/event-stream", b"data: invalid JSON\n\n"),
        ("text/event-stream", b"data: [DONE]\n\n"),
    ],
)
def test_invalid_response(server, input_file, content_type, body):
    options, state = server
    state["content_type"] = content_type
    state["body"] = body
    result = runner.invoke(app, [*options, "--input", input_file()])
    assert result.exit_code == 1
    assert "Invalid JSON response" in result.stderr


def test_connection_failure(monkeypatch, input_file):
    def fail(*_args, **_kwargs):
        raise httpx2.ConnectError("connection refused")

    monkeypatch.setattr(httpx2.Client, "send", fail)
    result = runner.invoke(app, ["--input", input_file()])
    assert result.exit_code == 1
    assert "Request failed: Connection error." in result.stderr


@pytest.mark.parametrize(
    "value, from_file",
    [
        ("[]", True),
        ("null", True),
        ("{}", True),
        ('{"messages":', True),
        ('{"messages":NaN}', True),
        ("missing-file.json", False),
        ('{"prompt":"hello"}', False),
    ],
)
def test_invalid_input(value, from_file, tmp_path):
    if from_file:
        path = tmp_path / "invalid.json"
        path.write_text(value, encoding="utf-8")
        value = str(path)
    result = runner.invoke(app, ["--input", value])
    assert result.exit_code == 2
    assert "--input" in result.output


@pytest.mark.parametrize("options", [["--port", "0"], ["--port", "65536"], ["--host", "http://localhost"]])
def test_invalid_address(options, input_file):
    result = runner.invoke(app, [*options, "--input", input_file()])
    assert result.exit_code == 2
