"""Check external engine pins and lockfiles against the latest stable releases.

Uzu uses the local Cargo workspace source, so it has no external version pin.
These checks need network access but do not build engines or download models.
"""

import json
import os
import re
import shlex
import tomllib
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

import pytest
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.engine_versions


def read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def fetch_json(url: str) -> dict:
    headers = {"Accept": "application/json", "User-Agent": "uzu-benchmark-version-tests"}
    if url.startswith("https://api.github.com/") and (token := os.environ.get("GITHUB_TOKEN")):
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=15) as response:
            return json.load(response)
    except (OSError, ValueError) as error:
        pytest.fail(f"Cannot check latest engine version from {url}: {error}", pytrace=False)


def latest_pypi_version(package: str) -> Version:
    data = fetch_json(f"https://pypi.org/pypi/{package}/json")
    versions = []
    for name, files in data["releases"].items():
        try:
            version = Version(name)
        except InvalidVersion:
            continue
        if not version.is_prerelease and not version.is_devrelease and any(not file["yanked"] for file in files):
            versions.append(version)
    assert versions, f"{package}: PyPI returned no stable, non-yanked releases"
    return max(versions)


def latest_github_tag(repository: str) -> str:
    release = fetch_json(f"https://api.github.com/repos/{repository}/releases/latest")
    assert not release["draft"] and not release["prerelease"], f"{repository}: expected a stable release"
    return release["tag_name"]


def locked_package(project: Path, package: str) -> dict:
    matches = [entry for entry in read_toml(project / "uv.lock")["package"] if entry["name"] == package]
    assert len(matches) == 1, f"{project.name}/uv.lock: expected exactly one {package} package"
    return matches[0]


@pytest.mark.parametrize("engine,package", [("mlx", "mlx"), ("mlx", "mlx-lm"), ("mtplx", "mtplx")])
def test_pypi_engine_uses_latest_version(engine: str, package: str) -> None:
    project = ROOT / f"engine-{engine}"
    dependencies = [Requirement(value) for value in read_toml(project / "pyproject.toml")["project"]["dependencies"]]
    requirements = [requirement for requirement in dependencies if requirement.name == package]
    assert len(requirements) == 1, f"{project.name}/pyproject.toml: expected a dependency on {package}"
    latest = latest_pypi_version(package)
    requirement = requirements[0]
    assert latest in requirement.specifier, (
        f"{project.name}/pyproject.toml: {requirement} excludes the latest stable release {latest}"
    )
    locked = locked_package(project, package)
    assert locked["source"].get("registry") == "https://pypi.org/simple", (
        f"{project.name}/uv.lock: {package} no longer comes from PyPI; update this version check"
    )
    assert Version(locked["version"]) == latest, (
        f"{project.name}/uv.lock: {package} is {locked['version']}, latest stable release is {latest}. "
        f"Update {project.name}/pyproject.toml and run uv lock --project {project.name} --upgrade-package {package}"
    )


def test_llamacpp_uses_latest_version() -> None:
    path = ROOT / "engine-llamacpp/CMakeLists.txt"
    source = re.sub(r"#[^\n]*", "", path.read_text())
    declaration = re.search(r"FetchContent_Declare\(\s*llama_cpp\b([^)]*)\)", source)
    assert declaration, f"{path.name}: missing llama_cpp declaration"
    tag = re.search(r"\bGIT_TAG\s+([^\s)]+)", declaration[1])
    assert tag, f"{path.name}: missing llama_cpp GIT_TAG"
    assert tag[1].strip('"') == "${LLAMA_VERSION}", f"{path.name}: llama_cpp GIT_TAG must use LLAMA_VERSION"
    version = re.search(r"\bset\(\s*LLAMA_VERSION\s+([^\s)]+)\s*\)", source)
    assert version, f"{path.name}: missing LLAMA_VERSION"
    current = version[1].strip('"')
    latest = latest_github_tag("ggml-org/llama.cpp")
    assert current == latest, f"engine-llamacpp/CMakeLists.txt: LLAMA_VERSION is {current}, latest release is {latest}"


def test_mlxserve_uses_latest_version() -> None:
    project = ROOT / "engine-mlxserve"
    source = (project / "bootstrap.sh").read_text()
    clone = re.search(r"git clone --depth 1 --branch (\S+) https://github.com/ddalcu/mlx-serve.git", source)
    assert clone, "engine-mlxserve/bootstrap.sh: expected a pinned source clone"
    assert clone[1] == '"$VERSION"', "engine-mlxserve/bootstrap.sh: --branch must use VERSION"
    version = re.search(r'^VERSION="([^"]+)"$', source, re.MULTILINE)
    assert version, "engine-mlxserve/bootstrap.sh: missing VERSION"
    current = version[1]
    latest = latest_github_tag("ddalcu/mlx-serve")
    assert current == latest, f"mlx-serve is {current}, latest release is {latest}"


def test_omlx_uses_latest_version() -> None:
    project = ROOT / "engine-omlx"
    source = read_toml(project / "pyproject.toml")["tool"]["uv"]["sources"]["omlx"]
    latest = latest_github_tag("jundot/omlx")
    assert source.get("tag") == latest, (
        f"engine-omlx/pyproject.toml: omlx tag is {source.get('tag')}, latest release is {latest}"
    )
    locked = locked_package(project, "omlx")
    locked_tag = parse_qs(urlsplit(locked["source"].get("git", "")).query).get("tag")
    assert locked_tag == [latest] and Version(locked["version"]) == Version(latest), (
        f"engine-omlx/uv.lock: omlx is {locked['version']} from {locked['source']}, latest release is {latest}. "
        "Run uv lock --project engine-omlx --upgrade-package omlx"
    )


def test_splash_uses_latest_version() -> None:
    path = ROOT / "engine-splash/bootstrap.sh"
    source = path.read_text().replace("\\\n", " ")
    clones = [shlex.split(line, comments=True) for line in source.splitlines() if re.match(r"\s*git\s+clone\s", line)]
    assert len(clones) == 1 and "--branch" in clones[0], f"{path.name}: expected a git clone with a pinned --branch"
    command = clones[0]
    current = command[command.index("--branch") + 1]
    latest = latest_github_tag("incoai/splash")
    assert current == latest, f"engine-splash/bootstrap.sh: --branch is {current}, latest release is {latest}"
