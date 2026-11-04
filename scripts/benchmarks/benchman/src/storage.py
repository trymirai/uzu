import asyncio
import hashlib
import json
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

import zstandard
from benchman.task import TaskRecord, TaskRequest
from benchman.version import VERSION
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.auth.transport.requests import Request
from google.cloud.storage import Blob, Client
from prefect_gcp.cloud_storage import GcsBucket
from pydantic import TypeAdapter
from requests.adapters import HTTPAdapter

TASKS = TypeAdapter(dict[str, TaskRequest])
# How many threads read and write at once.
THREADS = 32


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Storage:
    """What benchman keeps in a work pool's storage.

    `store/` has what flow runs are given: source archives and tasks. `tasks/RUN/` has the task records of a
    run. `tags/NAME.json` has a tag's runs and `tags/NAME/RUN/` its copies of their task records.
    """

    def __init__(self, block: GcsBucket) -> None:
        credentials = block.gcp_credentials.get_credentials_from_service_account()
        # Before threads use them: each would get an access token of its own.
        credentials.refresh(Request())
        client = Client(credentials=credentials, project=block.gcp_credentials.project)
        # A connection for each thread; measured, they use one more.
        client._http.mount("https://", HTTPAdapter(pool_maxsize=THREADS + 1))
        self.block = block
        self.bucket = client.bucket(block.bucket)
        self.root = f"{PurePosixPath(block.bucket_folder) / 'benchman'}/v{VERSION}"
        self.tasks_written: set[str] = set()

    def blob(self, path: str) -> Blob:
        return self.bucket.blob(f"{self.root}/{path}")

    def generations(self, folder: str) -> dict[str, int]:
        """The files in the folder, with the generations of their objects."""
        blobs = self.bucket.list_blobs(prefix=f"{self.root}/{folder}/", delimiter="/")
        return {PurePosixPath(blob.name).name: blob.generation for blob in blobs}

    def write_bundle(self, bucket: str, key: str, bundle: Any) -> None:
        """Stores a flow run's bundle as a Google Cloud Storage upload step of its work pool would."""
        self.bucket.client.bucket(bucket).blob(key).upload_from_string(json.dumps(bundle))

    def sources(self, start: str) -> list[str]:
        """The hashes of the stored source archives that start so."""
        blobs = self.bucket.list_blobs(prefix=f"{self.root}/store/{start}")
        names = [PurePosixPath(blob.name).name for blob in blobs]
        return [name.removesuffix(".nar.zst") for name in names if name.endswith(".nar.zst")]

    def write_archive(self, nar_hash: str, archive: bytes) -> None:
        compressed = zstandard.ZstdCompressor(level=10).compress(archive)
        self.blob(f"store/{nar_hash}.nar.zst").upload_from_string(compressed)

    def read_archive(self, nar_hash: str) -> bytes:
        return zstandard.ZstdDecompressor().decompress(self.blob(f"store/{nar_hash}.nar.zst").download_as_bytes())

    def write_tasks(self, tasks: dict[str, TaskRequest]) -> str:
        """Stores the tasks, which are too large for flow run parameters, and returns their key."""
        content = TASKS.dump_json(tasks)
        key = hashlib.sha256(content).hexdigest()
        if key not in self.tasks_written:
            self.blob(f"store/{key}.json").upload_from_string(content)
            self.tasks_written.add(key)
        return key

    def read_tasks(self, key: str) -> dict[str, TaskRequest]:
        return TASKS.validate_json(self.blob(f"store/{key}.json").download_as_bytes())

    def tags(self) -> list[str]:
        return sorted(PurePosixPath(file).stem for file in self.generations("tags"))

    def read_tag(self, name: str) -> bytes:
        """The runs that the tag is of, as saved. Raises LookupError if there is no such tag."""
        try:
            return self.blob(f"tags/{name}.json").download_as_bytes()
        except NotFound:
            raise LookupError(name) from None

    def write_tag(self, name: str, runs: bytes, records: Iterable["Records"]) -> None:
        """Makes the tag one of these runs, and removes its copies of other tasks' records."""
        kept = {f"{self.root}/{each.folder(name)}/{file}" for each in records for file in each.files.values()}
        for blob in self.bucket.list_blobs(prefix=f"{self.root}/tags/{name}/"):
            if blob.name not in kept:
                blob.delete()
        self.blob(f"tags/{name}.json").upload_from_string(runs)


class Records:
    """A run's task records, and their copies in tags."""

    def __init__(self, storage: Storage, run: str, tasks: dict[str, TaskRequest]) -> None:
        self.storage = storage
        self.run = run
        self.files = {name: f"{digest(request.model_dump(mode='json'))}.json" for name, request in tasks.items()}
        # The records as last read by `refresh` or written.
        self.stored: dict[str, TaskRecord] = {}

    def folder(self, tag: str | None = None) -> str:
        return f"tags/{tag}/{self.run}" if tag else f"tasks/{self.run}"

    def read_file(self, file: str, tag: str | None = None) -> TaskRecord | None:
        blob = self.storage.blob(f"{self.folder(tag)}/{file}")
        try:
            content = blob.download_as_bytes()
        except NotFound:
            return None
        return TaskRecord.model_validate_json(content).model_copy(update={"generation": blob.generation})

    def read(self, name: str) -> TaskRecord | None:
        return self.read_file(self.files[name])

    async def read_all(self, tag: str | None, known: dict[str, TaskRecord]) -> dict[str, TaskRecord]:
        """The stored records, or a tag's copies of them. A known one is read only if it changed since."""
        generations = await asyncio.to_thread(self.storage.generations, self.folder(tag))

        async def read(name: str) -> TaskRecord | None:
            if name in known and known[name].generation == generations[self.files[name]]:
                return known[name]
            return await asyncio.to_thread(self.read_file, self.files[name], tag)

        names = [name for name, file in self.files.items() if file in generations]
        records = await asyncio.gather(*map(read, names))
        return {name: record for name, record in zip(names, records, strict=True) if record is not None}

    async def refresh(self) -> None:
        self.stored = await self.read_all(None, self.stored)

    def others(self) -> list[TaskRecord]:
        """The records in the run's folder of other tasks than these."""
        files = self.storage.generations(self.folder()).keys() - self.files.values()
        return [record for file in files if (record := self.read_file(file)) is not None]

    def write(self, name: str, record: TaskRecord, over: TaskRecord | None) -> bool:
        """Stores the record over the one that was read, None for none. False if another was stored since."""
        blob = self.storage.blob(f"{self.folder()}/{self.files[name]}")
        try:
            blob.upload_from_string(record.model_dump_json(), if_generation_match=over.generation if over else 0)
        except PreconditionFailed:
            return False
        self.stored[name] = record.model_copy(update={"generation": blob.generation})
        return True

    def copy(self, name: str, record: TaskRecord, tag: str) -> None:
        self.storage.blob(f"{self.folder(tag)}/{self.files[name]}").upload_from_string(record.model_dump_json())

    def delete(self, name: str, record: TaskRecord) -> bool:
        """Removes the record that was read. False if another was stored since."""
        try:
            self.storage.blob(f"{self.folder()}/{self.files[name]}").delete(if_generation_match=record.generation)
        except (NotFound, PreconditionFailed):
            return False
        del self.stored[name]
        return True
