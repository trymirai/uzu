from pathlib import Path

import requests
from rich.progress import Progress


def download_file_with_resume(
    url: str, file_path: Path, progress: Progress, task_id
) -> None:
    part_file_path = file_path.with_suffix(file_path.suffix + ".part")

    resume_position = 0
    mode = "wb"
    if part_file_path.exists():
        resume_position = part_file_path.stat().st_size
        mode = "ab"

    headers = {}
    if resume_position > 0:
        headers["Range"] = f"bytes={resume_position}-"

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    if resume_position > 0 and response.status_code == 206:
        total_size += resume_position

    progress.update(task_id, total=total_size, completed=resume_position)

    with open(part_file_path, mode) as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress.update(task_id, advance=len(chunk))

    part_file_path.rename(file_path)
