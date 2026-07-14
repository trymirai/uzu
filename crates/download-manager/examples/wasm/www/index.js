const STATE_NOT_DOWNLOADED = "not_downloaded";
const STATE_DOWNLOADING = "downloading";
const STATE_PAUSED = "paused";
const STATE_DOWNLOADED = "downloaded";
const STATE_LOCKED = "locked";
const STATE_ERROR = "error";

const downloadButton = document.getElementById("btn-download");
const urlInput = document.getElementById("input-url");
const pathInput = document.getElementById("input-path");
const textError = document.getElementById("text-error");
const textState = document.getElementById("text-state");

function updateDownloadButton(state) {
  switch (state) {
    case STATE_DOWNLOADING:
      downloadButton.textContent = "Pause"
      break;
    default:
      downloadButton.textContent = "Download";
      break;
  }
}

function onStateChanged(state) {
  downloadTaskId = state.task_id;
  downloadState = state.phase;

  let content = state.phase;
  let buttonText = "";
  switch (state.phase) {
    case STATE_DOWNLOADING:
      content = content + ": " + state.downloaded_bytes + "/" + state.total_bytes;
      buttonText = "Pause"
      break;
    case STATE_PAUSED:
      content = content + ": " + state.downloaded_bytes + "/" + state.total_bytes;
      buttonText = "Resume"
      break;
    case STATE_ERROR:
      content = content + (state.message !== undefined ? " " + state.message : "");
      buttonText = "Download";
      break;
    default:
      buttonText = "Download";
      break;
  }
  textState.textContent = content;
  downloadButton.textContent = buttonText;
}

async function download() {
  downloadTaskId = null;
  const url = urlInput.value;
  const filePath = pathInput.value;

  try {
    await wasmBindings.download(url, filePath, onStateChanged);
  } catch (error) {
    textError.textContent = "Error: " + error;
    throw error;
  }
}

async function pause() {
  const taskId = downloadTaskId;
  if (taskId != null) {
    await wasmBindings.pause(taskId);
  }
}

async function resume() {
  const taskId = downloadTaskId;
  if (taskId != null) {
    await wasmBindings.resume(taskId);
  }
}

let downloadState = STATE_NOT_DOWNLOADED;
let downloadTaskId = null;

addEventListener("TrunkApplicationStarted", () => {
  updateDownloadButton(downloadState);
  downloadButton.addEventListener("click", async (event) => {
    textError.textContent = null;
    console.log("STATE: " + downloadState);
    switch (downloadState) {
      case STATE_DOWNLOADING:
        await pause();
        break;
      case STATE_PAUSED:
        await resume();
        break;
      default:
        await download();
        break;
    }
  })
});
