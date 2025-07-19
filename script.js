const intro = document.getElementById("intro");
const main = document.getElementById("main");
const ideas = document.getElementById("ideas");
const recordBtn = document.getElementById("recordBtn");
const viewIdeasBtn = document.getElementById("viewIdeas");
const recordingsList = document.getElementById("recordingsList");

let isRecording = false;
let mediaRecorder;
let chunks = [];

// Automatically switch to main screen after intro
setTimeout(() => {
  intro.classList.add("hidden");
  main.classList.remove("hidden");
}, 5000);

// Handle recording
recordBtn.addEventListener("click", async () => {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      chunks = [];

      mediaRecorder.ondataavailable = e => chunks.push(e.data);

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        saveRecording(url);
      };

      mediaRecorder.start();
      isRecording = true;
      recordBtn.textContent = "Stop Recording";
    } catch (err) {
      alert("Microphone permission is required.");
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    recordBtn.textContent = "Start Recording";
  }
});

// Save recording to localStorage
function saveRecording(url) {
  const saved = JSON.parse(localStorage.getItem("recordings") || "[]");
  saved.push(url);
  localStorage.setItem("recordings", JSON.stringify(saved));
}

// View saved recordings
viewIdeasBtn.addEventListener("click", () => {
  main.classList.add("hidden");
  ideas.classList.remove("hidden");
  showRecordings();
});

function showRecordings() {
  recordingsList.innerHTML = "";
  const saved = JSON.parse(localStorage.getItem("recordings") || "[]");
  if (saved.length === 0) {
    recordingsList.innerHTML = "<p>No recordings yet.</p>";
    return;
  }

  saved.forEach((url, index) => {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = url;

    const div = document.createElement("div");
    div.appendChild(audio);
    recordingsList.appendChild(div);
  });
}

function goBack() {
  ideas.classList.add("hidden");
  main.classList.remove("hidden");
}
