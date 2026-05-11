const imageInput = document.getElementById("imageInput");
const dropzone = document.getElementById("dropzone");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const fileName = document.getElementById("fileName");
const resultLabel = document.getElementById("resultLabel");
const diseaseName = document.getElementById("diseaseName");
const resultNote = document.getElementById("resultNote");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceValue = document.getElementById("confidenceValue");
const statusPill = document.getElementById("statusPill");

let selectedFile = null;

function setResult(label, confidence, note, status, pillClass) {
    resultLabel.textContent = label;
    diseaseName.textContent = label;
    resultNote.textContent = note;
    confidenceFill.style.width = `${confidence}%`;
    confidenceValue.textContent = `${confidence}%`;
    statusPill.textContent = status;
    statusPill.className = `status-pill ${pillClass}`;
}

function handleFile(file) {
    if (!file) return;

    selectedFile = file;
    fileName.textContent = file.name;
    showPreview(file);
    setResult("Ready to analyze", 0, "Click Analyze Image to generate a prediction.", "Ready", "neutral");
}

function clearState() {
    selectedFile = null;
    imageInput.value = "";
    previewImage.src = "";
    previewImage.style.display = "none";
    previewPlaceholder.style.display = "grid";
    fileName.textContent = "No file selected";
    setResult("Waiting for image", 0, "Upload a leaf image to analyze it.", "Idle", "neutral");
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = () => {
        previewImage.src = reader.result;
        previewImage.style.display = "block";
        previewPlaceholder.style.display = "none";
    };
    reader.readAsDataURL(file);
}

function getDemoPrediction(file) {
    const tokens = [
        "Apple___Apple_scab",
        "Tomato___Late_blight",
        "Potato___Early_blight",
        "Corn___Common_rust",
        "Grape___Black_rot",
        "Pepper_bell___Bacterial_spot",
    ];
    const index = (file.name.length + file.size) % tokens.length;
    const confidence = 62 + ((file.size + file.name.length) % 28);
    return {
        label: tokens[index],
        confidence,
    };
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files && imageInput.files[0];
    handleFile(file);
});

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        setResult("No image selected", 0, "Choose a leaf image first.", "Missing image", "neutral");
        return;
    }

    analyzeBtn.textContent = "Analyzing...";
    analyzeBtn.disabled = true;

    try {
        const demo = getDemoPrediction(selectedFile);
        setResult(
            demo.label,
            demo.confidence,
            "This UI is ready to connect to your Python prediction endpoint. Replace the demo logic with a fetch request when the API is available.",
            demo.confidence > 85 ? "High confidence" : "Sample result",
            demo.confidence > 85 ? "good" : "neutral"
        );
    } finally {
        analyzeBtn.textContent = "Analyze Image";
        analyzeBtn.disabled = false;
    }
});

resetBtn.addEventListener("click", clearState);

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
    });
});

dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (!file) return;
    handleFile(file);
});

clearState();
