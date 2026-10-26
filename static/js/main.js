document.addEventListener("DOMContentLoaded", () => {
    const analysisForm = document.getElementById("analysis-form");
    if (analysisForm) {
        const fileInput = document.getElementById("mri-image");
        const uploadZone = document.getElementById("upload-zone");
        const preview = document.getElementById("image-preview");
        const uploadTitle = document.getElementById("upload-title");
        const uploadDescription = document.getElementById("upload-description");
        const errorBox = document.getElementById("form-error");
        const analyzeButton = document.getElementById("analyze-button");
        const buttonLabel = document.getElementById("button-label");
        let previewUrl = null;

        const showFile = (file) => {
            if (!file) return;
            if (!["image/png", "image/jpeg"].includes(file.type) || file.size > 8 * 1024 * 1024) {
                errorBox.textContent = "Choose a PNG or JPG image smaller than 8 MB.";
                errorBox.classList.remove("hidden");
                fileInput.value = "";
                return;
            }
            if (previewUrl) URL.revokeObjectURL(previewUrl);
            previewUrl = URL.createObjectURL(file);
            preview.src = previewUrl;
            preview.classList.remove("hidden");
            uploadTitle.textContent = file.name;
            uploadDescription.textContent = "Click to choose a different image";
            errorBox.classList.add("hidden");
        };

        fileInput.addEventListener("change", () => showFile(fileInput.files[0]));
        uploadZone.addEventListener("dragover", (event) => {
            event.preventDefault();
            uploadZone.classList.add("is-dragging");
        });
        ["dragleave", "drop"].forEach((eventName) => {
            uploadZone.addEventListener(eventName, () => uploadZone.classList.remove("is-dragging"));
        });
        uploadZone.addEventListener("drop", (event) => {
            event.preventDefault();
            const [file] = event.dataTransfer.files;
            if (!file) return;
            const transfer = new DataTransfer();
            transfer.items.add(file);
            fileInput.files = transfer.files;
            showFile(file);
        });

        analysisForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            errorBox.classList.add("hidden");
            analyzeButton.disabled = true;
            buttonLabel.textContent = "Analyzing and saving…";
            try {
                const response = await fetch("/api/predict", {
                    method: "POST",
                    body: new FormData(analysisForm),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "The analysis could not be completed.");
                document.getElementById("empty-result").classList.add("hidden");
                document.getElementById("result-content").classList.remove("hidden");
                document.getElementById("result-class").textContent = data.predicted_class;
                document.getElementById("result-case").textContent = `${data.case_label} · ${data.model_used}`;
                document.getElementById("result-confidence").textContent = `${data.confidence}%`;
                document.getElementById("original-image").src = data.original_image_url;
                document.getElementById("gradcam-image").src = data.gradcam_image_url;
                document.getElementById("history-link").href = `${data.history_url}#case-${data.case_id}`;
                const bars = document.getElementById("probability-bars");
                bars.replaceChildren();
                Object.entries(data.probabilities).forEach(([label, value]) => {
                    const row = document.createElement("div");
                    row.className = "probability-row";
                    const name = document.createElement("span");
                    name.textContent = label;
                    const track = document.createElement("span");
                    track.className = "probability-track";
                    const fill = document.createElement("span");
                    fill.className = "probability-fill";
                    fill.style.width = `${value}%`;
                    track.append(fill);
                    const score = document.createElement("strong");
                    score.textContent = `${value}%`;
                    row.append(name, track, score);
                    bars.append(row);
                });
                document.getElementById("result-content").scrollIntoView({ behavior: "smooth", block: "start" });
            } catch (error) {
                errorBox.textContent = error.message;
                errorBox.classList.remove("hidden");
            } finally {
                analyzeButton.disabled = false;
                buttonLabel.textContent = "Analyze and save case";
            }
        });
    }

    document.querySelectorAll("[data-review-form]").forEach((form) => {
        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const card = form.closest("[data-case-id]");
            const message = form.querySelector(".review-message");
            const values = new FormData(form);
            try {
                const response = await fetch(`/api/cases/${card.dataset.caseId}/review`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        reviewed_class: values.get("reviewed_class"),
                        review_note: values.get("review_note"),
                    }),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "Could not save review.");
                message.textContent = "Review saved.";
                window.setTimeout(() => window.location.reload(), 350);
            } catch (error) {
                message.textContent = error.message;
            }
        });
    });

    document.querySelectorAll("[data-delete-case]").forEach((button) => {
        button.addEventListener("click", async () => {
            const card = button.closest("[data-case-id]");
            if (!window.confirm("Delete this case and its saved images?")) return;
            try {
                const response = await fetch(`/api/cases/${card.dataset.caseId}`, { method: "DELETE" });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "Could not delete case.");
                card.remove();
                if (!document.querySelector("[data-case-id]")) window.location.reload();
            } catch (error) {
                window.alert(error.message);
            }
        });
    });
});
