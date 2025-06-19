// script.js
document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const imagePreview = document.getElementById('image-preview');
    const previewImage = imagePreview.querySelector('img');
    const uploadStatus = document.getElementById('upload-status');
    const resultImage = document.getElementById('result-image');
    const resultText = document.getElementById('result-text');

    const startWebcamBtn = document.getElementById('start-webcam');
    const stopWebcamBtn = document.getElementById('stop-webcam');
    const webcam = document.getElementById('webcam');
    const webcamResult = document.getElementById('webcam-result');
    const webcamStatus = document.getElementById('webcam-status');

    let pollClassificationInterval = null;

    const showMessage = (element, message, type = 'info') => {
        element.classList.remove('hidden', 'text-green-500', 'text-red-500', 'text-blue-500', 'bg-green-100', 'bg-red-100', 'bg-blue-100');
        element.classList.add('p-2', 'rounded');
        if (type === 'success') {
            element.classList.add('bg-green-100', 'text-green-500');
        } else if (type === 'error') {
            element.classList.add('bg-red-100', 'text-red-500');
        } else if (type === 'info') {
            element.classList.add('bg-blue-100', 'text-blue-500');
        }
        element.innerHTML = message;
    };

    const hideMessage = (element) => {
        element.classList.add('hidden');
        element.innerHTML = '';
    };

    // --- Image Upload Functionality ---
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('border-blue-500', 'bg-blue-50');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('border-blue-500', 'bg-blue-50');
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('border-blue-500', 'bg-blue-50');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            fileInput.files = e.dataTransfer.files;
            displayImagePreview(file);
        } else {
            showMessage(uploadStatus, 'Please drop an image file.', 'error');
        }
    });

    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            displayImagePreview(fileInput.files[0]);
        } else {
            imagePreview.classList.add('hidden');
            previewImage.src = '#';
        }
    });

    function displayImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            imagePreview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    uploadBtn.addEventListener('click', async () => {
        if (fileInput.files.length === 0) {
            showMessage(uploadStatus, 'Please select an image to upload!', 'error');
            return;
        }

        hideMessage(uploadStatus);
        resultImage.innerHTML = '';
        resultText.innerHTML = '';
        showMessage(uploadStatus, 'Uploading and detecting...', 'info');

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Something went wrong during detection.');
            }

            const data = await response.json();
            showMessage(uploadStatus, 'Upload and detection successful!', 'success');

            resultImage.innerHTML = `<img src="data:image/jpeg;base64,${data.image}" alt="Detected Image" class="w-full rounded">`;

            let resultHtml = '';
            if (data.results.recyclable.length > 0) {
                resultHtml += `<div class="bg-yellow-200 p-4 rounded"><strong>Recyclable items:</strong><br>- ${data.results.recyclable.join('<br>- ')}</div>`;
            }
            if (data.results.non_recyclable.length > 0) {
                resultHtml += `<div class="bg-blue-200 p-4 rounded"><strong>Non-Recyclable items:</strong><br>- ${data.results.non_recyclable.join('<br>- ')}</div>`;
            }
            if (data.results.hazardous.length > 0) {
                resultHtml += `<div class="bg-red-200 p-4 rounded"><strong>Hazardous items:</strong><br>- ${data.results.hazardous.join('<br>- ')}</div>`;
            }
            resultText.innerHTML = resultHtml;

        } catch (error) {
            showMessage(uploadStatus, `Error: ${error.message}`, 'error');
            resultImage.innerHTML = '';
            resultText.innerHTML = '';
        }
    });

    // --- Webcam Functionality ---
    startWebcamBtn.addEventListener('click', async () => {
        // Clear previous states
        hideMessage(webcamStatus);
        webcamResult.innerHTML = '';

        showMessage(webcamStatus, 'Starting webcam...', 'info');

        // Reset webcam src to force reload the stream
        webcam.src = ''; 
        webcam.src = '/video_feed';
        webcam.classList.remove('hidden');
        startWebcamBtn.classList.add('hidden');
        stopWebcamBtn.classList.remove('hidden');

        // Set up onload and onerror for webcam image
        webcam.onload = () => {
            showMessage(webcamStatus, 'Webcam stream started.', 'success');
        };
        webcam.onerror = () => {
            showMessage(webcamStatus, 'Error loading webcam stream. Check backend logs or webcam settings.', 'error');
            webcam.classList.add('hidden');
            startWebcamBtn.classList.remove('hidden');
            stopWebcamBtn.classList.add('hidden');
        };

        // Start polling for classification data
        if (pollClassificationInterval) {
            clearInterval(pollClassificationInterval);
        }
        pollClassificationInterval = setInterval(async () => {
            try {
                const response = await fetch('/webcam_classification');
                if (!response.ok) {
                    throw new Error('Failed to fetch classification data.');
                }
                const data = await response.json();

                let resultHtml = '';
                if (data.recyclable.length > 0) {
                    resultHtml += `<div class="bg-yellow-200 p-4 rounded"><strong>Recyclable items:</strong><br>- ${data.recyclable.join('<br>- ')}</div>`;
                }
                if (data.non_recyclable.length > 0) {
                    resultHtml += `<div class="bg-blue-200 p-4 rounded"><strong>Non-Recyclable items:</strong><br>- ${data.non_recyclable.join('<br>- ')}</div>`;
                }
                if (data.hazardous.length > 0) {
                    resultHtml += `<div class="bg-red-200 p-4 rounded"><strong>Hazardous items:</strong><br>- ${data.hazardous.join('<br>- ')}</div>`;
                }
                webcamResult.innerHTML = resultHtml;

            } catch (error) {
                console.error('Frontend: Error fetching webcam classification:', error);
                // Only show error if the webcam stream itself isn't stopped
                if (webcam.src !== '') {
                    // Avoid spamming error messages if backend is just shutting down
                    if (!error.message.includes("Failed to fetch")) { // Simple check to avoid network errors during shutdown
                        showMessage(webcamStatus, `Error fetching classification: ${error.message}`, 'error');
                    }
                }
            }
        }, 1000); // Poll every 1 second (adjust as needed)
    });

    stopWebcamBtn.addEventListener('click', async () => {
        // First, tell the backend to stop the stream
        showMessage(webcamStatus, 'Sending stop signal to backend...', 'info');
        try {
            const response = await fetch('/stop_webcam_backend', { 
                method: 'POST',
                // Optional: set headers if needed, e.g., 'Content-Type': 'application/json' for body
            });
            if (!response.ok) {
                // If backend returns an error status
                const errorData = await response.json();
                console.error('Frontend: Failed to send stop signal to backend:', errorData.message);
                showMessage(webcamStatus, `Failed to stop webcam on backend: ${errorData.message}`, 'error');
            } else {
                console.log('Frontend: Stop signal sent to backend successfully.');
                // Backend will handle releasing the camera now
            }
        } catch (error) {
            // Network error (e.g., backend crashed, unreachable)
            console.error('Frontend: Error sending stop signal:', error);
            showMessage(webcamStatus, `Error sending stop signal: ${error.message}`, 'error');
        } finally {
            // Regardless of backend signal success, stop frontend display and polling
            if (webcam.src !== '') { 
                webcam.src = ''; // Stop the video stream by clearing src
                webcam.classList.add('hidden');
                startWebcamBtn.classList.remove('hidden');
                stopWebcamBtn.classList.add('hidden');
                webcamResult.innerHTML = '';
            }
            if (pollClassificationInterval) {
                clearInterval(pollClassificationInterval); // Stop polling
                pollClassificationInterval = null;
            }
            // Give a moment for backend to actually release before showing 'stopped' status
            setTimeout(() => {
                showMessage(webcamStatus, 'Webcam stopped successfully.', 'success');
            }, 1000); // Waktu yang cukup untuk backend melepaskan kamera (misal 1 detik)
        }
    });
});