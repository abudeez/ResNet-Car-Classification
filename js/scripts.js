document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    const formData = new FormData();
    const fileInput = document.getElementById('imageInput');
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('http://localhost:8000/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        document.getElementById('result').innerText = result.message;
    } catch (error) {
        console.error('There was a problem with the fetch operation:', error);
        document.getElementById('result').innerText = 'Error uploading image';
    }
});