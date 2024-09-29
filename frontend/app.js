let accessToken = "";

// Handle the login form submission
document.getElementById('login-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        const response = await fetch('http://127.0.0.1:8000/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });

        if (!response.ok) {
            throw new Error('Invalid login credentials');
        }

        const result = await response.json();
        accessToken = result.access_token;

        // Show the API options and hide the login form
        document.getElementById('api-options').style.display = 'block';
        document.getElementById('login-form').style.display = 'none'; // Hide login form
    } catch (error) {
        alert('Login failed: ' + error.message);
    }
});

// Event listeners for API buttons
document.getElementById('predict-btn').addEventListener('click', function () {
    toggleTextInput(true);
});

document.getElementById('predictions-btn').addEventListener('click', async function () {
    await sendApiRequest('GET', '/predictions');
});

document.getElementById('metadata-btn').addEventListener('click', async function () {
    await sendApiRequest('GET', '/model_metadata');
});

// Function to toggle text input area
function toggleTextInput(show) {
    const textArea = document.getElementById('text-area');
    textArea.style.display = show ? 'block' : 'none';
}

// Handle text submission (for /predict)
document.getElementById('submit-text-btn').addEventListener('click', async function () {
    const textInput = document.getElementById('text-input').value;

    if (!textInput.trim()) {
        alert('Please enter some text!');
        return;
    }

    await sendApiRequest('POST', '/predict', { text: textInput });
});

// Generic function to send API requests
async function sendApiRequest(method, endpoint, body = null) {
    const responseOutput = document.getElementById('response-output');

    try {
        const response = await fetch(`http://127.0.0.1:8000${endpoint}`, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${accessToken}`,
            },
            body: body ? JSON.stringify(body) : null,
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const result = await response.json();
        responseOutput.textContent = JSON.stringify(result, null, 2);
    } catch (error) {
        responseOutput.textContent = 'Error: ' + error.message;
    }
}
