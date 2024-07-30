function addMessage(content, sender) {
    const chatContainer = document.getElementById('chat-container');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);

    const avatarElement = document.createElement('div');
    avatarElement.classList.add('avatar');

    const contentElement = document.createElement('div');
    contentElement.classList.add('message-content');
    contentElement.innerHTML = content;  // Asegura que el contenido HTML se renderice correctamente

    messageElement.appendChild(avatarElement);
    messageElement.appendChild(contentElement);

    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput) {
        addMessage(userInput, 'user');
        document.getElementById('user-input').value = '';

        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: userInput }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                addMessage(data.error, 'bot');
            } else {
                const summary = data.summary;
                addMessage(summary, 'bot');  // Asegúrate de que el texto formateado se pase correctamente aquí
            }
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    }
}
