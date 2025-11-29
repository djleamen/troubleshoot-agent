function sendMessage() {
    // Get user input
    var userMessage = document.getElementById("userInput").value;
    if (userMessage.trim() === "") return;  // Ignore empty messages

    // Display user message
    var messages = document.getElementById("messages");
    var userMessageDiv = document.createElement("div");
    userMessageDiv.className = "message user-message";
    userMessageDiv.innerHTML = "<strong>You</strong>" + userMessage;
    messages.appendChild(userMessageDiv);

    // Clear input field
    document.getElementById("userInput").value = "";

    // Scroll to bottom
    messages.scrollTop = messages.scrollHeight;

    // Simulate chatbot response
    setTimeout(function() {
        var botMessageDiv = document.createElement("div");
        botMessageDiv.className = "message bot-message";
        botMessageDiv.innerHTML = "<strong>Bot</strong>" + getBotResponse(userMessage);
        messages.appendChild(botMessageDiv);
        messages.scrollTop = messages.scrollHeight;
    }, 1000);
}

function quickReply(message) {
    document.getElementById("userInput").value = message;
    sendMessage();
}

// Simulate a basic bot response based on user input
function getBotResponse(message) {
    var responses = {
        "hello": "Hi! How can I help you today?",
        "how are you": "I'm just a bot, but I'm doing great! How about you?",
        "what is your name": "I'm your friendly chatbot, here to assist you.",
        "goodbye": "Goodbye! Have a great day!"
    };
    message = message.toLowerCase();
    return responses[message] || "Sorry, I didn't understand that. Could you rephrase?";
}

