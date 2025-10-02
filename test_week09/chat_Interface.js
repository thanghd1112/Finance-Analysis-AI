// Function to handle chat interface interactions
document.addEventListener('DOMContentLoaded', function() {
    // Add event listener for the send button if it exists
    const sendButton = document.querySelector('.chat-send-button');
    if (sendButton) {
        sendButton.addEventListener('click', function() {
            // This will trigger the Streamlit form submit
            document.querySelector('[data-testid="stFormSubmitButton"]').click();
        });
    }
    
    // Format code blocks with syntax highlighting
    document.querySelectorAll('pre code').forEach(function(block) {
        // Add class for styling
        block.parentNode.classList.add('code-block');
    });
    
    // Scroll to the bottom of the chat container
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});

// Function to escape HTML special characters to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Format Python code with some basic highlighting
function formatPythonCode(code) {
    if (!code) return '';
    
    // Replace Python keywords with highlighted spans
    const keywords = ['def', 'if', 'else', 'elif', 'for', 'while', 'return', 'import', 'from', 'class', 'try', 'except', 'finally', 'with'];
    
    let formattedCode = escapeHtml(code);
    
    // Highlight keywords
    keywords.forEach(keyword => {
        const regex = new RegExp(`\\b${keyword}\\b`, 'g');
        formattedCode = formattedCode.replace(regex, `<span style="color: #0000FF;">${keyword}</span>`);
    });
    
    // Highlight strings
    formattedCode = formattedCode.replace(/(["'])(.*?)\1/g, '<span style="color: #008000;">$1$2$1</span>');
    
    // Highlight comments
    formattedCode = formattedCode.replace(/(#.*)$/gm, '<span style="color: #808080;">$1</span>');
    
    return formattedCode;
}