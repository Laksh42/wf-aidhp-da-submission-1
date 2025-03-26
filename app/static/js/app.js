// app.js - Multi-Modal Financial Advisor Chatbot UI

// Global variables
let authToken = localStorage.getItem('authToken');
let userId = localStorage.getItem('userId');
let currentConversation = null;

// DOM elements we'll use frequently
const elements = {
  login: {
    form: document.getElementById('login-form'),
    userId: document.getElementById('login-userid'),
    password: document.getElementById('login-password'),
    errorMsg: document.getElementById('login-error')
  },
  register: {
    form: document.getElementById('register-form'),
    userId: document.getElementById('register-userid'),
    password: document.getElementById('register-password'),
    errorMsg: document.getElementById('register-error')
  },
  chat: {
    form: document.getElementById('chat-form'),
    input: document.getElementById('chat-input'),
    messages: document.getElementById('messages'),
    recommendations: document.getElementById('recommendations')
  },
  pages: {
    login: document.getElementById('login-page'),
    register: document.getElementById('register-page'),
    chat: document.getElementById('chat-page')
  },
  nav: {
    logoutBtn: document.getElementById('logout-btn')
  }
};

// ----- Event Listeners -----

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

// Register page link
document.getElementById('register-link').addEventListener('click', (e) => {
  e.preventDefault();
  showRegisterPage();
});

// Login page link
document.getElementById('login-link').addEventListener('click', (e) => {
  e.preventDefault();
  showLoginPage();
});

// ----- Functions -----

// Initialize the application
async function init() {
  try {
    // Set up event listeners
    if (elements.login.form) {
      elements.login.form.addEventListener('submit', handleLogin);
    }
    
    if (elements.register.form) {
      elements.register.form.addEventListener('submit', handleRegister);
    }
    
    if (elements.chat.form) {
      elements.chat.form.addEventListener('submit', handleSendMessage);
    }
    
    if (elements.nav.logoutBtn) {
      elements.nav.logoutBtn.addEventListener('click', handleLogout);
    }
    
    // Verify auth token if exists
    if (authToken && userId) {
      try {
        console.log('Verifying authentication token...');
        const response = await fetch('/api/auth/verify', {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json'
          }
        });
        
        if (response.ok) {
          // Token is valid, show chat interface
          console.log('Token verified successfully. Loading chat interface...');
          elements.nav.logoutBtn.classList.remove('hidden');
          showChatInterface();
          
          // Load recommendations and create conversation with proper error handling
          try {
            // Load recommendations first
            await loadRecommendations();
            console.log('Recommendations loaded successfully');
          } catch (recError) {
            console.error('Error loading recommendations:', recError);
            // Continue with conversation creation even if recommendations fail
          }
          
          try {
            // Create a new conversation
            await createNewConversation();
            console.log('Conversation created successfully');
          } catch (convError) {
            console.error('Error creating conversation:', convError);
            // Add fallback message if conversation creation fails
            addBotMessage('Welcome! I encountered an issue starting our conversation. Please try sending a message to begin chatting.');
          }
        } else {
          // Token is invalid, clear it and show login
          console.log('Invalid token detected, clearing auth data');
          localStorage.removeItem('authToken');
          localStorage.removeItem('userId');
          authToken = null;
          userId = null;
          showLoginPage();
        }
      } catch (error) {
        console.error('Auth verification failed:', error);
        // Clear invalid auth data
        localStorage.removeItem('authToken');
        localStorage.removeItem('userId');
        authToken = null;
        userId = null;
        showLoginPage();
      }
    } else {
      // No token, show login page
      console.log('No auth token found, showing login page');
      showLoginPage();
    }
  } catch (error) {
    console.error('Initialization error:', error);
    // Show login page on any initialization error
    showLoginPage();
  }
}

// ----- Authentication Functions -----

// Handle login form submission
async function handleLogin(e) {
  e.preventDefault();
  
  const userIdInput = elements.login.userId.value.trim();
  const password = elements.login.password.value;
  
  if (!userIdInput || !password) {
    showError(elements.login.errorMsg, 'Please enter both User ID and Password');
    return;
  }
  
  try {
    const response = await fetch('/api/auth/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: new URLSearchParams({
        'username': userIdInput,
        'password': password
      })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.detail || 'Login failed');
    }
    
    // Clear any existing tokens first
    localStorage.removeItem('authToken');
    localStorage.removeItem('userId');
    
    // Save new auth token and user ID
    localStorage.setItem('authToken', data.access_token);
    localStorage.setItem('userId', data.user_id);
    authToken = data.access_token;
    userId = data.user_id;
    
    // Verify the token immediately after login
    const verifyResponse = await fetch('/api/auth/verify', {
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      }
    });
    
    if (!verifyResponse.ok) {
      throw new Error('Token verification failed after login');
    }
    
    // Show logout button
    elements.nav.logoutBtn.classList.remove('hidden');
    
    // Switch to chat interface
    showChatInterface();
    await Promise.all([
      loadRecommendations(),
      createNewConversation()
    ]);
    
  } catch (error) {
    console.error('Login error:', error);
    // Clear any invalid tokens
    localStorage.removeItem('authToken');
    localStorage.removeItem('userId');
    authToken = null;
    userId = null;
    showError(elements.login.errorMsg, error.message);
  }
}

// Handle registration form submission
async function handleRegister(e) {
  e.preventDefault();
  
  const userIdInput = elements.register.userId.value.trim();
  const password = elements.register.password.value;
  
  if (!userIdInput || !password) {
    showError(elements.register.errorMsg, 'Please enter both User ID and Password');
    return;
  }
  
  try {
    const response = await fetch('/api/auth/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        user_id: userIdInput,
        password: password
      })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.detail || 'Registration failed');
    }
    
    // Clear any existing tokens first
    localStorage.removeItem('authToken');
    localStorage.removeItem('userId');
    authToken = null;
    userId = null;
    
    // Auto login after registration
    const loginResponse = await fetch('/api/auth/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: new URLSearchParams({
        'username': userIdInput,
        'password': password
      })
    });
    
    const loginData = await loginResponse.json();
    
    if (!loginResponse.ok) {
      throw new Error(loginData.detail || 'Login failed after registration');
    }
    
    // Save new auth token and user ID
    localStorage.setItem('authToken', loginData.access_token);
    localStorage.setItem('userId', loginData.user_id);
    authToken = loginData.access_token;
    userId = loginData.user_id;
    
    // Verify the token immediately after login
    const verifyResponse = await fetch('/api/auth/verify', {
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      }
    });
    
    if (!verifyResponse.ok) {
      throw new Error('Token verification failed after registration');
    }
    
    // Show logout button
    elements.nav.logoutBtn.classList.remove('hidden');
    
    // Switch to chat interface
    showChatInterface();
    await Promise.all([
      loadRecommendations(),
      createNewConversation()
    ]);
    
  } catch (error) {
    console.error('Registration error:', error);
    // Clear any invalid tokens
    localStorage.removeItem('authToken');
    localStorage.removeItem('userId');
    authToken = null;
    userId = null;
    showError(elements.register.errorMsg, error.message);
  }
}

// Handle logout
function handleLogout() {
  // Clear auth token and user ID
  localStorage.removeItem('authToken');
  localStorage.removeItem('userId');
  authToken = null;
  userId = null;
  currentConversation = null;
  
  // Hide logout button
  elements.nav.logoutBtn.classList.add('hidden');
  
  // Switch to login page
  showLoginPage();
}

// ----- Chat Functions -----

// Create a new conversation
async function createNewConversation() {
  try {
    // Make sure we have authentication
    if (!authToken) {
      throw new Error('Authentication token not found. Please log in again.');
    }
    
    // Verify the token before proceeding
    try {
      const verifyResponse = await fetch('/api/auth/verify', {
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!verifyResponse.ok) {
        throw new Error('Token verification failed');
      }
    } catch (error) {
      // Handle token verification error
      console.error("Token verification failed:", error);
      localStorage.removeItem('authToken');
      localStorage.removeItem('userId');
      authToken = null;
      userId = null;
      showLoginPage();
      throw new Error('Authentication failed. Please log in again.');
    }
    
    // Get the current user to build proper request
    const userResponse = await fetch('/api/auth/me', {
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      }
    });
    
    if (!userResponse.ok) {
      throw new Error('Failed to get user information');
    }
    
    const userData = await userResponse.json();
    console.log('Creating conversation with user ID:', userData.user_id);
    
    // Create conversation with the correct model structure
    const response = await fetch('/api/conversations', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        user_id: userData.user_id,
        title: `Chat ${new Date().toLocaleString()}`
      })
    });
    
    if (!response.ok) {
      // Properly extract error details 
      const errorData = await response.json();
      console.error('Conversation creation error details:', errorData);
      throw new Error(JSON.stringify(errorData.detail || 'Failed to create conversation'));
    }
    
    const data = await response.json();
    currentConversation = data;
    console.log('Conversation created successfully');
    
    // Add welcome message
    addBotMessage('Hello! I am your financial advisor chatbot. How can I help you today?');
    
    return currentConversation;
  } catch (error) {
    console.error('Error creating conversation:', error);
    addBotMessage('There was an error starting the conversation. Please try refreshing the page.');
    throw error;
  }
}

// Load financial product recommendations
async function loadRecommendations() {
  try {
    if (!authToken) {
      throw new Error('Authentication token not found. Please log in again.');
    }
    
    // Verify the token before proceeding
    try {
      const verifyResponse = await fetch('/api/auth/verify', {
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!verifyResponse.ok) {
        throw new Error('Token verification failed');
      }
    } catch (error) {
      // Handle token verification error
      console.error("Token verification failed:", error);
      localStorage.removeItem('authToken');
      localStorage.removeItem('userId');
      authToken = null;
      userId = null;
      showLoginPage();
      throw new Error('Authentication failed. Please log in again.');
    }
    
    // Update to the correct endpoint URL
    const response = await fetch('/api/recommendations/', {
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to load recommendations');
    }
    
    const data = await response.json();
    
    // Display recommendations
    if (data && data.products && data.products.length > 0) {
      const recommendationsHtml = data.products.map(product => `
        <div class="recommendation-item">
          <h3>${product.name}</h3>
          <p>${product.description}</p>
          <p class="recommendation-reason">Why: ${product.reason}</p>
          <div class="recommendation-score">
            <span>Confidence: ${Math.round(product.score)}%</span>
          </div>
        </div>
      `).join('');
      
      elements.chat.recommendations.innerHTML = recommendationsHtml;
    } else {
      elements.chat.recommendations.innerHTML = `
        <p>We currently don't have specific recommendations. Please chat with our advisor for personalized suggestions.</p>
      `;
    }
    
  } catch (error) {
    console.error('Error loading recommendations:', error);
    elements.chat.recommendations.innerHTML = '<p>Unable to load recommendations at this time.</p>';
  }
}

// Handle sending a message
async function handleSendMessage(e) {
  e.preventDefault();
  
  const message = elements.chat.input.value.trim();
  
  if (!message) return;
  
  // Add user message to chat
  addUserMessage(message);
  
  // Clear input
  elements.chat.input.value = '';
  
  try {
    // Check if we have a conversation
    if (!currentConversation) {
      console.log('No active conversation, creating one first...');
      try {
        // Wait for conversation creation to complete
        currentConversation = await createNewConversation();
        
        if (!currentConversation || !currentConversation._id) {
          throw new Error('Failed to create conversation');
        }
        
        console.log('Successfully created new conversation:', currentConversation._id);
      } catch (convError) {
        console.error('Failed to create conversation before sending message:', convError);
        addBotMessage('I apologize, but I encountered an error creating a conversation. Please refresh the page and try again.');
        return; // Exit early as we can't send a message without a conversation
      }
    }
    
    // Verify we have a valid conversation ID
    if (!currentConversation || !currentConversation._id) {
      throw new Error('Missing conversation ID, cannot send message');
    }
    
    console.log('Sending message to conversation:', currentConversation._id);
    
    // Send message to API
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        conversation_id: currentConversation._id,
        role: 'user', // Using lowercase 'user' instead of 'USER'
        content: message
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      console.error('Message sending error details:', errorData);
      throw new Error(errorData.detail || 'Failed to send message');
    }
    
    const data = await response.json();
    
    // Add bot response to chat
    if (data && data.content) {
      addBotMessage(data.content);
      
      // Refresh recommendations after each message
      await loadRecommendations();
    } else {
      addBotMessage('I apologize, but I was unable to process your request at this time. Please try again later.');
    }
    
  } catch (error) {
    console.error('Error sending message:', error);
    addBotMessage('I apologize, but I encountered an error while processing your message. Please try again.');
  }
}

// Add a user message to the chat
function addUserMessage(message) {
  const messageElement = document.createElement('div');
  messageElement.classList.add('message', 'message-user');
  messageElement.textContent = message;
  
  elements.chat.messages.appendChild(messageElement);
  scrollToBottom();
}

// Add a bot message to the chat
function addBotMessage(message) {
  const messageElement = document.createElement('div');
  messageElement.classList.add('message', 'message-bot');
  messageElement.textContent = message;
  
  elements.chat.messages.appendChild(messageElement);
  scrollToBottom();
}

// ----- UI Helper Functions -----

// Show login page
function showLoginPage() {
  elements.pages.login.classList.remove('hidden');
  elements.pages.register.classList.add('hidden');
  elements.pages.chat.classList.add('hidden');
  elements.nav.logoutBtn.classList.add('hidden');
}

// Show registration page
function showRegisterPage() {
  elements.pages.login.classList.add('hidden');
  elements.pages.register.classList.remove('hidden');
  elements.pages.chat.classList.add('hidden');
  elements.nav.logoutBtn.classList.add('hidden');
}

// Show chat interface
function showChatInterface() {
  elements.pages.login.classList.add('hidden');
  elements.pages.register.classList.add('hidden');
  elements.pages.chat.classList.remove('hidden');
  elements.nav.logoutBtn.classList.remove('hidden');
}

// Show error message
function showError(element, message) {
  element.textContent = message;
  element.classList.remove('hidden');
  
  // Hide error after 5 seconds
  setTimeout(() => {
    element.classList.add('hidden');
  }, 5000);
}

// Scroll chat to bottom
function scrollToBottom() {
  elements.chat.messages.scrollTop = elements.chat.messages.scrollHeight;
}

// Verify token periodically (every 5 minutes)
setInterval(async () => {
  if (authToken) {
    try {
      const response = await fetch('/api/auth/verify', {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      });
      
      if (!response.ok) {
        // Token is invalid or expired
        handleLogout();
      }
    } catch (error) {
      console.error('Token verification error:', error);
    }
  }
}, 5 * 60 * 1000); 