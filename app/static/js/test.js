// Test suite for chat functionality
const testCases = {
  // Test auth token verification
  async testAuthVerification() {
    console.log('Testing auth verification...');
    try {
      const response = await fetch('/api/auth/verify', {
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        }
      });
      console.log('Auth verification status:', response.status);
      return response.ok;
    } catch (error) {
      console.error('Auth verification failed:', error);
      return false;
    }
  },

  // Test conversation creation
  async testConversationCreation() {
    console.log('Testing conversation creation...');
    try {
      // Log the auth token (partially)
      const tokenPreview = authToken ? `${authToken.substring(0, 10)}...` : 'null';
      console.log('Using auth token:', tokenPreview);

      // First get the current user to get the user_id
      console.log('Getting current user info...');
      const userResponse = await fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!userResponse.ok) {
        console.error('Failed to get user info, status:', userResponse.status);
        return false;
      }
      
      const userData = await userResponse.json();
      console.log('User data:', JSON.stringify(userData, null, 2));
      
      // Prepare request data with user_id
      const requestData = {
        user_id: userData.user_id,
        title: `Test Chat ${new Date().toLocaleString()}`,
        metadata: {}
      };
      console.log('Request data:', JSON.stringify(requestData, null, 2));

      const response = await fetch('/api/conversations', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });

      // Log full response details
      console.log('Response status:', response.status);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      const responseData = await response.json();
      console.log('Full response data:', JSON.stringify(responseData, null, 2));

      if (!response.ok) {
        console.error('Conversation creation failed with status:', response.status);
        if (responseData.detail && Array.isArray(responseData.detail)) {
          console.error('Validation errors:', JSON.stringify(responseData.detail, null, 2));
          responseData.detail.forEach((error, index) => {
            console.error(`Error ${index + 1}:`, JSON.stringify(error, null, 2));
          });
        } else {
          console.error('Error details:', JSON.stringify(responseData, null, 2));
        }
        return false;
      }

      // Store the created conversation to make it available for following tests
      currentConversation = responseData;

      return responseData._id ? true : false;
    } catch (error) {
      console.error('Conversation creation failed with error:', error);
      return false;
    }
  },

  // Test message sending
  async testMessageSending(conversationId) {
    console.log('Testing message sending...');
    try {
      const response = await fetch(`/api/chat`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          content: 'This is a test message',
          role: 'user',
          metadata: {}
        })
      });
      console.log('Message sending status:', response.status);
      if (!response.ok) {
        const error = await response.json();
        console.error('Message sending error:', error);
        return false;
      }
      const data = await response.json();
      console.log('Message response:', data);
      return data.content ? true : false;
    } catch (error) {
      console.error('Message sending failed:', error);
      return false;
    }
  },

  // Test recommendations loading
  async testRecommendationsLoading() {
    console.log("Testing recommendations loading...");
    
    try {
      const response = await fetch('/api/recommendations/', {
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        }
      });
      
      // Log response status and headers
      console.log(`Response status: ${response.status}`);
      
      if (!response.ok) {
        console.error(`Recommendations failed with status: ${response.status}`);
        if (response.headers.get('content-type')?.includes('application/json')) {
          const errorData = await response.json();
          console.error("Error details:", errorData);
        }
        return false;
      }
      
      const data = await response.json();
      console.log("Recommendations loaded:", data);
      
      return data && Array.isArray(data.products);
    } catch (error) {
      console.error("Error loading recommendations:", error);
      return false;
    }
  },

  // Test LLM service directly
  async testLLMService() {
    console.log('Testing LLM service...');
    try {
      const response = await fetch('/api/chat/test', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: 'Test message for LLM service',
          metadata: {
            test: true
          }
        })
      });
      console.log('LLM service status:', response.status);
      if (!response.ok) {
        const error = await response.json();
        console.error('LLM service error:', error);
        return false;
      }
      const data = await response.json();
      console.log('LLM response:', data);
      return data.response ? true : false;
    } catch (error) {
      console.error('LLM service test failed:', error);
      return false;
    }
  }
};

// Run all tests
async function runTests() {
  console.log('Starting test suite...');
  
  // Test auth first
  const authOk = await testCases.testAuthVerification();
  if (!authOk) {
    console.error('Auth verification failed, stopping tests');
    return;
  }
  
  // Test conversation creation
  const conversationOk = await testCases.testConversationCreation();
  if (!conversationOk) {
    console.error('Conversation creation failed, stopping tests');
    return;
  }
  
  // Get the conversation ID from the current conversation
  const conversationId = currentConversation?._id;
  if (!conversationId) {
    console.error('No conversation ID available, stopping tests');
    return;
  }
  
  // Test message sending
  const messageOk = await testCases.testMessageSending(conversationId);
  if (!messageOk) {
    console.error('Message sending failed');
  }
  
  // Test recommendations
  const recommendationsOk = await testCases.testRecommendationsLoading();
  if (!recommendationsOk) {
    console.error('Recommendations loading failed');
  }
  
  /* LLM service test endpoint does not exist
  // Test LLM service
  const llmOk = await testCases.testLLMService();
  if (!llmOk) {
    console.error('LLM service test failed');
  }
  */
  
  console.log('Test results:');
  console.log('- Auth verification:', authOk ? '✅' : '❌');
  console.log('- Conversation creation:', conversationOk ? '✅' : '❌');
  console.log('- Message sending:', messageOk ? '✅' : '❌');
  console.log('- Recommendations:', recommendationsOk ? '✅' : '❌');
  // console.log('- LLM service:', llmOk ? '✅' : '❌');
}

// Export test functions
window.runChatTests = runTests; 