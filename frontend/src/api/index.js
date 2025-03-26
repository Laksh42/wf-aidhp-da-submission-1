import axios from 'axios';
import { config } from '../config';

// Create axios instance with base URL from config
const apiClient = axios.create({
  baseURL: config.api.baseUrl,
  timeout: 15000, // 15 seconds timeout
  headers: {
    'Content-Type': 'application/json'
  }
});

// Security event logging
const logSecurityEvent = (eventType, data) => {
  console.log(`[SECURITY] ${eventType}:`, data);
  // In a real app, you might want to send this to a security monitoring service
};

// Standalone function to check API connection
const checkConnection = async () => {
  try {
    console.log('Checking API connection to:', config.api.baseUrl);
    const response = await apiClient.get('/api/health');
    return response.status === 200;
  } catch (error) {
    console.error('API connection check failed:', error);
    return false;
  }
};

// API module
const api = {
  // Set auth token for all requests
  setAuthToken: (token) => {
    if (token) {
      apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      console.log('Setting auth token in API instance');
    } else {
      delete apiClient.defaults.headers.common['Authorization'];
    }
  },

  // Export the connection check function
  checkConnection,

  // Authentication endpoints
  auth: {
    // Login endpoint
    login: async (userId, password) => {
      console.log('Login attempt with:', userId);
      try {
        // Log the auth request (but not the password)
        logSecurityEvent('AUTH_REQUEST', {
          method: 'post',
          url: '/api/auth/token',
          timestamp: new Date().toISOString()
        });

        // Log headers being sent (for debugging)
        console.log('API Request: POST /api/auth/token');
        console.log('Headers being sent:', {
          'Accept': 'application/json',
          'Content-Type': 'application/x-www-form-urlencoded',
          ...(apiClient.defaults.headers.common['Authorization'] ? 
            { 'Authorization': apiClient.defaults.headers.common['Authorization'] } : {})
        });

        // Use URLSearchParams for form data
        const formData = new URLSearchParams();
        formData.append('username', userId);
        formData.append('password', password);

        const response = await apiClient.post('/api/auth/token', formData, {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
          }
        });
        
        // Log auth success (but not the token)
        logSecurityEvent('AUTH_SUCCESS', {
          method: 'post',
          url: '/api/auth/token',
          timestamp: new Date().toISOString()
        });

        console.log('Login response:', response.data);
        return response.data;
      } catch (error) {
        console.error('Login API error:', error);
        
        // Log auth failure
        logSecurityEvent('AUTH_FAILURE', {
          method: 'post',
          url: '/api/auth/token',
          timestamp: new Date().toISOString(),
          status: error.response?.status,
          message: error.message
        });
        
        throw error;
      }
    },

    // Register endpoint
    register: async (userData) => {
      try {
        const response = await apiClient.post('/api/auth/register', userData);
        return response.data;
      } catch (error) {
        console.error('Registration API error:', error);
        throw error;
      }
    },

    // Get user profile
    getUserProfile: async (userId) => {
      try {
        const response = await apiClient.get(`/api/users/${userId}`);
        return response.data;
      } catch (error) {
        console.error('Get user profile API error:', error);
        throw error;
      }
    },

    // Support the original function name for backward compatibility
    getUserData: async () => {
      try {
        // Since we don't have the userId from the context, we can fetch the current user
        logSecurityEvent('AUTH_REQUEST', {
          method: 'get',
          url: '/api/auth/me',
          timestamp: new Date().toISOString()
        });
        
        console.log('API Request: GET /api/auth/me');
        console.log('Headers being sent:', {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          ...(apiClient.defaults.headers.common['Authorization'] ? 
            { 'Authorization': apiClient.defaults.headers.common['Authorization'] } : {})
        });
        
        const response = await apiClient.get('/api/auth/me');
        
        logSecurityEvent('AUTH_SUCCESS', {
          method: 'get',
          url: '/api/auth/me',
          timestamp: new Date().toISOString()
        });
        
        return response.data;
      } catch (error) {
        console.error('Get user data API error:', error);
        throw error;
      }
    },

    // Validate token endpoint
    validateToken: async () => {
      try {
        const response = await apiClient.get('/api/auth/validate');
        return response.data;
      } catch (error) {
        console.error('Token validation error:', error);
        throw error;
      }
    }
  },

  // Chat endpoints
  chat: {
    // Send message to chatbot
    sendMessage: async (message, sessionId = null) => {
      try {
        console.log('API Request: POST /api/chat/send');
        console.log('Headers being sent:', {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          ...(apiClient.defaults.headers.common['Authorization'] ? 
            { 'Authorization': apiClient.defaults.headers.common['Authorization'] } : {})
        });
        
        const payload = {
          message,
          session_id: sessionId
        };

        console.log('Sending chat message:', payload);
        const response = await apiClient.post('/api/chat/send', payload);
        console.log('Chat response received:', response.data);
        return response.data;
      } catch (error) {
        console.error('Chat API error:', error);
        throw error;
      }
    },

    // Get chat history
    getHistory: async (sessionId) => {
      try {
        const response = await apiClient.get(`/api/chat/history/${sessionId}`);
        return response.data;
      } catch (error) {
        console.error('Chat history API error:', error);
        throw error;
      }
    }
  },

  // Recommendations endpoints
  recommendations: {
    // Get recommendations
    getRecommendations: async () => {
      try {
        console.log('API Request: GET /api/recommendations/');
        console.log('Headers being sent:', {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          ...(apiClient.defaults.headers.common['Authorization'] ? 
            { 'Authorization': apiClient.defaults.headers.common['Authorization'] } : {})
        });
        
        const response = await apiClient.get('/api/recommendations');
        return response.data;
      } catch (error) {
        console.error('Recommendations API error:', error);
        throw error;
      }
    }
  }
};

// Add response interceptor to handle common error responses
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Log API errors but don't handle them here
      // Let the caller handle specific errors
      console.error('API Error:', {
        status: error.response.status,
        url: error.config.url,
        method: error.config.method,
        data: error.response.data
      });
    } else if (error.request) {
      // Network error or no response
      console.error('API Request Error (No Response):', error.request);
    } else {
      // Other errors
      console.error('API Request Setup Error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

export default api; 