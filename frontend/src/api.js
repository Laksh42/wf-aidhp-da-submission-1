import axios from 'axios';
import { config } from './config';
import mockApi from './mockApi';

// Create a minimal axios instance with reduced headers
const api_instance = axios.create({
  baseURL: config.apiUrl,
  headers: {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
  },
  // Disable automatic transforms and extra features
  transformRequest: [(data) => JSON.stringify(data)],
  transformResponse: [(data) => {
    try {
      return JSON.parse(data);
    } catch (e) {
      return data;
    }
  }],
  // Disable browser features that might add headers
  withCredentials: false,
  xsrfCookieName: null,
  xsrfHeaderName: null,
});

// Security monitoring function
const logSecurityEvent = (type, details) => {
  console.warn(`[SECURITY] ${type}:`, details);
  // In production, you would log to a secure endpoint or analytics system
};

// Track failed authentication attempts to detect potential brute force attacks
let failedAuthAttempts = {};
const MAX_FAILED_ATTEMPTS = 5;
const LOCKOUT_DURATION_MS = 10 * 60 * 1000; // 10 minutes

const trackFailedAuth = (username) => {
  const now = new Date().getTime();
  
  // Clean up expired lockouts
  Object.keys(failedAuthAttempts).forEach(user => {
    if (failedAuthAttempts[user].lockoutUntil && failedAuthAttempts[user].lockoutUntil < now) {
      delete failedAuthAttempts[user];
    }
  });
  
  // Initialize or increment counter
  if (!failedAuthAttempts[username]) {
    failedAuthAttempts[username] = { count: 1, lastAttempt: now };
  } else {
    failedAuthAttempts[username].count += 1;
    failedAuthAttempts[username].lastAttempt = now;
    
    // Check if account should be locked
    if (failedAuthAttempts[username].count >= MAX_FAILED_ATTEMPTS) {
      failedAuthAttempts[username].lockoutUntil = now + LOCKOUT_DURATION_MS;
      logSecurityEvent('ACCOUNT_LOCKOUT', { username, attempts: failedAuthAttempts[username].count });
    }
  }
  
  return failedAuthAttempts[username];
};

// Check if a user is locked out
const isUserLockedOut = (username) => {
  const now = new Date().getTime();
  return failedAuthAttempts[username] && 
         failedAuthAttempts[username].lockoutUntil && 
         failedAuthAttempts[username].lockoutUntil > now;
};

// Intercept requests to strip unnecessary headers
api_instance.interceptors.request.use(config => {
  // Keep only essential headers
  const essentialHeaders = ['Accept', 'Content-Type', 'Authorization'];
  const newHeaders = {};
  
  Object.keys(config.headers).forEach(key => {
    if (essentialHeaders.includes(key)) {
      newHeaders[key] = config.headers[key];
    }
  });
  
  config.headers = newHeaders;
  
  // Debug log - show what headers are being sent
  console.log('API Request:', config.method.toUpperCase(), config.url);
  console.log('Headers being sent:', JSON.stringify(config.headers));
  
  // Log requests to sensitive endpoints
  if (config.url.includes('/auth/')) {
    logSecurityEvent('AUTH_REQUEST', { 
      method: config.method, 
      url: config.url,
      timestamp: new Date().toISOString()
    });
  }
  
  return config;
});

// Setup response interceptor for error handling
api_instance.interceptors.response.use(
  response => {
    // Log successful auth operations
    if (response.config.url.includes('/auth/')) {
      logSecurityEvent('AUTH_SUCCESS', {
        method: response.config.method,
        url: response.config.url,
        timestamp: new Date().toISOString()
      });
    }
    return response;
  },
  error => {
    console.error('API Error:', error);
    
    // Log auth failures with additional security context
    if (error.config && error.config.url.includes('/auth/')) {
      logSecurityEvent('AUTH_FAILURE', {
        method: error.config.method,
        url: error.config.url,
        status: error.response ? error.response.status : 'network_error',
        timestamp: new Date().toISOString()
      });
      
      // If it's a login attempt, extract username from request data
      if (error.config.url.includes('/auth/token') && error.config.data) {
        try {
          const params = new URLSearchParams(error.config.data);
          const username = params.get('username');
          if (username) {
            const attemptInfo = trackFailedAuth(username);
            logSecurityEvent('FAILED_LOGIN', { 
              username, 
              attemptCount: attemptInfo.count,
              isLockedOut: !!attemptInfo.lockoutUntil
            });
          }
        } catch (e) {
          console.error('Error tracking failed login:', e);
        }
      }
    }
    
    return Promise.reject(error);
  }
);

/**
 * Transform raw API response from backend to the expected frontend format
 * This provides a consistent data structure regardless of backend changes
 */
const transforms = {
  recommendations: {
    format: (apiResponse) => {
      // If the data is already in the expected format, return as is
      if (apiResponse.recommendations) {
        return apiResponse;
      }
      
      // Transform the backend format to the frontend expected format
      return {
        recommendations: apiResponse.products || [],
        user_insights: apiResponse.user_insights || null
      };
    }
  }
};

// Create an API wrapper that uses either the real API or mockApi
const api = {
  // Auth functions
  auth: {
    login: async (username, password) => {
      if (config.features.enableMockData) {
        return mockApi.login(username, password);
      }
      
      try {
        // Check if the user is locked out due to too many failed attempts
        if (isUserLockedOut(username)) {
          const lockoutInfo = failedAuthAttempts[username];
          const lockoutRemainingMs = lockoutInfo.lockoutUntil - new Date().getTime();
          const lockoutMinutes = Math.ceil(lockoutRemainingMs / 60000);
          
          logSecurityEvent('LOGIN_REJECTED', { 
            username, 
            reason: 'ACCOUNT_LOCKED',
            remainingLockoutMinutes: lockoutMinutes
          });
          
          throw {
            response: {
              status: 429,
              data: {
                detail: `Too many failed login attempts. Please try again in ${lockoutMinutes} minutes.`
              }
            }
          };
        }
        
        // Use URLSearchParams for x-www-form-urlencoded data (OAuth2 format)
        const formData = new URLSearchParams();
        formData.append('grant_type', 'password'); // Required for OAuth2 password flow
        formData.append('username', username);
        formData.append('password', password);
        
        console.log('Login attempt with:', username);
        
        // Use application/x-www-form-urlencoded for OAuth2
        const response = await api_instance.post(config.endpoints.auth.login, formData.toString(), {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          // Disable the global transformRequest for this specific call
          transformRequest: [(data) => data], // Don't transform the data
        });
        
        console.log('Login response:', response.data);

        // Set auth token immediately after login
        if (response.data && response.data.access_token) {
          api.setAuthToken(response.data.access_token);
          
          // Clear failed attempts on successful login
          if (failedAuthAttempts[username]) {
            delete failedAuthAttempts[username];
          }
        }
        
        return response.data;
      } catch (error) {
        console.error('Login error details:', error.response ? error.response.data : error.message);
        throw error;
      }
    },
    
    register: async (userData) => {
      if (config.features.enableMockData) {
        return mockApi.register(userData.user_id, userData.password);
      }
      
      // Log the data being sent to the backend for debugging
      console.log('API: Sending registration data to backend:', userData);
      
      // Only include fields that the backend accepts
      const userDataToSend = {
        user_id: userData.user_id,
        password: userData.password,
        email: userData.email || `${userData.user_id}@example.com`,
        full_name: userData.full_name || userData.user_id
      };
      
      const response = await api_instance.post(config.endpoints.auth.register, userDataToSend);
      return response.data;
    },
    
    getUserData: async () => {
      if (config.features.enableMockData) {
        return mockApi.getUserData('testuser');
      }
      
      const response = await api_instance.get(config.endpoints.auth.me);
      return response.data;
    }
  },
  
  // Recommendations functions
  recommendations: {
    getRecommendations: async () => {
      if (config.features.enableMockData) {
        return mockApi.getRecommendations();
      }
      
      try {
        // Get the token directly from localStorage for this request
        const token = localStorage.getItem(config.auth.tokenStorageKey);
        
        // Make the request with explicitly set headers
        const response = await api_instance.get(config.endpoints.recommendations.list, {
          headers: {
            'Authorization': token ? `Bearer ${token}` : undefined
          }
        });
        
        // Transform the API response to expected frontend format
        return transforms.recommendations.format(response.data);
      } catch (error) {
        console.error('Failed to fetch recommendations:', error);
        // Return empty data structure on error to prevent UI crashes
        return { recommendations: [], user_insights: null };
      }
    },
    
    refreshRecommendations: async () => {
      if (config.features.enableMockData) {
        return mockApi.refreshRecommendations();
      }
      
      try {
        // Get the token directly from localStorage for this request
        const token = localStorage.getItem(config.auth.tokenStorageKey);
        
        // Use GET with explicitly set headers
        const response = await api_instance.get(config.endpoints.recommendations.list, {
          headers: {
            'Authorization': token ? `Bearer ${token}` : undefined
          }
        });
        
        return transforms.recommendations.format(response.data);
      } catch (error) {
        console.error('Failed to refresh recommendations:', error);
        throw error; // Let the component handle the error
      }
    },
    
    provideFeedback: async (recommendationId, isRelevant) => {
      if (config.features.enableMockData) {
        return mockApi.provideFeedback(recommendationId, isRelevant);
      }
      
      const response = await api_instance.post(config.endpoints.recommendations.feedback, {
        recommendation_id: recommendationId,
        is_relevant: isRelevant
      });
      return response.data;
    }
  },
  
  // Chat functions
  chat: {
    sendMessage: async (message, sessionId = null) => {
      if (config.features.enableMockData) {
        return mockApi.sendChatMessage(message, sessionId);
      }
      
      // Get the token directly from localStorage for this request
      const token = localStorage.getItem(config.auth.tokenStorageKey);
      
      const response = await api_instance.post(config.endpoints.chat.send, {
        message,
        session_id: sessionId
      }, {
        headers: {
          'Authorization': token ? `Bearer ${token}` : undefined
        }
      });
      return response.data;
    },
    
    getConversations: async () => {
      if (config.features.enableMockData) {
        return [];
      }
      
      // Get the token directly from localStorage for this request
      const token = localStorage.getItem(config.auth.tokenStorageKey);
      
      const response = await api_instance.get(config.endpoints.chat.conversations, {
        headers: {
          'Authorization': token ? `Bearer ${token}` : undefined
        }
      });
      return response.data;
    }
  },

  // Helper function to set auth token
  setAuthToken: (token) => {
    if (token) {
      console.log('Setting auth token in API instance');
      api_instance.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      console.log('Removing auth token from API instance');
      delete api_instance.defaults.headers.common['Authorization'];
    }
  },
  
  // Check if the API is reachable
  checkConnection: async () => {
    try {
      console.log('Checking API connection to:', config.apiUrl);
      // Send a minimal request to check connectivity
      // Try health endpoint first
      try {
        const response = await api_instance.get('/api/health', { timeout: 5000 });
        console.log('API health check succeeded:', response.status);
        return true;
      } catch (healthError) {
        console.log('Health endpoint not available, trying base URL');
        // Try base URL as fallback
        const response = await fetch(config.apiUrl, { 
          method: 'GET',
          headers: { 'Accept': 'application/json' },
          mode: 'cors',
          timeout: 5000 
        });
        console.log('API base URL check succeeded:', response.status);
        return true;
      }
    } catch (error) {
      console.error('API connection check failed:', error);
      return false;
    }
  }
};

// Set auth token for axios if it exists in localStorage
const token = localStorage.getItem(config.auth.tokenStorageKey);
if (token) {
  api.setAuthToken(token);
}

export default api; 