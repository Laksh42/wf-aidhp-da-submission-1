import React, { createContext, useState, useContext, useEffect } from 'react';
import api from '../api';
import { config } from '../config';

// Create the authentication context
export const AuthContext = createContext();

// Custom hook to use the auth context
export const useAuth = () => useContext(AuthContext);

// Log auth events (for debugging and security monitoring)
const logAuthEvent = (eventType, data) => {
  console.log('[AUTH EVENT]', eventType + ':', data);
  // In a real app, you might want to send this to a security monitoring service
};

// Provider component that wraps the app and provides auth context
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [token, setToken] = useState(null);
  const [userPersona, setUserPersona] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  // Add connection state to track API connectivity
  const [apiConnected, setApiConnected] = useState(true);

  // Helper function to check connection safely
  const safeCheckConnection = async () => {
    try {
      console.log("Attempting to check API connection...");
      if (typeof api.checkConnection === 'function') {
        return await api.checkConnection();
      } else {
        console.error("api.checkConnection is not a function!", api);
        // Fallback: Try a simple fetch to the backend base URL
        try {
          // Try to ping the API health endpoint first
          const healthResponse = await fetch(`${config.apiUrl}/api/health`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' },
            mode: 'cors'
          });
          
          if (healthResponse.ok) {
            return true;
          }
          
          // Fallback to the root URL if health endpoint doesn't exist
          const response = await fetch(config.apiUrl, {
            method: 'GET',
            headers: { 'Accept': 'application/json' },
            mode: 'cors'
          });
          
          return response.ok;
        } catch (e) {
          console.error("Fallback connection check failed:", e);
          return false;
        }
      }
    } catch (e) {
      console.error("Safe connection check failed:", e);
      return false;
    }
  };

  // Initialize auth state on component mount
  useEffect(() => {
    const initAuth = async () => {
      try {
        // Check if user is authenticated by token
        const storedToken = localStorage.getItem(config.auth.tokenStorageKey);
        console.log('AuthContext initAuth: Token from localStorage:', storedToken ? 'Found' : 'Not found');
        
        if (!storedToken) {
          console.log('AuthContext initAuth: No token found, setting loading to false');
          setLoading(false);
          return;
        }
  
        // Ensure token is set in API instance
        api.setAuthToken(storedToken);
        
        // Check API connectivity first - use try/catch to handle potential errors
        const isConnected = await safeCheckConnection();
        setApiConnected(isConnected);
        
        if (!isConnected) {
          console.log('AuthContext initAuth: API connection failed');
          setLoading(false);
          return;
        }
        
        try {
          // Validate token and get user data
          const userData = await api.auth.getUserData();
          
          if (userData) {
            console.log('AuthContext initAuth: Token valid, setting user data');
            setUser(userData);
            setToken(storedToken);
            setIsAuthenticated(true);
            logAuthEvent('TOKEN_VALIDATION', { 
              success: true, 
              userId: userData.user_id
            });
          }
        } catch (e) {
          console.log('AuthContext initAuth: Token validation failed, clearing token');
          localStorage.removeItem(config.auth.tokenStorageKey);
          api.setAuthToken(null);
          logAuthEvent('TOKEN_VALIDATION', { 
            success: false, 
            error: e.message
          });
        }
      } catch (e) {
        console.error('AuthContext initAuth error:', e);
      } finally {
        // Load persona data if it exists
        try {
          const storedPersona = localStorage.getItem(config.auth.userStorageKey);
          if (storedPersona) {
            setUserPersona(JSON.parse(storedPersona));
          }
        } catch (err) {
          console.error('Error loading user persona:', err);
        }
        
        setLoading(false);
      }
    };

    initAuth();
  }, []);

  // Periodically check API connectivity
  useEffect(() => {
    const checkConnectivity = async () => {
      const isConnected = await safeCheckConnection();
      setApiConnected(isConnected);
    };
    
    // Check connection immediately
    checkConnectivity();
    
    // Then check every 30 seconds
    const interval = setInterval(checkConnectivity, 30000);
    
    return () => clearInterval(interval);
  }, []);

  // Login user with userId and password
  const login = async (userId, password, forceRefresh = false) => {
    try {
      // Check API connection before attempting login
      const isConnected = await safeCheckConnection();
      setApiConnected(isConnected);
      
      if (!isConnected) {
        const errorMsg = 'Cannot connect to authentication server. Please check your connection.';
        setError(errorMsg);
        return false;
      }
      
      setLoading(true);
      setError(null);
      
      console.log('AuthContext: Attempting login for', userId);
      logAuthEvent('LOGIN_ATTEMPT', { userId });
      
      // Clear existing token if force refresh
      if (forceRefresh) {
        localStorage.removeItem(config.auth.tokenStorageKey);
        api.setAuthToken(null);
      }
      
      const response = await api.auth.login(userId, password);

      if (response && response.access_token) {
        // Set token in state and local storage
        setToken(response.access_token);
        localStorage.setItem(config.auth.tokenStorageKey, response.access_token);
        api.setAuthToken(response.access_token);
        
        try {
          // Create a basic user object from the response
          // This avoids the need to call getUserProfile which might not be implemented
          const basicUserData = {
            user_id: response.user_id || userId,
            email: response.email || null,
            is_active: true
          };
          
          setUser(basicUserData);
          setIsAuthenticated(true);
          localStorage.setItem('user', JSON.stringify(basicUserData));
          
          logAuthEvent('LOGIN_SUCCESS', { userId });
          
          // Redirect to dashboard after successful login
          if (window.location.pathname === '/login' || window.location.pathname === '/') {
            window.location.href = '/dashboard';
          }
          
          return true;
        } catch (userError) {
          console.error('Error setting user data:', userError);
          // Continue with login even if we can't get full user data
          setUser({ user_id: userId });
          setIsAuthenticated(true);
          logAuthEvent('LOGIN_SUCCESS_PARTIAL', { 
            userId, 
            error: userError.message 
          });
          
          // Redirect to dashboard after successful login
          if (window.location.pathname === '/login' || window.location.pathname === '/') {
            window.location.href = '/dashboard';
          }
          
          return true;
        }
      } else {
        const errorMsg = 'Invalid login response';
        setError(errorMsg);
        logAuthEvent('LOGIN_ERROR', { userId, error: errorMsg });
        return false;
      }
    } catch (error) {
      console.error('Login error:', error);
      
      let errorType = 'UNKNOWN';
      let errorMessage = error.message || 'An unknown error occurred';
      let url = 'unknown';
      
      if (error.response) {
        // Backend API error
        errorType = 'API';
        errorMessage = error.response.data?.detail || `Server error: ${error.response.status}`;
        url = error.response.config.url;
      } else if (error.request) {
        // No response from server
        errorType = 'NETWORK';
        errorMessage = 'Network error: Server did not respond';
        url = error.config?.url || 'unknown';
      }
      
      setError(errorMessage);
      
      logAuthEvent('LOGIN_FAILURE', { 
        userId, 
        errorType,
        errorMessage,
        url: url.startsWith('/') ? `${config.api.baseUrl}${url}` : url
      });
      
      return false;
    } finally {
      setLoading(false);
    }
  };

  // Register a new user
  const register = async (userData) => {
    try {
      // Check API connection before attempting registration
      const isConnected = await safeCheckConnection();
      setApiConnected(isConnected);
      
      if (!isConnected) {
        const errorMsg = 'Cannot connect to authentication server. Please check your connection.';
        setError(errorMsg);
        return false;
      }
      
      setLoading(true);
      setError(null);

      console.log('AuthContext: Attempting registration for', userData.user_id);
      logAuthEvent('REGISTRATION_ATTEMPT', { userId: userData.user_id });

      const response = await api.auth.register(userData);

      if (response) {
        // Auto login after registration
        return await login(userData.user_id, userData.password);
      } else {
        const errorMsg = 'Invalid registration response';
        setError(errorMsg);
        logAuthEvent('REGISTRATION_ERROR', { userId: userData.user_id, error: errorMsg });
        return false;
      }
    } catch (error) {
      console.error('Registration error:', error);
      
      let errorType = 'UNKNOWN';
      let errorMessage = error.message || 'An unknown error occurred';
      
      if (error.response) {
        // Backend API error
        errorType = 'API';
        
        // Handle validation errors which might be in different formats
        if (error.response.data && error.response.status === 422) {
          console.log('Validation error details:', error.response.data);
          
          // FastAPI can return validation errors in several formats
          if (Array.isArray(error.response.data.detail)) {
            errorMessage = error.response.data.detail.map(err => {
              if (err.msg) return `${err.loc.join('.')}: ${err.msg}`;
              return String(err);
            }).join(', ');
          } else if (typeof error.response.data.detail === 'string') {
            errorMessage = error.response.data.detail;
          } else if (error.response.data.message) {
            errorMessage = error.response.data.message;
          } else {
            errorMessage = 'Validation error in the provided data. Please check all required fields.';
          }
        } else {
          errorMessage = error.response.data?.detail || `Server error: ${error.response.status}`;
        }
      } else if (error.request) {
        // No response from server
        errorType = 'NETWORK';
        errorMessage = 'Network error: Server did not respond';
      }
      
      logAuthEvent('REGISTRATION_FAILURE', { 
        userId: userData.user_id, 
        errorType,
        errorMessage,
        url: error.config?.url || 'unknown'
      });
      
      setError(errorMessage);
      return false;
    } finally {
      setLoading(false);
    }
  };

  // Save user persona data
  const saveUserPersona = (persona) => {
    setUserPersona(persona);
    localStorage.setItem(config.auth.userStorageKey, JSON.stringify(persona));
    logAuthEvent('PERSONA_UPDATE', { 
      userId: user?.user_id,
      success: true
    });
  };

  // Logout user
  const handleLogout = () => {
    const userId = user?.user_id;
    
    // Remove token from localStorage
    localStorage.removeItem(config.auth.tokenStorageKey);
    localStorage.removeItem('user');
    
    // Clear auth token in API
    api.setAuthToken(null);
    
    // Reset state
    setUser(null);
    setToken(null);
    setUserPersona(null);
    setIsAuthenticated(false);
    setError('');
    
    logAuthEvent('LOGOUT', { userId });
    
    // Use window.location.href for logout as it's a full app reset
    window.location.href = '/login';
  };

  // Function to get a valid token, refreshing if needed
  const getToken = async () => {
    // If no token exists, return null (user needs to login)
    if (!token) {
      return null;
    }
    
    try {
      // Test if token is valid with a simple API call
      await api.auth.validateToken();
      return token;
    } catch (err) {
      // If token is invalid and we have user info, try to refresh
      if (err.response && err.response.status === 401 && user) {
        try {
          // Clear existing token
          localStorage.removeItem(config.auth.tokenStorageKey);
          api.setAuthToken(null);
          
          // Try to login again with stored credentials
          // This is a simplified version for demo purposes
          const response = await api.auth.login(user.user_id, 'password');
          
          if (response && response.access_token) {
            // Set new token
            setToken(response.access_token);
            localStorage.setItem(config.auth.tokenStorageKey, response.access_token);
            api.setAuthToken(response.access_token);
            return response.access_token;
          }
        } catch (refreshErr) {
          console.error('Token refresh failed:', refreshErr);
          // If refresh fails, logout
          handleLogout();
          return null;
        }
      }
      
      return null;
    }
  };

  // Check API connection explicitly
  const checkApiConnection = async () => {
    const isConnected = await safeCheckConnection();
    setApiConnected(isConnected);
    return isConnected;
  };

  // Create value object with auth state and functions
  const value = {
    user,
    token,
    loading,
    error,
    userPersona,
    isAuthenticated,
    apiConnected, // Expose API connection status
    login,
    register,
    logout: handleLogout,
    getToken,
    saveUserPersona,
    checkApiConnection
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}; 