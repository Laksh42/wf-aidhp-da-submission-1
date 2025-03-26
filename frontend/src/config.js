// Configuration for the Financial Advisor frontend application

const environment = process.env.NODE_ENV || 'development';

// Configuration for different environments
const envConfig = {
  development: {
    baseApiUrl: 'http://localhost:8000', // Explicitly set to backend server
    debug: true
  },
  production: {
    baseApiUrl: '',  // Base URL without /api in production
    debug: false
  }
};

// Export the configuration
export const config = {
  // Base URL for API requests
  apiUrl: envConfig[environment].baseApiUrl,
  
  // API structure for the API module
  api: {
    baseUrl: envConfig[environment].baseApiUrl
  },
  
  // Application settings
  app: {
    name: 'Wells Fargo Financial Assistant',
    version: '1.0.0',
    logoUrl: '/logo.png'
  },
  
  // Authentication settings
  auth: {
    tokenStorageKey: 'token',
    userStorageKey: 'currentUser',
    refreshTokenKey: 'refreshToken',
    expiryKey: 'tokenExpiry'
  },
  
  // Chat interface settings
  chat: {
    maxMessages: 10,
    typingIndicatorDelay: 1000,
    aiResponseDelay: 800,
    sessionStorageKey: 'chatSession',
    welcomeMessage: 'Welcome to the Wells Fargo Financial Assistant! I\'m here to help you with your financial goals.'
  },
  
  // API endpoints
  endpoints: {
    auth: {
      register: '/api/auth/register',
      login: '/api/auth/token',
      verify: '/api/auth/verify',
      me: '/api/auth/me',
      userData: '/api/auth/user-data'
    },
    recommendations: {
      list: '/api/recommendations/',
      details: '/api/recommendations/:id',
      refresh: '/api/recommendations/',
      feedback: '/api/recommendations/feedback'
    },
    chat: {
      send: '/api/chat/send',
      history: '/api/chat/history'
    }
  },
  
  // Recommendation settings
  recommendations: {
    max: 5,
    storageKey: 'financial_advisor_recommendations',
    refreshInterval: 24 * 60 * 60 * 1000 // 24 hours
  },
  
  // Feature flags
  features: {
    enableMockData: false, // Using real backend with proper transformations now
    enableDebug: envConfig[environment].debug
  },
  
  // Services (for component-specific integrations)
  services: {
    understander: {
      url: envConfig[environment].baseApiUrl
    }
  },
  
  // Debug settings
  debug: {
    logApiCalls: envConfig[environment].debug
  },
  
  // UI Configuration
  ui: {
    theme: {
      primary: '#D41B2C', // Wells Fargo red
      secondary: '#FFCD34', // Wells Fargo yellow
    },
    layout: {
      sidebarWidth: 240,
      headerHeight: 64
    }
  }
}; 