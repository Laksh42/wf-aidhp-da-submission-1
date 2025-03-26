// init.js - Helper functions and polyfills for Financial Advisor Chatbot

// Console logging wrapper for debugging
function logDebug(message, data) {
  console.log(`[DEBUG] ${message}`, data || '');
}

// Add error handling for fetch calls
async function safeFetch(url, options = {}) {
  try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      // Try to parse error as JSON
      let errorData;
      try {
        errorData = await response.json();
      } catch (e) {
        // If can't parse as JSON, use status text
        errorData = { detail: response.statusText };
      }
      
      // Create a custom error with response details
      const error = new Error(`HTTP Error ${response.status}: ${JSON.stringify(errorData)}`);
      error.status = response.status;
      error.statusText = response.statusText;
      error.responseData = errorData;
      throw error;
    }
    
    return response;
  } catch (error) {
    logDebug('Fetch error:', error);
    throw error;
  }
}

// Check if browser supports async/await
if (typeof Promise === 'undefined' || typeof fetch === 'undefined') {
  console.error('Your browser does not support modern JavaScript features. Please upgrade to a newer browser.');
  
  // Add basic alert for really old browsers
  if (typeof alert !== 'undefined') {
    alert('Your browser is outdated. Please upgrade to a newer browser to use this application.');
  }
}

// Export functions to global scope
window.logDebug = logDebug;
window.safeFetch = safeFetch; 