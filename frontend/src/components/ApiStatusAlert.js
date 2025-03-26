import React from 'react';
import { Alert, Button, Box, Typography } from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import { config } from '../config';

/**
 * Component to display API connectivity issues with a retry button
 */
const ApiStatusAlert = () => {
  const { apiConnected, checkApiConnection, error } = useAuth();
  
  if (apiConnected && !error) {
    return null;
  }
  
  const handleRetry = async () => {
    await checkApiConnection();
  };
  
  // Format error message
  const formatErrorMessage = (error) => {
    if (!error) return '';
    
    if (typeof error === 'string') {
      return error;
    }
    
    if (typeof error === 'object') {
      // If it's a validation error object from the API
      if (error.detail && Array.isArray(error.detail)) {
        return error.detail.map(err => err.msg || String(err)).join(', ');
      }
      
      // Handle other error object formats
      return JSON.stringify(error);
    }
    
    return String(error);
  };
  
  return (
    <Box mb={2}>
      <Alert 
        severity="error" 
        action={
          <Button 
            color="inherit" 
            size="small" 
            onClick={handleRetry}
          >
            Retry
          </Button>
        }
      >
        <Typography variant="body2">
          {!apiConnected ? 
            `Cannot connect to authentication server at ${config.apiUrl}. Please check your connection or contact support.` 
            : formatErrorMessage(error)
          }
        </Typography>
        <Typography variant="caption" component="div" mt={1}>
          If the problem persists, please ensure the backend server is running at {config.apiUrl}.
        </Typography>
      </Alert>
    </Box>
  );
};

export default ApiStatusAlert; 