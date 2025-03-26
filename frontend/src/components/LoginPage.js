import React, { useState } from 'react';
import { 
  Box, 
  Container,
  Typography, 
  TextField, 
  Button, 
  Link, 
  Alert,
  Paper,
  Grid,
  CircularProgress
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { config } from '../config';
import ApiStatusAlert from './ApiStatusAlert';

const LoginPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');

  const { login, register, error: authError } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Reset error and success messages
    setError('');
    setSuccessMessage('');
    setLoading(true);
    
    try {
      // Validate form
      if (!username || !password) {
        setError('Username and password are required');
        setLoading(false);
        return;
      }
      
      if (!isLogin) {
        // Password strength check
        if (password.length < 6) {
          setError('Password must be at least 6 characters');
          setLoading(false);
          return;
        }
        
        // Confirm password match
        if (password !== confirmPassword) {
          setError('Passwords do not match');
          setLoading(false);
          return;
        }
      }

      if (isLogin) {
        console.log('Attempting login with:', username);
        // Use the AuthContext login function
        const result = await login(username, password);
        
        if (result.success) {
          console.log('Login successful, navigating to dashboard');
          // Navigate to the recommendations dashboard
          navigate('/dashboard');
        } else {
          setError(result.error || 'Login failed');
        }
      } else {
        console.log('Attempting registration for:', username);
        // Create a user object with only the fields that the backend accepts
        const userData = {
          user_id: username,
          password: password,
          email: `${username}@example.com`,
          full_name: username
        };
        
        console.log('Sending registration data:', userData);
        
        // Use the AuthContext register function
        const success = await register(userData);
        
        if (success) {
          // Show success message for registration
          setSuccessMessage('Registration successful! You can now log in.');
          setIsLogin(true);
        } else {
          // Use the error from AuthContext if available
          setError(authError || 'Registration failed. Please try again.');
        }
      }
    } catch (error) {
      console.error('Auth error:', error);
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const toggleAuthMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setSuccessMessage('');
  };

  // Helper function to format error messages
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
      
      // Try to extract error message from various API formats
      if (error.message) return error.message;
      if (error.error) return error.error;
      
      // Handle other error object formats
      return JSON.stringify(error);
    }
    
    return String(error);
  };

  return (
    <Container component="main" maxWidth="md">
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <ApiStatusAlert />
        
        <Paper elevation={3} sx={{ p: 4, width: '100%' }}>
          <Grid container spacing={4}>
            <Grid item xs={12} md={6} sx={{ 
              display: 'flex', 
              flexDirection: 'column', 
              justifyContent: 'center',
              borderRight: { md: '1px solid #e0e0e0' },
              pr: { md: 4 }
            }}>
              <Typography component="h1" variant="h4" align="center" gutterBottom>
                {config.app.name}
              </Typography>
              <Typography variant="body1" align="center" color="text.secondary" paragraph>
                Your personalized financial advisor to help you achieve your financial goals.
                Get tailored recommendations based on your unique financial situation.
              </Typography>
              <Box sx={{ 
                mt: 2, 
                display: 'flex', 
                justifyContent: 'center',
                '& img': {
                  maxWidth: '100%',
                  height: 'auto'
                }
              }}>
                <img 
                  src="/logo.png" 
                  alt="Financial Advisor Logo" 
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.style.display = 'none';
                  }}
                />
              </Box>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography component="h2" variant="h5" align="center">
                {isLogin ? 'Sign In' : 'Create Account'}
              </Typography>
              
              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {formatErrorMessage(error)}
                </Alert>
              )}
              
              {successMessage && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  {successMessage}
                </Alert>
              )}
              
              <Box component="form" onSubmit={handleSubmit} sx={{ mt: 3 }}>
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="username"
                  label="Username"
                  name="username"
                  autoComplete="username"
                  autoFocus
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                />
                
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  name="password"
                  label="Password"
                  type="password"
                  id="password"
                  autoComplete={isLogin ? "current-password" : "new-password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                
                {!isLogin && (
                  <TextField
                    margin="normal"
                    required
                    fullWidth
                    name="confirmPassword"
                    label="Confirm Password"
                    type="password"
                    id="confirmPassword"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                  />
                )}
                
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  sx={{ mt: 3, mb: 2 }}
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : (isLogin ? 'Sign In' : 'Create Account')}
                </Button>
                
                <Box sx={{ textAlign: 'center', mt: 2 }}>
                  <Link href="#" variant="body2" onClick={toggleAuthMode}>
                    {isLogin ? "Don't have an account? Sign Up" : "Already have an account? Sign In"}
                  </Link>
                </Box>
                
                {isLogin && (
                  <Box sx={{ textAlign: 'center', mt: 1 }}>
                    <Link href="#" variant="body2">
                      Forgot password?
                    </Link>
                  </Box>
                )}
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    </Container>
  );
};

export default LoginPage; 