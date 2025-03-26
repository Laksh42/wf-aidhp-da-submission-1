import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Container, 
  Grid, 
  Paper, 
  Typography, 
  Button, 
  CircularProgress,
  Divider,
  Card,
  CardContent,
  AppBar,
  Toolbar,
  IconButton,
  Menu,
  MenuItem,
  Avatar,
  Alert
} from '@mui/material';
import { 
  AccountCircle,
  RefreshOutlined,
  Dashboard as DashboardIcon,
  Logout as LogoutIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { config } from '../config';
import RecommendationCard from '../components/RecommendationCard';
import ChatButton from '../components/ChatButton';
import api from '../api';

const DashboardPage = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [userInsights, setUserInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);
  
  const { currentUser, logout } = useAuth();
  const navigate = useNavigate();
  
  // Handle profile menu open/close
  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };
  
  // Handle logout
  const handleLogout = () => {
    logout();
    navigate('/');
  };
  
  // Fetch recommendations from API
  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      setError('');
      
      const data = await api.recommendations.getRecommendations();
      
      if (data) {
        console.log('Recommendations data:', data);
        setRecommendations(data.recommendations || []);
        
        // If user insights are available in the response
        if (data.user_insights) {
          setUserInsights(data.user_insights);
        }
      }
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError('Failed to load recommendations. Please try again later.');
    } finally {
      setLoading(false);
    }
  };
  
  // Refresh recommendations
  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      setError('');
      
      await api.recommendations.refreshRecommendations();
      
      // Fetch the updated recommendations
      await fetchRecommendations();
      
    } catch (error) {
      console.error('Error refreshing recommendations:', error);
      setError('Failed to refresh recommendations. Please try again later.');
    } finally {
      setRefreshing(false);
    }
  };
  
  // Load recommendations on component mount
  useEffect(() => {
    fetchRecommendations();
  }, []);
  
  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Top App Bar */}
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Financial Advisor Dashboard
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {currentUser && (
              <Typography variant="body2" sx={{ mr: 2 }}>
                Welcome, {currentUser.username || 'User'}
              </Typography>
            )}
            
            <IconButton
              size="large"
              aria-label="account of current user"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleMenu}
              color="inherit"
            >
              <AccountCircle />
            </IconButton>
            
            <Menu
              id="menu-appbar"
              anchorEl={anchorEl}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={Boolean(anchorEl)}
              onClose={handleClose}
            >
              <MenuItem onClick={handleClose}>
                <DashboardIcon fontSize="small" sx={{ mr: 1 }} />
                Dashboard
              </MenuItem>
              
              <MenuItem onClick={handleLogout}>
                <LogoutIcon fontSize="small" sx={{ mr: 1 }} />
                Logout
              </MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>
      
      {/* Main Content */}
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          {/* Page Header */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography component="h1" variant="h5">
                Your Personalized Financial Recommendations
              </Typography>
              
              <Button 
                variant="outlined" 
                startIcon={<RefreshOutlined />}
                onClick={handleRefresh}
                disabled={refreshing}
              >
                {refreshing ? 'Refreshing...' : 'Refresh Recommendations'}
              </Button>
            </Paper>
          </Grid>
          
          {/* Error Message (if any) */}
          {error && (
            <Grid item xs={12}>
              <Alert severity="error">{error}</Alert>
            </Grid>
          )}
          
          {/* User Insights Section */}
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Your Financial Snapshot
              </Typography>
              
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
                  <CircularProgress />
                </Box>
              ) : userInsights ? (
                <Box>
                  {userInsights.account && (
                    <Card variant="outlined" sx={{ mb: 2 }}>
                      <CardContent>
                        <Typography variant="subtitle2" color="text.secondary">
                          Account Balance
                        </Typography>
                        <Typography variant="h4">
                          ${userInsights.account.balance.toLocaleString()}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {userInsights.account.account_type} Account
                        </Typography>
                      </CardContent>
                    </Card>
                  )}
                  
                  {userInsights.demographics && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Profile
                      </Typography>
                      <Typography variant="body2">
                        <strong>Age:</strong> {userInsights.demographics.age}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Occupation:</strong> {userInsights.demographics.occupation}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Income Bracket:</strong> {userInsights.demographics.income_bracket}
                      </Typography>
                    </Box>
                  )}
                  
                  {userInsights.credit_score && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Credit Score
                      </Typography>
                      <Typography variant="h5">
                        {userInsights.credit_score}
                      </Typography>
                    </Box>
                  )}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No financial data available at this time.
                </Typography>
              )}
            </Paper>
          </Grid>
          
          {/* Recommendations Section */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Recommended Financial Products
              </Typography>
              
              <Divider sx={{ mb: 2 }} />
              
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4, mb: 4 }}>
                  <CircularProgress />
                </Box>
              ) : recommendations && recommendations.length > 0 ? (
                <Box>
                  {recommendations.map((product, index) => (
                    <RecommendationCard 
                      key={product.id || index}
                      product={product}
                      index={index}
                    />
                  ))}
                </Box>
              ) : (
                <Alert severity="info" sx={{ mt: 2 }}>
                  No recommendations available. Please refresh or check back later.
                </Alert>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Container>
      
      {/* Chat Button */}
      <ChatButton />
    </Box>
  );
};

export default DashboardPage; 