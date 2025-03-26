import React, { useState, useEffect, useRef, useContext } from 'react';
import { 
  Container, 
  Paper, 
  Typography, 
  TextField, 
  Button, 
  Box, 
  CircularProgress, 
  Alert,
  Divider,
  Card,
  CardContent,
  Grid,
  AppBar,
  Toolbar,
  IconButton,
  Avatar
} from '@mui/material';
import { 
  Send as SendIcon, 
  AutoAwesome as AutoAwesomeIcon,
  AttachFile as AttachFileIcon
} from '@mui/icons-material';
import axios from 'axios';
import { config } from '../config';
import { useAuth } from '../contexts/AuthContext';
import api from '../api';
import { styled } from '@mui/material/styles';
import RecommendationList from './RecommendationList';
import TypingIndicator from './TypingIndicator';
import MarkdownRenderer from './MarkdownRenderer';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

const MessageContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  marginBottom: theme.spacing(2),
  alignItems: 'flex-start',
}));

const MessageContent = styled(Box)(({ theme, isUser }) => ({
  padding: theme.spacing(2),
  borderRadius: theme.spacing(2),
  maxWidth: '70%',
  backgroundColor: isUser ? theme.palette.primary.main : theme.palette.background.paper,
  color: isUser ? theme.palette.primary.contrastText : theme.palette.text.primary,
  boxShadow: theme.shadows[1],
  marginLeft: isUser ? 'auto' : '10px',
  marginRight: isUser ? '10px' : 'auto',
  position: 'relative',
  wordBreak: 'break-word',
}));

const ChatContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  display: 'flex',
  flexDirection: 'column',
  height: 'calc(100vh - 64px)',
  position: 'relative',
  backgroundColor: theme.palette.mode === 'dark' 
      ? theme.palette.background.default 
      : theme.palette.grey[100],
}));

const MessagesBox = styled(Box)(({ theme }) => ({
  flexGrow: 1,
  overflowY: 'auto',
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

const InputArea = styled(Box)(({ theme }) => ({
  display: 'flex',
  padding: theme.spacing(1),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[1],
}));

const TypingBox = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1),
  position: 'absolute',
  bottom: theme.spacing(8),
  left: theme.spacing(8),
}));

const FeedbackContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'flex-end',
  marginTop: theme.spacing(1),
}));

const ChatInterface = ({ isOnboarding = false, isFullWidth = false }) => {
  // State for chat messages
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const [userPersona, setUserPersona] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  
  // Reference for auto-scrolling to bottom
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const { saveUserPersona, isAuthenticated, user, login, getToken } = useAuth();

  // Initialize chat with welcome message
  useEffect(() => {
    // Load existing session or start new one
    const savedSession = sessionStorage.getItem(config.chat.sessionStorageKey);
    
    if (savedSession) {
      try {
        const parsedSession = JSON.parse(savedSession);
        setMessages(parsedSession.messages || []);
        setSessionId(parsedSession.sessionId || '');
        setIsComplete(parsedSession.isComplete || false);
      } catch (e) {
        console.error('Error parsing saved session:', e);
        initializeChat();
      }
    } else {
      initializeChat();
    }
  }, []);

  // Function to initialize a new chat
  const initializeChat = async () => {
    // Check for existing session in session storage
    const savedSession = sessionStorage.getItem(config.chat.sessionStorageKey);
    
    if (savedSession) {
      try {
        const session = JSON.parse(savedSession);
        setMessages(session.messages || []);
        setSessionId(session.sessionId);
        setIsComplete(session.isComplete || false);
      } catch (error) {
        console.error('Error parsing saved session:', error);
        // If there's an error parsing the saved session, start fresh
        sessionStorage.removeItem(config.chat.sessionStorageKey);
        await startNewChat();
      }
    } else {
      await startNewChat();
    }
  };

  // Start a new chat session
  const startNewChat = async () => {
    // Add welcome message
    const welcomeMessage = {
      text: config.chat.welcomeMessage,
      sender: 'bot',
      timestamp: new Date().toISOString()
    };
    
    setMessages([welcomeMessage]);
    
    // Initialize chat with backend service
    if (!config.features.enableMockData) {
      try {
        // Send empty message to start session
        const response = await api.chat.sendMessage('', null);
        
        if (response && response.session_id) {
          setSessionId(response.session_id);
          
          // Add the bot's first question
          if (response.text || response.response) {
            setMessages(prevMessages => [
              ...prevMessages,
              {
                text: response.text || response.response,
                sender: 'bot',
                timestamp: new Date().toISOString()
              }
            ]);
          }
          
          // Save to session storage
          saveToSessionStorage(
            [welcomeMessage, {
              text: response.text || response.response,
              sender: 'bot',
              timestamp: new Date().toISOString()
            }],
            response.session_id,
            false
          );
        }
      } catch (error) {
        console.error('Error initializing chat:', error);
        setError('Failed to initialize chat. Please try again.');
      }
    }
  };

  // Save chat session to session storage
  const saveToSessionStorage = (messages, sessionId, isComplete) => {
    const sessionData = {
      messages,
      sessionId,
      isComplete,
      lastUpdated: new Date().toISOString()
    };
    
    sessionStorage.setItem(config.chat.sessionStorageKey, JSON.stringify(sessionData));
  };

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    // Add user message to chat
    const userMessage = {
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    
    console.log('Sending user message:', userMessage);
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputMessage('');
    setLoading(true);
    setIsTyping(true);
    setError('');
    
    // For demo purposes with mock data
    if (config.features.enableMockData) {
      console.log('Mock data enabled, returning mock response');
      setTimeout(() => {
        const mockResponse = {
          text: `Thanks for your message: "${inputMessage}". I'm a mock response as we're in development mode. In production, I would connect to our AI backend.`,
          sender: 'bot',
          timestamp: new Date().toISOString()
        };
        
        setMessages(prevMessages => [...prevMessages, mockResponse]);
        setLoading(false);
        
        // Save updated chat to session storage
        saveToSessionStorage(
          [...messages, userMessage, mockResponse],
          sessionId || 'mock-session-id',
          false
        );
      }, 1000);
      return;
    }
    
    try {
      // Try to ensure we have a valid token before making the request
      await getToken();
      
      console.log('Sending chat message to backend:', inputMessage, 'with session ID:', sessionId);
      // Send message to backend service using our API client
      const response = await api.chat.sendMessage(inputMessage, sessionId);
      console.log('Received response from chat API:', response);
      
      // Update session ID if needed
      if (response && response.session_id) {
        console.log('Setting session ID from response:', response.session_id);
        setSessionId(response.session_id);
      }
      
      // Add bot response to chat
      if (response && (response.text || response.response)) {
        console.log('Adding bot response to chat:', response.text || response.response);
        const botMessage = {
          text: response.text || response.response,
          sender: 'bot',
          timestamp: new Date().toISOString()
        };
        
        setMessages(prevMessages => [...prevMessages, botMessage]);
        
        // Check if dialogue is complete
        if (response.state && response.state.is_complete) {
          console.log('Dialog is complete, extracting user persona');
          setIsComplete(true);
          
          // Extract user persona
          extractUserPersona(response.session_id);
        }
        
        // Save updated chat to session storage
        saveToSessionStorage(
          [...messages, userMessage, botMessage],
          response.session_id,
          response.state?.is_complete || false
        );
        
        // Update recommendations
        if (response.recommendations && response.recommendations.length > 0) {
          console.log('Setting recommendations from response:', response.recommendations);
          setRecommendations(response.recommendations);
        }
      } else {
        console.warn('No response text received from API:', response);
        // Add fallback message if no text in response
        setMessages(prevMessages => [
          ...prevMessages,
          {
            text: "I received your message but didn't get a proper response. Please try again.",
            sender: 'bot',
            timestamp: new Date().toISOString(),
            isError: true
          }
        ]);
      }
    } catch (err) {
      console.error('Error sending message:', err);
      
      // If token expired (401), refresh it and try again
      if (err.response && err.response.status === 401) {
        console.log('Token expired (401), attempting to refresh');
        try {
          // Force refresh the token
          await login(user?.user_id || 'testuser', 'password', true);
          
          // Retry the message with new token
          console.log('Retrying message with new token');
          const response = await api.chat.sendMessage(inputMessage, sessionId);
          console.log('Received response on retry:', response);
          
          // Add bot response to chat
          if (response && (response.text || response.response)) {
            console.log('Adding bot response after token refresh:', response.text || response.response);
            const botMessage = {
              text: response.text || response.response,
              sender: 'bot',
              timestamp: new Date().toISOString()
            };
            
            setMessages(prevMessages => [...prevMessages, botMessage]);
            
            // Check if dialogue is complete
            if (response.state && response.state.is_complete) {
              setIsComplete(true);
              
              // Extract user persona
              extractUserPersona(response.session_id);
            }
            
            // Save updated chat to session storage
            saveToSessionStorage(
              [...messages, userMessage, botMessage],
              response.session_id,
              response.state?.is_complete || false
            );
            
            // Update recommendations
            if (response.recommendations && response.recommendations.length > 0) {
              setRecommendations(response.recommendations);
            }
          } else {
            console.warn('No response text received on retry');
            // Add fallback message if no text in response
            setMessages(prevMessages => [
              ...prevMessages,
              {
                text: "I received your message after reconnecting but didn't get a proper response. Please try again.",
                sender: 'bot',
                timestamp: new Date().toISOString(),
                isError: true
              }
            ]);
          }
        } catch (retryErr) {
          console.error('Retry after token refresh failed:', retryErr);
          // If retry fails, show error
          setMessages(prev => [
            ...prev, 
            {
              text: "I'm sorry, I'm having trouble connecting to my services right now. Please try again later.",
              sender: 'bot',
              timestamp: new Date().toISOString(),
              isError: true
            }
          ]);
          setError("Authentication error. Please log out and log in again.");
        }
      } else {
        console.error('Other error sending message:', err);
        // Show error message
        setMessages(prev => [
          ...prev, 
          {
            text: "I apologize, but I'm experiencing some technical difficulties. Please try again later.",
            sender: 'bot',
            timestamp: new Date().toISOString(),
            isError: true
          }
        ]);
        setError("Unable to process your request at this time.");
      }
    } finally {
      setLoading(false);
      setIsTyping(false);
    }
  };

  // Extract user persona from dialogue history
  const extractUserPersona = async (sid) => {
    try {
      // This would call to the backend to get user profile data
      // For now, we'll create a mock persona for demonstration
      const mockPersona = {
        age: 35,
        income: '75000-100000',
        job: 'Software Engineer',
        goals: ['Retirement', 'Home Purchase'],
        risk_tolerance: 'Moderate'
      };
      
      setUserPersona(mockPersona);
      return mockPersona;
    } catch (error) {
      console.error('Error extracting user persona:', error);
      setError('Failed to create your financial profile. Please try again.');
      return null;
    }
  };

  // Handle key press (Enter to send)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Handle retrying if there was an error
  const handleRetry = () => {
    setError('');
    // The last message would be the user's, so we try to resend it
    if (messages.length > 0 && messages[messages.length - 1].sender === 'user') {
      setInputMessage(messages[messages.length - 1].text);
      // Remove the last message since we're retrying it
      setMessages(prevMessages => prevMessages.slice(0, -1));
    }
  };

  // Reset the chat
  const handleResetChat = () => {
    // Clear session storage
    sessionStorage.removeItem(config.chat.sessionStorageKey);
    
    // Reinitialize chat
    initializeChat();
  };

  const handleFileButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleFileSelect = (e) => {
    // TODO: Implement file handling
    console.log('File selected:', e.target.files[0]);
  };

  const handleFeedback = (messageId, isPositive) => {
    // TODO: Implement feedback API call
    console.log(`Feedback for message ${messageId}: ${isPositive ? 'positive' : 'negative'}`);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* App Bar - Only show in standalone mode */}
      {!isOnboarding && false && (
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Financial Assistant Chat
            </Typography>
          </Toolbar>
        </AppBar>
      )}

      <Container maxWidth="md" sx={{ flex: 1, display: 'flex', flexDirection: 'column', py: 2 }}>
        {/* Information card for onboarding */}
        {isOnboarding && (
          <Card sx={{ mb: 3, bgcolor: 'primary.light', color: 'white' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <AutoAwesomeIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Welcome to Your Financial Profile Setup
              </Typography>
              <Typography variant="body2">
                I'll ask you a few questions to understand your financial situation and goals.
                This helps me provide personalized recommendations tailored to your needs.
                Your information is kept secure and will only be used to improve your experience.
              </Typography>
            </CardContent>
          </Card>
        )}

        {/* Chat area */}
        <ChatContainer elevation={3} sx={{ width: isFullWidth ? '100%' : '800px' }}>
          <MessagesBox>
            {messages.map((message, index) => (
              <MessageContainer key={index} sx={{ flexDirection: message.sender === 'user' ? 'row-reverse' : 'row' }}>
                <Avatar 
                  sx={{ 
                    bgcolor: message.sender === 'user' ? 'primary.main' : 'secondary.main',
                    width: 36, 
                    height: 36,
                    mr: message.sender === 'user' ? 0 : 1,
                    ml: message.sender === 'user' ? 1 : 0
                  }}
                >
                  {message.sender === 'user' ? user?.user_id?.charAt(0).toUpperCase() || 'U' : 'W'}
                </Avatar>
                <MessageContent isUser={message.sender === 'user'}>
                  {message.sender === 'user' ? (
                    <Typography variant="body1">{message.text}</Typography>
                  ) : (
                    <>
                      {message.isError ? (
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <ErrorOutlineIcon color="error" sx={{ mr: 1 }} />
                          <Typography variant="body2" color="error.main">
                            Error
                          </Typography>
                        </Box>
                      ) : null}
                      <Box sx={{ mt: 1 }}>
                        <MarkdownRenderer content={message.text} />
                      </Box>
                      {message.sender === 'bot' && !message.isError && (
                        <FeedbackContainer>
                          <IconButton 
                            size="small" 
                            onClick={() => handleFeedback(message.id, true)}
                            sx={{ color: 'text.secondary' }}
                          >
                            <ThumbUpIcon fontSize="small" />
                          </IconButton>
                          <IconButton 
                            size="small" 
                            onClick={() => handleFeedback(message.id, false)}
                            sx={{ color: 'text.secondary' }}
                          >
                            <ThumbDownIcon fontSize="small" />
                          </IconButton>
                        </FeedbackContainer>
                      )}
                    </>
                  )}
                </MessageContent>
              </MessageContainer>
            ))}
            {isTyping && (
              <TypingBox>
                <TypingIndicator />
              </TypingBox>
            )}
            <div ref={messagesEndRef} />
          </MessagesBox>
          
          {/* Error message */}
          {error && (
            <Box sx={{ mb: 2 }}>
              <Card sx={{ bgcolor: 'error.light' }}>
                <CardContent>
                  <Typography color="error.dark">
                    {error}
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          )}
          
          {recommendations.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <RecommendationList recommendations={recommendations} />
            </Box>
          )}
          
          {/* Input area */}
          <InputArea>
            <TextField
              variant="outlined"
              size="small"
              fullWidth
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              disabled={loading}
              multiline
              maxRows={4}
              sx={{ mr: 1 }}
            />
            <input
              type="file"
              style={{ display: 'none' }}
              ref={fileInputRef}
              onChange={handleFileSelect}
            />
            <IconButton 
              color="primary" 
              onClick={handleFileButtonClick}
              disabled={loading}
            >
              <AttachFileIcon />
            </IconButton>
            <Button
              variant="contained"
              color="primary"
              endIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
              onClick={handleSendMessage}
              disabled={loading || !inputMessage.trim()}
            >
              Send
            </Button>
          </InputArea>
        </ChatContainer>
      </Container>
    </Box>
  );
};

export default ChatInterface; 