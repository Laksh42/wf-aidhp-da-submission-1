import React, { useState } from 'react';
import {
  Fab,
  Dialog,
  DialogContent,
  IconButton,
  Box,
  useTheme,
  useMediaQuery
} from '@mui/material';
import {
  Chat as ChatIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import ChatInterface from '../components/ChatInterface';

const ChatButton = () => {
  const [open, setOpen] = useState(false);
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down('md'));

  const handleOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  return (
    <>
      {/* Floating Chat Button */}
      <Fab
        color="primary"
        aria-label="chat"
        onClick={handleOpen}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          zIndex: 1000,
          boxShadow: theme.shadows[8],
        }}
      >
        <ChatIcon />
      </Fab>

      {/* Chat Dialog */}
      <Dialog
        open={open}
        onClose={handleClose}
        fullScreen={fullScreen}
        maxWidth="md"
        PaperProps={{
          elevation: 12,
          sx: {
            height: fullScreen ? '100%' : '80vh',
            width: fullScreen ? '100%' : '80%',
            maxWidth: 800,
            borderRadius: '12px',
            overflow: 'hidden'
          }
        }}
      >
        <Box sx={{ position: 'absolute', top: 8, right: 8, zIndex: 1 }}>
          <IconButton
            edge="end"
            color="inherit"
            onClick={handleClose}
            aria-label="close"
          >
            <CloseIcon />
          </IconButton>
        </Box>
        
        <DialogContent sx={{ padding: 0, display: 'flex', flexDirection: 'column', height: '100%' }}>
          <ChatInterface />
        </DialogContent>
      </Dialog>
    </>
  );
};

export default ChatButton; 