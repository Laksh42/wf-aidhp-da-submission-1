import React from 'react';
import { Box, CircularProgress } from '@mui/material';
import { styled, keyframes } from '@mui/material/styles';

// Animation for the dots
const bounce = keyframes`
  0%, 80%, 100% { 
    transform: translateY(0);
  }
  40% { 
    transform: translateY(-5px);
  }
`;

const DotContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(1),
  boxShadow: theme.shadows[1],
  maxWidth: 'fit-content',
}));

const Dot = styled('span')(({ theme, delay }) => ({
  width: '8px',
  height: '8px',
  margin: '0 3px',
  borderRadius: '50%',
  backgroundColor: theme.palette.primary.main,
  display: 'inline-block',
  animation: `${bounce} 1.4s infinite ease-in-out both`,
  animationDelay: `${delay}s`,
}));

/**
 * A typing indicator component that shows animated dots
 * to indicate that someone is typing a message.
 */
const TypingIndicator = () => {
  return (
    <DotContainer>
      <Dot delay={0} />
      <Dot delay={0.2} />
      <Dot delay={0.4} />
    </DotContainer>
  );
};

export default TypingIndicator; 