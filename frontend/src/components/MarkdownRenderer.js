import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Typography, Box, Link, Paper } from '@mui/material';
import { styled } from '@mui/material/styles';

const MarkdownContainer = styled(Box)(({ theme }) => ({
  '& p': {
    marginTop: theme.spacing(1),
    marginBottom: theme.spacing(1),
  },
  '& h1, & h2, & h3, & h4, & h5, & h6': {
    marginTop: theme.spacing(2),
    marginBottom: theme.spacing(1),
    fontWeight: 600,
  },
  '& h1': {
    fontSize: '1.7rem',
  },
  '& h2': {
    fontSize: '1.5rem',
  },
  '& h3': {
    fontSize: '1.3rem',
  },
  '& h4': {
    fontSize: '1.2rem',
  },
  '& h5': {
    fontSize: '1.1rem',
  },
  '& h6': {
    fontSize: '1rem',
  },
  '& ul, & ol': {
    paddingLeft: theme.spacing(3),
    marginBottom: theme.spacing(2),
  },
  '& li': {
    marginBottom: theme.spacing(0.5),
  },
  '& blockquote': {
    borderLeft: `4px solid ${theme.palette.grey[300]}`,
    paddingLeft: theme.spacing(2),
    fontStyle: 'italic',
    margin: theme.spacing(2, 0),
    color: theme.palette.text.secondary,
  },
  '& code': {
    backgroundColor: theme.palette.grey[100],
    padding: theme.spacing(0.5),
    borderRadius: '3px',
    fontFamily: 'monospace',
    fontSize: '0.9rem',
  },
  '& pre': {
    backgroundColor: theme.palette.grey[100],
    padding: theme.spacing(1.5),
    borderRadius: theme.shape.borderRadius,
    overflowX: 'auto',
    '& code': {
      backgroundColor: 'transparent',
      padding: 0,
    },
  },
  '& table': {
    borderCollapse: 'collapse',
    width: '100%',
    marginBottom: theme.spacing(2),
    '& th, & td': {
      border: `1px solid ${theme.palette.grey[300]}`,
      padding: theme.spacing(0.5, 1),
    },
    '& th': {
      backgroundColor: theme.palette.grey[200],
      fontWeight: 'bold',
    },
  },
  '& a': {
    color: theme.palette.primary.main,
    textDecoration: 'none',
    '&:hover': {
      textDecoration: 'underline',
    },
  },
  '& img': {
    maxWidth: '100%',
    height: 'auto',
    borderRadius: theme.shape.borderRadius,
    margin: theme.spacing(1, 0),
  },
}));

/**
 * Component for rendering Markdown content with MUI styling
 * 
 * @param {Object} props
 * @param {string} props.content - The markdown content to render
 */
const MarkdownRenderer = ({ content }) => {
  if (!content) return null;

  return (
    <MarkdownContainer>
      <ReactMarkdown
        components={{
          h1: ({ node, ...props }) => <Typography variant="h4" {...props} />,
          h2: ({ node, ...props }) => <Typography variant="h5" {...props} />,
          h3: ({ node, ...props }) => <Typography variant="h6" {...props} />,
          h4: ({ node, ...props }) => <Typography variant="subtitle1" fontWeight="bold" {...props} />,
          h5: ({ node, ...props }) => <Typography variant="subtitle2" fontWeight="bold" {...props} />,
          h6: ({ node, ...props }) => <Typography variant="body1" fontWeight="bold" {...props} />,
          p: ({ node, ...props }) => <Typography variant="body1" paragraph {...props} />,
          a: ({ node, ...props }) => <Link {...props} target="_blank" rel="noopener noreferrer" />,
          code: ({ node, inline, ...props }) => 
            inline ? 
              <code style={{ backgroundColor: '#f5f5f5', padding: '2px 4px', borderRadius: '3px' }} {...props} /> : 
              <Paper elevation={0} sx={{ bgcolor: '#f5f5f5', p: 1.5, my: 1.5, overflowX: 'auto' }}>
                <code style={{ display: 'block', fontFamily: 'monospace' }} {...props} />
              </Paper>,
          pre: ({ node, ...props }) => 
            <Box component="pre" sx={{ my: 2, p: 0 }} {...props} />,
        }}
      >
        {content}
      </ReactMarkdown>
    </MarkdownContainer>
  );
};

export default MarkdownRenderer; 