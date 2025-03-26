import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Chip,
  Box,
  IconButton,
  Collapse,
  Tooltip,
  Divider,
  Rating
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import LocalOfferIcon from '@mui/icons-material/LocalOffer';
import { 
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  Star as StarIcon
} from '@mui/icons-material';
import axios from 'axios';
import { config } from '../config';
import api from '../api';

const RecommendationCard = ({ product, index }) => {
  const [expanded, setExpanded] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleExpandClick = () => {
    setExpanded(!expanded);
  };

  const provideFeedback = async (isRelevant) => {
    if (feedbackGiven) return;
    
    setLoading(true);
    try {
      await api.recommendations.provideFeedback(product.id, isRelevant);
      
      setFeedbackGiven(true);
    } catch (error) {
      console.error('Error providing feedback:', error);
    } finally {
      setLoading(false);
    }
  };

  // Determine the appropriate icon color based on recommendation confidence or score
  const getConfidenceColor = (score) => {
    if (score >= 0.8) return 'success.main';
    if (score >= 0.6) return 'warning.main';
    return 'info.main';
  };

  // Truncate long descriptions for the main view
  const truncateText = (text, maxLength = 120) => {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <Card 
      variant="outlined" 
      sx={{ 
        mb: 2,
        borderLeft: product.priority === 'high' ? '4px solid' : 'none',
        borderLeftColor: 'primary.main',
        transition: 'transform 0.3s ease',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: 3
        }
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box>
            <Typography variant="h6" component="div">
              {product.name}
            </Typography>
            
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
              {product.rating && (
                <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
                  <Rating 
                    value={product.rating} 
                    readOnly 
                    size="small"
                    precision={0.5}
                    icon={<StarIcon fontSize="inherit" />}
                  />
                </Box>
              )}
              
              {product.category && (
                <Chip 
                  label={product.category} 
                  size="small" 
                  color="secondary" 
                  variant="outlined"
                  sx={{ mr: 1 }}
                />
              )}
              
              {product.tags && product.tags.map((tag, idx) => (
                <Chip 
                  key={idx}
                  label={tag}
                  size="small"
                  sx={{ mr: 1, fontSize: '0.7rem' }}
                />
              ))}
            </Box>
            
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {product.short_description || product.description?.substring(0, 120) + '...'}
            </Typography>
          </Box>
          
          {product.priority === 'high' && (
            <Chip 
              label="Top Recommendation" 
              color="primary"
              size="small"
              sx={{ fontWeight: 'bold' }}
            />
          )}
        </Box>
        
        {/* Key features */}
        {product.key_features && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Key Features:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 0.5 }}>
              {product.key_features.map((feature, idx) => (
                <Chip 
                  key={idx}
                  label={feature}
                  size="small"
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        )}
        
        {/* Call to action buttons */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mt: 2 }}>
          <Button 
            variant="contained" 
            color="primary"
            size="small"
            href={product.link || '#'}
            target="_blank"
            rel="noopener noreferrer"
          >
            Learn More
          </Button>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
              Was this helpful?
            </Typography>
            
            <IconButton 
              size="small" 
              color={feedbackGiven ? 'primary' : 'default'}
              onClick={() => provideFeedback(true)}
              disabled={loading || feedbackGiven}
            >
              <ThumbUpIcon fontSize="small" />
            </IconButton>
            
            <IconButton 
              size="small" 
              color={feedbackGiven ? 'default' : 'default'}
              onClick={() => provideFeedback(false)}
              disabled={loading || feedbackGiven}
              sx={{ ml: 0.5 }}
            >
              <ThumbDownIcon fontSize="small" />
            </IconButton>
            
            <IconButton
              size="small"
              onClick={handleExpandClick}
              sx={{
                ml: 1,
                transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s'
              }}
            >
              <ExpandMoreIcon />
            </IconButton>
          </Box>
        </Box>
        
        {/* Expanded details */}
        <Collapse in={expanded} timeout="auto" unmountOnExit>
          <Divider sx={{ my: 2 }} />
          
          {product.description && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.primary">
                Description
              </Typography>
              <Typography variant="body2">
                {product.description}
              </Typography>
            </Box>
          )}
          
          {product.benefits && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.primary">
                Benefits
              </Typography>
              <ul style={{ margin: '8px 0', paddingLeft: 20 }}>
                {product.benefits.map((benefit, idx) => (
                  <li key={idx}>
                    <Typography variant="body2">{benefit}</Typography>
                  </li>
                ))}
              </ul>
            </Box>
          )}
          
          {product.why_recommended && (
            <Box sx={{ mb: 1, bgcolor: 'rgba(212, 27, 44, 0.05)', p: 1.5, borderRadius: 1 }}>
              <Typography variant="subtitle2" color="primary">
                Why We Recommended This For You
              </Typography>
              <Typography variant="body2">
                {product.why_recommended}
              </Typography>
            </Box>
          )}
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default RecommendationCard; 