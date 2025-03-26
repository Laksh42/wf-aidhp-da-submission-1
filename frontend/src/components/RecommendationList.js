import React from 'react';
import { 
  Box, 
  Typography, 
  List, 
  ListItem, 
  ListItemText, 
  Chip,
  Card,
  CardContent,
  Divider,
  Button
} from '@mui/material';
import { styled } from '@mui/material/styles';

const RecommendationTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 'bold',
  marginBottom: theme.spacing(1),
}));

const RecommendationContainer = styled(Box)(({ theme }) => ({
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

const RecommendationCard = styled(Card)(({ theme, type }) => {
  const colorMap = {
    'investment': theme.palette.info.light,
    'savings': theme.palette.success.light,
    'debt': theme.palette.warning.light,
    'tax': theme.palette.secondary.light,
    'insurance': theme.palette.error.light,
    'default': theme.palette.grey[200]
  };

  return {
    marginBottom: theme.spacing(1),
    borderLeft: `4px solid ${colorMap[type] || colorMap.default}`,
    '&:hover': {
      boxShadow: theme.shadows[3],
      transform: 'translateY(-2px)',
      transition: 'transform 0.3s ease, box-shadow 0.3s ease',
    }
  };
});

const TypeChip = styled(Chip)(({ theme, type }) => {
  const colorMap = {
    'investment': theme.palette.info.main,
    'savings': theme.palette.success.main,
    'debt': theme.palette.warning.main,
    'tax': theme.palette.secondary.main,
    'insurance': theme.palette.error.main,
    'default': theme.palette.grey[500]
  };

  return {
    backgroundColor: colorMap[type] || colorMap.default,
    color: '#fff',
    fontWeight: 'bold',
    fontSize: '0.7rem',
  };
});

/**
 * Displays a list of financial recommendations
 * 
 * @param {Object} props
 * @param {Array} props.recommendations - Array of recommendation objects
 */
const RecommendationList = ({ recommendations = [] }) => {
  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  return (
    <RecommendationContainer>
      <RecommendationTitle variant="h6">
        Recommendations For You
      </RecommendationTitle>
      
      <Divider sx={{ mb: 2 }} />
      
      <List sx={{ padding: 0 }}>
        {recommendations.map((recommendation, index) => (
          <RecommendationCard 
            key={recommendation.id || index} 
            variant="outlined"
            type={recommendation.type}
          >
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  {recommendation.title}
                </Typography>
                <TypeChip 
                  label={recommendation.type || 'general'} 
                  size="small"
                  type={recommendation.type}
                />
              </Box>
              
              <Typography variant="body2" color="text.secondary" paragraph>
                {recommendation.description}
              </Typography>
              
              {recommendation.action && (
                <Button 
                  size="small" 
                  variant="outlined" 
                  onClick={() => console.log('Action clicked for', recommendation.id)}
                >
                  {recommendation.action}
                </Button>
              )}
            </CardContent>
          </RecommendationCard>
        ))}
      </List>
    </RecommendationContainer>
  );
};

export default RecommendationList; 