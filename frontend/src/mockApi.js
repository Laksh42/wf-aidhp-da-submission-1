import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import { config } from './config';

// In-memory data for mocking API responses
let userCredentials = {
  "testuser": "password"
};

let users = [
  {
    id: 1,
    username: "testuser",
    firstName: "Test",
    lastName: "User",
    email: "test@example.com",
    persona: {
      age: 35,
      income: "75000-100000",
      occupation: "Software Engineer",
      goals: ["Retirement", "Home Purchase"],
      risk_tolerance: "Moderate"
    }
  }
];

const mockRecommendations = [
  {
    id: "rec-001",
    name: "High-Yield Savings Account",
    category: "Savings",
    tags: ["Low Risk", "Liquid"],
    priority: "high",
    short_description: "Earn higher interest on your savings with no fees and easy access to your money.",
    description: "Our High-Yield Savings Account offers significantly better rates than traditional savings accounts, with no minimum balance requirements and no monthly fees. Your money stays liquid while earning competitive interest rates.",
    key_features: ["No monthly fees", "FDIC insured", "Mobile banking access", "2.15% APY"],
    benefits: [
      "Earn 5x the national average interest rate",
      "No minimum balance requirements",
      "Easy access to your funds",
      "No risk to principal"
    ],
    why_recommended: "Based on your cash reserves and short-term goals, this account provides a safe place to grow your emergency fund while maintaining liquidity.",
    rating: 4.5,
    link: "https://example.com/high-yield-savings"
  },
  {
    id: "rec-002",
    name: "Wells Fargo Red Card™",
    category: "Credit Card",
    tags: ["Rewards", "No Annual Fee"],
    short_description: "Earn 2% cash back on all purchases with no annual fee and special Wells Fargo benefits.",
    description: "The Wells Fargo Red Card™ offers unlimited 2% cash back on all purchases, with no rotating categories to track. Enjoy no annual fee, special financing offers, and exclusive Wells Fargo customer benefits.",
    key_features: ["2% unlimited cash back", "No annual fee", "Special financing", "Mobile app integration"],
    benefits: [
      "Simple rewards structure with no categories to track",
      "Redeem rewards directly to your Wells Fargo account",
      "Special financing on large purchases",
      "Enhanced fraud protection"
    ],
    why_recommended: "This card complements your spending habits and offers better rewards than your current credit card, with no annual fee to offset the benefits.",
    rating: 4.0,
    link: "https://example.com/red-card"
  },
  {
    id: "rec-003",
    name: "Balanced Growth Portfolio",
    category: "Investment",
    tags: ["Moderate Risk", "Diversified"],
    short_description: "A professionally managed diversified portfolio designed for moderate growth with managed risk.",
    description: "Our Balanced Growth Portfolio offers a diversified mix of stocks, bonds, and alternative investments designed to provide moderate long-term growth while managing downside risk. This portfolio is professionally managed and rebalanced quarterly.",
    key_features: ["Professional management", "Quarterly rebalancing", "Tax optimization", "Performance reporting"],
    benefits: [
      "Diversified exposure to multiple asset classes",
      "Risk-adjusted returns targeting 6-8% annually",
      "Lower volatility than all-equity portfolios",
      "Automatic rebalancing and tax optimization"
    ],
    why_recommended: "Your risk tolerance and investment time horizon align well with this moderately aggressive portfolio, offering growth potential while maintaining some downside protection.",
    rating: 4.7,
    link: "https://example.com/balanced-growth"
  },
  {
    id: "rec-004",
    name: "Mortgage Refinance - 15 Year Fixed",
    category: "Mortgage",
    tags: ["Lower Rate", "Shorter Term"],
    short_description: "Save on interest and pay off your home faster with our competitive 15-year fixed-rate mortgage.",
    description: "Refinancing to our 15-year fixed-rate mortgage can help you save thousands in interest over the life of your loan while building equity faster. With current rates significantly lower than your existing mortgage, this could be an excellent opportunity to reduce your overall costs.",
    key_features: ["4.25% fixed rate", "No points option", "Low closing costs", "Online application"],
    benefits: [
      "Build equity faster than with a 30-year mortgage",
      "Save thousands in total interest payments",
      "Pay off your home in half the time",
      "Lock in historically competitive rates"
    ],
    why_recommended: "Based on your current mortgage rate of 5.75% and your stable income, refinancing could save you approximately $58,000 in interest over the life of your loan.",
    rating: 4.2,
    link: "https://example.com/mortgage-refinance"
  }
];

const userInsights = {
  account: {
    balance: 12500,
    account_type: "Checking"
  },
  demographics: {
    age: 35,
    occupation: "Software Engineer",
    income_bracket: "$75,000 - $100,000"
  },
  credit_score: 760
};

// Mock API endpoints
const mockApi = {
  // Auth endpoints
  login: async (username, password) => {
    console.log('Mock API: Login attempt with', { username, password });
    
    // Validate credentials
    if (userCredentials[username] === password) {
      // Find the user record
      const user = users.find(u => u.username === username);
      
      if (user) {
        return {
          access_token: `mock-token-${username}-${Date.now()}`,
          user_id: user.id,
          username: user.username
        };
      }
    }
    
    // Authentication failed
    const error = new Error('Invalid username or password');
    error.status = 401;
    throw error;
  },
  
  register: async (username, password) => {
    console.log('Mock API: Register attempt with', { username });
    
    // Check if username is already taken
    if (userCredentials[username]) {
      const error = new Error('Username already exists');
      error.status = 400;
      throw error;
    }
    
    // Create new user
    const newUser = {
      id: users.length + 1,
      username,
      firstName: "", 
      lastName: "",
      email: "",
    };
    
    // Add user to our collections
    users.push(newUser);
    userCredentials[username] = password;
    
    console.log('Mock API: Registered new user', newUser);
    
    return {
      id: newUser.id,
      username: newUser.username,
      message: 'User registered successfully'
    };
  },
  
  getUserData: async (userId) => {
    // In reality this would validate a token and return user data
    const user = users.find(u => u.id.toString() === userId.toString() || u.username === userId);
    
    if (!user) {
      const error = new Error('User not found');
      error.status = 404;
      throw error;
    }
    
    return user;
  },
  
  // Recommendations endpoints
  getRecommendations: async () => {
    // Normally this would fetch personalized recommendations for the user
    return {
      recommendations: mockRecommendations,
      user_insights: userInsights
    };
  },
  
  refreshRecommendations: async () => {
    // In a real implementation, this would request new recommendations
    return { success: true, message: 'Recommendations refreshed successfully' };
  },
  
  provideFeedback: async (recommendationId, isRelevant) => {
    console.log('Mock API: Recommendation feedback:', { recommendationId, isRelevant });
    return { success: true };
  },
  
  // Chat endpoints
  sendChatMessage: async (message, sessionId = null) => {
    // This would normally call an AI service to process the message
    const newSessionId = sessionId || `mock-session-${Date.now()}`;
    
    // Mock response based on message content
    let response;
    
    if (message.toLowerCase().includes('hello') || message.toLowerCase().includes('hi')) {
      response = "Hello! I'm your Wells Fargo financial assistant. How can I help you today? You can ask me about banking services, investment options, or financial planning.";
    } else if (message.toLowerCase().includes('invest')) {
      response = "Based on your profile, there are several investment options that might be suitable for you. I'd recommend considering our Balanced Growth Portfolio which aligns with your moderate risk tolerance. Would you like more details about this option?";
    } else if (message.toLowerCase().includes('mortgage') || message.toLowerCase().includes('home')) {
      response = "I see that you might benefit from our mortgage refinancing options. With current rates, you could potentially save significantly over the life of your loan. Would you like me to explain the 15-year fixed option I'm seeing for your profile?";
    } else if (message.toLowerCase().includes('credit') || message.toLowerCase().includes('card')) {
      response = "You have excellent credit at 760, which qualifies you for our best credit card offers. The Wells Fargo Red Card with 2% cash back on all purchases might be perfect for your spending habits. Would you like to know more about the benefits?";
    } else {
      response = "Thanks for your message. I understand you're interested in improving your financial situation. Based on your profile, I can recommend both short-term savings solutions and long-term investment strategies. Would you like to focus on either area first?";
    }
    
    return {
      text: response,
      session_id: newSessionId,
      state: {
        is_complete: false
      }
    };
  }
};

export default mockApi; 