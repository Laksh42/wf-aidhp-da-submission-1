# Frontend Documentation

## Overview
This is the frontend application for the Wells Fargo Financial Assistant. It's built with React and uses Material-UI for the UI components.

## Key Features
- User authentication (login/registration)
- Dashboard with personalized financial recommendations
- Chat interface with AI-powered financial assistant
- Portfolio management
- Transaction history and analysis

## Setup
1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

4. The application will be available at http://localhost:3000

## Configuration
The main configuration can be found in `src/config.js`. It includes:
- API endpoints
- Authentication settings
- Feature flags
- UI theme configuration (Wells Fargo red and yellow)

## Project Structure
- `src/components/`: Reusable UI components
- `src/pages/`: Page components
- `src/services/`: API client services
- `src/utils/`: Utility functions
- `src/context/`: React context providers for state management

## Integration with Backend
The frontend connects to the FastAPI backend at http://localhost:8000 with endpoints:
- Authentication: `/api/auth/*`
- Chat: `/api/chat/*`
- Recommendations: `/api/recommendations/*` 