# Financial Advisor Dashboard Frontend

This is a modern React-based frontend for the Financial Advisor application. It focuses on displaying personalized financial product recommendations in a clean, user-friendly interface.

## Features

- **Modern UI**: Built with React and Material-UI for a responsive, accessible interface
- **Recommendation-Centric**: Prioritizes financial product recommendations over chat interaction
- **Personalized Dashboard**: Displays tailored financial recommendations and user insights
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Getting Started

### Prerequisites

- Node.js (v14 or newer)
- npm or yarn

### Installation

1. Clone the repository
2. Navigate to the frontend directory:
   ```
   cd frontend
   ```
3. Install dependencies:
   ```
   npm install
   ```
   or
   ```
   yarn install
   ```

### Running the Development Server

Start the development server:
```
npm start
```
or
```
yarn start
```

The application will be available at [http://localhost:3000](http://localhost:3000).

### Mock Data Access

In development mode, the application uses a mock API to simulate backend responses.

Use these credentials to log in:
- **User ID**: testuser
- **Password**: password

## Project Structure

- `/src` - Source code
  - `/components` - Reusable UI components
  - `/contexts` - React contexts for state management
  - `/pages` - Page components
  - `/routes` - Routing configuration
  - `App.js` - Main application component
  - `config.js` - Application configuration
  - `mockApi.js` - Mock API for development

## Building for Production

To create a production build:
```
npm run build
```
or
```
yarn build
```

The build files will be located in the `build` directory.

## Integration with Backend

The frontend is designed to work with the existing Financial Advisor backend API. Configuration for API endpoints can be found in `src/config.js`. 