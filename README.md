# Multi-Modal Financial Advisor Chatbot

A personalized financial advisor chatbot that leverages multi-modal inputs to provide tailored financial product recommendations.

## Features

- **Chat Interface**: Natural language conversation with context awareness
- **Document Analysis**: Upload and analyze financial documents (receipts, statements, etc.)
- **Personalized Recommendations**: Financial product recommendations based on user profile
- **User Authentication**: Secure login and user management
- **Meta-Prompt Generation**: Creates personalized context for each user
- **Multiple LLM Support**: OpenAI, Mistral AI, and HuggingFace model integration

## Architecture

The system consists of several core components:

1. **Data Ingestion**: Process multi-modal inputs (text, images, documents)
2. **Meta-Prompt Engineering**: Generate user-specific context
3. **Recommendation Engine**: Suggest financial products using a RAG approach
4. **Conversational Interface**: User interaction via natural language
5. **Image Analysis**: Extract information from financial documents
6. **Fallback Mechanisms**: Graceful degradation when services are unavailable

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: MongoDB
- **AI Services**: 
  - OpenAI GPT-3.5/4
  - Mistral AI (Mistral-7B)
  - HuggingFace models
- **Data Processing**: Pandas, NumPy
- **Vector Storage**: In-memory vector store with OpenAI embeddings
- **HTTP Client**: HTTPX for async API calls

## Setup Instructions

### Prerequisites

- Python 3.8+
- MongoDB
- LLM API keys (one of the following):
  - OpenAI API key
  - Mistral API key
  - HuggingFace API token

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=financial_advisor
MONGODB_USER=your_mongodb_user
MONGODB_PASSWORD=your_mongodb_password

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password

# API Keys (choose one or more)
OPENAI_API_KEY=your-openai-api-key
HUGGINGFACE_TOKEN=your-huggingface-token
MISTRAL_API_KEY=your-mistral-api-key

# Model Configuration
DEFAULT_MODEL=gpt-4-turbo-preview
FINANCE_MODEL=pixiu-financial
CHAT_MODEL=mistralai/Mistral-7B-v0.1
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Application Settings
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/financial-advisor-chatbot.git
   cd financial-advisor-chatbot
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application using one of these options:

   a. Using the provided start script (recommended):
   ```
   chmod +x start.sh  # Make the script executable (only needed once)
   ./start.sh
   ```
   This will start both the backend server and frontend application together.

   b. Manually start the backend:
   ```
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

5. Visit `http://localhost:3000` to access the application interface, or `http://localhost:8000/docs` to see the API documentation.

## Using the Application

After starting the application with `./start.sh`, you can access:
- Frontend interface at `http://localhost:3000`
- Backend API at `http://localhost:8000`

The application comes with a pre-configured test user:
- Username: `testuser`
- Password: `password`

## API Endpoints

- **Authentication**
  - `POST /api/auth/register`: Register a new user
  - `POST /api/auth/token`: Get authentication token
  - `GET /api/auth/me`: Get current user information

- **Chat**
  - `POST /api/chat/message`: Send a message to the chatbot
  - `GET /api/chat/conversations`: List user conversations
  - `GET /api/chat/conversations/{id}`: Get a specific conversation
  - `DELETE /api/chat/conversations/{id}`: Delete a conversation

- **Recommendations**
  - `GET /api/recommendations`: Get personalized recommendations
  - `GET /api/recommendations/history`: Get recommendation history
  - `POST /api/recommendations/feedback`: Provide feedback on recommendations

- **Images**
  - `POST /api/images/upload`: Upload and analyze a financial document
  - `GET /api/images/analyses`: List document analyses
  - `GET /api/images/analyses/{id}`: Get specific document analysis
  - `DELETE /api/images/analyses/{id}`: Delete a document analysis

## LLM Provider Configuration

The application supports multiple LLM providers:

### OpenAI
Set `OPENAI_API_KEY` to use GPT models.

### Mistral AI
Set `MISTRAL_API_KEY` to use Mistral models.

### HuggingFace
Set `HUGGINGFACE_TOKEN` to use models hosted on HuggingFace.

The application will automatically select a provider based on available API keys with this priority order:
1. Mistral AI
2. HuggingFace
3. OpenAI

If no API keys are provided, the application will use a basic keyword-based mock response system.

## Database Configuration

The application uses MongoDB for storing user data, financial information, chat history, and more. 

If MongoDB is not available or credentials are incorrect, the application will:
1. Log warnings about the unavailable database
2. Use fallback mock data for financial profiles
3. Continue functioning with limited personalization features

## Development

### Project Structure

```
financial-advisor-chatbot/
├── app/
│   ├── api/
│   │   ├── auth.py
│   │   ├── chat.py
│   │   ├── recommendations.py
│   │   └── images.py
│   ├── database/
│   │   ├── models.py
│   │   ├── mongodb.py
│   │   └── initialize_db.py
│   ├── repository/
│   │   └── financial_repository.py
│   ├── models/
│   │   ├── user.py
│   │   ├── financial.py
│   │   └── chat.py
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── auth_service.py
│   │   └── recommendation_service.py
│   ├── config.py
│   └── main.py
├── data/
│   ├── products.csv
│   ├── demographic_data.csv
│   ├── account_data.csv
│   └── transaction_data.csv
├── .env
├── README.md
└── requirements.txt
```

## Future Enhancements

- **Personalized Financial Planning**: Generate long-term financial plans
- **Investment Portfolio Analysis**: Analyze and provide feedback on investment portfolios
- **Expense Tracking**: Track and categorize expenses from uploaded receipts
- **Budget Recommendations**: Suggest budgeting strategies based on spending patterns
- **Mobile App Integration**: API integration with mobile applications
- **Fine-tuned Financial Models**: Domain-specific fine-tuning of LLMs for finance

## License

MIT

## Contact

For any questions or feedback, please contact the project maintainers.