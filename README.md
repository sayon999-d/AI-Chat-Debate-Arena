# AI Debate Arena

A secure, multi-provider AI debate platform that orchestrates parallel responses from various Large Language Models (LLMs) to provide diverse perspectives on user queries. The system integrates real-time web research to ground AI responses in current data.

## Features

- **Multi-Provider Orchestration**: Parallel execution of multiple AI providers (Groq, OpenRouter, Chutes, Bytez).
- **Web Research Integration**: Automated web search using Tavily API to provide context to LLMs.
- **Answer Synthesis**: Intelligent synthesis of multiple AI responses into a cohesive final answer.
- **Authentication System**: Secure user management with Email/Password and Google OAuth support.
- **Real-time Analytics**: Dashboard for monitoring provider latency, token usage, and system health.
- **Security First**: Implemented with strict CSP, rate limiting, input sanitization, and secure headers.
- **Optimized Performance**: Token usage optimization and concurrent asynchronous processing.

## Technology Stack

- **Backend**: Python 3.11, FastAPI
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (Embedded)
- **Database**: JSON-based flat file storage (Development), In-memory session management
- **AI Integration**: Custom provider adapters, Tavily Search API
- **Deployment**: Render.com configuration ready (render.yaml)

## Prerequisites

- Python 3.11 or higher
- pip package manager
- API Keys for utilized providers (Groq, OpenRouter, etc.)

## Installation

1. Clone the repository:
   git clone https://github.com/sayon999-d/AI-Chat-Debate-Arena.git
   cd AI-Chat-Debate-Arena

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r backend/requirements.txt

## Configuration

1. Copy the example environment file:
   cp backend/.env.example backend/.env

2. Edit backend/.env and populate your API keys:
   - GROQ_API_KEY
   - OPENROUTER_API_KEY
   - CHUTES_API_KEY
   - BYTEZ_API_KEY
   - TAVILY_API_KEY
   - GOOGLE_CLIENT_ID (Optional for OAuth)
   - GOOGLE_CLIENT_SECRET (Optional for OAuth)

## usage

Start the application server:
cd backend
python main.py

Access the application at http://localhost:8000.

## API Documentation

When the server is running, you can access the interactive API documentation (Swagger UI) at http://localhost:8000/docs if enabled in configuration.

## Deployment

This project includes a render.yaml configuration file for automated deployment on Render.com.

1. Connect your repository to Render.
2. Select "Blueprint" deployment.
3. Configure environment variables in the Render dashboard.

## License

This project is open source and available under the MIT License.