# Dolmetsch-Trainer Pro 2.0

An AI-powered training environment for professional interpreters, ready to be deployed on Vercel.

## Features

- Multiple interpreting modes: Simultaneous, Consecutive, Shadowing, Dialogue, and Sight Translation.
- AI-generated exercises based on user-defined topics and difficulty.
- Support for multiple languages.
- Real-time recording and transcription of user's interpretation.
- Detailed AI-powered feedback on content, expression, and terminology.
- Ability to correct transcripts before getting feedback.
- Secure API key management via environment variables.

## Tech Stack

- **Framework:** React
- **Build Tool:** Vite
- **Language:** TypeScript
- **AI:** Google Gemini API (`@google/genai`)
- **Speech:** Web Speech API (SpeechRecognition and SpeechSynthesis)

## Getting Started

### Prerequisites

- Node.js (v18 or higher recommended)
- A code editor like VS Code
- A Google AI API Key

### Local Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[YOUR_USERNAME]/[YOUR_REPOSITORY_NAME].git
    cd [YOUR_REPOSITORY_NAME]
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```
    
3.  **Create an environment file:**
    Create a file named `.env` in the root of your project and add your Google AI API key:
    ```
    API_KEY="YOUR_GOOGLE_AI_API_KEY"
    ```
    The project is configured via `vite.config.ts` to make the `API_KEY` from your `.env` file securely available in the application code as `process.env.API_KEY`.


4.  **Run the development server:**
    ```bash
    npm run dev
    ```
    The application will be available at `http://localhost:5173` (or another port if 5173 is busy).

## Deployment

This project is optimized for deployment on **Vercel**.

1.  Push your code to a GitHub repository.
2.  Sign up or log in to [Vercel](https://vercel.com).
3.  Click "Add New..." -> "Project".
4.  Import your GitHub repository.
5.  Vercel will automatically detect that this is a Vite project and configure the build settings correctly.
6.  **Configure Environment Variables:** In the project settings on Vercel, go to "Environment Variables" and add a new variable:
    -   **Name:** `API_KEY`
    -   **Value:** Your Google AI API Key
7.  Click **Deploy**.
