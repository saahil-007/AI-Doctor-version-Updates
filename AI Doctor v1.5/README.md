# AI Doctor - Medical Chatbot

A lightweight Flask web application that integrates Google's Generative AI model to provide medical assistance and general conversation.

## Features

- **AI-Powered Medical Assistance**: Get medical advice and information using Google's Gemini AI
- **Multilingual Support**: Communicate in English, Hindi, and Marathi
- **Real-time Translation**: Translate responses between supported languages
- **Text-to-Speech**: Listen to responses with natural-sounding voices
- **Speech Recognition**: Speak your questions instead of typing
- **Responsive Design**: Works on desktop and mobile devices

## Technologies Used

- **Backend**: Python, Flask
- **AI**: Google Generative AI (Gemini)
- **Frontend**: React, Vite, TypeScript, shadcn-ui, Tailwind CSS
- **APIs**: MyMemory Translation API
- **Voice**: Web Speech API

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Node.js and npm
- Google Generative AI API key

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saahil-007/BluePlanet-AI-Doctor.git
   cd BluePlanet-AI-Doctor
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install backend dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   ```

6. **Build the frontend**:
   ```bash
   npm run build
   ```

7. **Run the application**:
   ```bash
   cd ..
   python app.py
   ```

8. **Access the application**:
   Open your browser and go to `http://localhost:5000`

## Usage

1. Type your medical questions or general conversation in the input box
2. Select your preferred language from the language selector
3. Use the microphone button for voice input
4. Use the speaker button to listen to responses
5. Translate responses using the translation controls:
   - Use the globe icon (🌐) to translate individual messages
   - Use "Apply Translation" to translate all previous messages

## Project Structure

```
AI-Doctor/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed)
├── .env.example           # Example environment variables file
├── .gitignore             # Git ignore file
├── system_prompt.txt      # AI system prompt
├── runtime.txt            # Python runtime version for Vercel
├── vercel.json            # Vercel deployment configuration
├── frontend/              # React frontend application
│   ├── dist/              # Built frontend files
│   ├── src/               # Source code
│   ├── package.json       # Frontend dependencies
│   └── vite.config.ts     # Vite configuration
└── README.md              # This file
```

## Deployment

### Vercel Deployment

This application is configured for deployment on Vercel:

1. Fork this repository to your GitHub account
2. Sign up/log in to [Vercel](https://vercel.com)
3. Create a new project and import your forked repository
4. Add your `GOOGLE_API_KEY` as an environment variable in Vercel project settings:
   - Key: `GOOGLE_API_KEY`
   - Value: Your actual Google API key
5. Make sure you have a `runtime.txt` file specifying the Python version (e.g., `python-3.9.18`)
6. Deploy the project

The `vercel.json` and `runtime.txt` configurations are already included in the repository.

### Environment Variables

For deployment, you need to set the following environment variable:
- `GOOGLE_API_KEY`: Your Google Generative AI API key

## Troubleshooting Vercel Deployment

If you're seeing the "Gemini model is not available for requests - check API key configuration" error:

1. **Verify your API key**: Make sure you have a valid Google Generative AI API key from [Google AI Studio](https://aistudio.google.com/)

2. **Check environment variable setup**: In your Vercel project settings, ensure the `GOOGLE_API_KEY` environment variable is correctly set:
   - Go to your Vercel project dashboard
   - Click on "Settings" → "Environment Variables"
   - Add a new variable with:
     - Key: `GOOGLE_API_KEY`
     - Value: Your actual Google API key
     - Check "Production", "Preview", and "Development" environments

3. **Redeploy your application**: After adding the environment variable, redeploy your application:
   - Go to the "Deployments" tab
   - Click on the three dots next to your latest deployment
   - Select "Redeploy"

4. **Check runtime version**: Ensure you have a `runtime.txt` file in your project root specifying the Python version:
   ```
   python-3.9.18
   ```

5. **Verify dependencies**: Make sure your `requirements.txt` includes:
   ```
   flask==3.1.2
   google-generativeai==0.8.5
   python-dotenv==1.0.0
   ```

## API Endpoints

- `POST /chat`: Main chat endpoint for sending messages to the AI
- `POST /translate`: Endpoint for translating text between languages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application provides general medical information only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.