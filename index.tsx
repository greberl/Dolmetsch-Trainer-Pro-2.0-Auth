// FIX: The original content of this file was invalid placeholder text.
// It has been replaced with a functional React component that demonstrates Gemini API usage.
import React, { useState, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI } from '@google/genai';

// FIX: Initialize the GoogleGenAI client once at the module level for efficiency, as per best practices.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

function App() {
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt.');
      return;
    }
    setLoading(true);
    setError('');
    setResult('');

    try {
      // FIX: Use the 'gemini-2.5-flash' model for text generation, following the guidelines.
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: prompt,
      });

      // FIX: Extract the text directly from the `response.text` property as per guidelines.
      setResult(response.text);
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
      setError(`Failed to generate content: ${errorMessage}`);
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [prompt]);

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif', maxWidth: '800px', margin: 'auto' }}>
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1>Gemini API React Demo</h1>
        <p>Enter a prompt and click "Generate" to get a response from the Gemini API.</p>
      </header>
      <main>
        <textarea
          value={prompt}
          onChange={(e) => {
            setPrompt(e.target.value);
            if (error) setError('');
          }}
          placeholder="e.g., Why is the sky blue?"
          rows={5}
          style={{ width: '100%', padding: '10px', fontSize: '16px', marginBottom: '10px', boxSizing: 'border-box', borderRadius: '5px', border: '1px solid #ccc' }}
        />
        <button 
          onClick={handleGenerate} 
          disabled={loading || !prompt.trim()} 
          style={{ 
            padding: '10px 20px', 
            fontSize: '16px', 
            cursor: 'pointer',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            opacity: (loading || !prompt.trim()) ? 0.6 : 1,
          }}
        >
          {loading ? 'Generating...' : 'Generate'}
        </button>

        {loading && <p style={{ textAlign: 'center', marginTop: '20px' }}>Loading, please wait...</p>}
        
        {error && 
          <div style={{ color: '#721c24', backgroundColor: '#f8d7da', border: '1px solid #f5c6cb', marginTop: '20px', padding: '15px', borderRadius: '5px' }}>
            <strong>Error:</strong> {error}
          </div>
        }

        {result && (
          <div style={{ marginTop: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '5px', backgroundColor: '#f9f9f9' }}>
            <h2>Response:</h2>
            <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word', fontFamily: 'monospace', fontSize: '14px', lineHeight: '1.5' }}>
              {result}
            </pre>
          </div>
        )}
      </main>
    </div>
  );
}

const rootElement = document.getElementById('root');
if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
} else {
  console.error("Fatal: The root element with ID 'root' was not found in the DOM.");
}
