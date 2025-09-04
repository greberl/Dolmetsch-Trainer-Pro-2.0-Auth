



import React, { useState, useRef, useCallback, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";

// --- WEB SPEECH API TYPES ---
// Fix for TS2304, TS2339: Provide types for the browser's SpeechRecognition API.
interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}
interface SpeechRecognitionResult {
  isFinal: boolean;
  length: number;
  [index: number]: SpeechRecognitionAlternative;
}
interface SpeechRecognitionResultList {
  length: number;
  [index: number]: SpeechRecognitionResult;
}
interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}
interface SpeechRecognitionErrorEvent extends Event {
    error: 'no-speech' | 'aborted' | 'audio-capture' | 'network' | 'not-allowed' | 'service-not-allowed' | 'bad-grammar' | 'language-not-supported' | 'interrupted' | 'audio-busy' | 'synthesis-failed';
    message: string;
}
interface SpeechRecognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: (event: SpeechRecognitionEvent) => void;
  // Fix: Add 'onstart' property to the SpeechRecognition interface.
  onstart: () => void;
  onerror: (event: SpeechRecognitionErrorEvent) => void;
  onend: () => void;
  start(): void;
  stop(): void;
}
declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition;
    webkitSpeechRecognition: new () => SpeechRecognition;
  }
}
// --- END WEB SPEECH API TYPES ---

// --- TYPES ---
type InterpretingMode = "Vortragsdolmetschen" | "Simultandolmetschen" | "Shadowing" | "Gesprächsdolmetschen" | "Stegreifübersetzen";
type Language = "Deutsch" | "Englisch" | "Russisch" | "Spanisch" | "Französisch";
type SourceTextType = "ai" | "upload";
type QALength = "1-3 Sätze" | "2-4 Sätze" | "3-5 Sätze" | "4-6 Sätze";
type SpeechLength = "Kurz" | "Mittel" | "Prüfung";
type VoiceQuality = "Standard" | "Premium";
type DialogueState = 'idle' | 'synthesizing' | 'playing' | 'waiting_for_record' | 'recording' | 'finished';
type PracticeAreaTab = 'original' | 'transcript' | 'feedback' | 'analysis';


interface DialogueSegment {
    type: 'Frage' | 'Antwort';
    text: string;
    lang: Language;
}

interface Settings {
  mode: InterpretingMode;
  sourceLang: Language;
  targetLang: Language;
  sourceType: SourceTextType;
  topic: string;
  qaLength: QALength;
  speechLength: SpeechLength;
  voiceQuality: VoiceQuality;
}

interface ErrorAnalysisItem {
    original: string;
    interpretation: string;
    suggestion: string;
}

interface Feedback {
    summary: string;
    ratings: {
        content: number;
        expression: number;
        terminology: number;
    };
    errorAnalysis: ErrorAnalysisItem[];
}

// --- CUSTOM ERROR TYPE ---
class TtsAuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TtsAuthError';
  }
}

// --- CONSTANTS ---
const MODES: InterpretingMode[] = ["Vortragsdolmetschen", "Simultandolmetschen", "Shadowing", "Gesprächsdolmetschen", "Stegreifübersetzen"];
const LANGUAGES: Language[] = ["Deutsch", "Englisch", "Russisch", "Spanisch", "Französisch"];
const QA_LENGTHS: QALength[] = ["1-3 Sätze", "2-4 Sätze", "3-5 Sätze", "4-6 Sätze"];
const SPEECH_LENGTHS: SpeechLength[] = ["Kurz", "Mittel", "Prüfung"];

const LANGUAGE_CODES: Record<Language, string> = {
  "Deutsch": "de-DE",
  "Englisch": "en-US",
  "Russisch": "ru-RU",
  "Spanisch": "es-ES",
  "Französisch": "fr-FR",
};

const WAVENET_VOICES: Record<Language, string> = {
    "Deutsch": "de-DE-Wavenet-F", // Female
    "Englisch": "en-US-Wavenet-J", // Male
    "Russisch": "ru-RU-Wavenet-E", // Female
    "Spanisch": "es-ES-Wavenet-B", // Male
    "Französisch": "fr-FR-Wavenet-E", // Female
};


const SPEECH_LENGTH_CONFIG: Record<SpeechLength, { min: number, max: number }> = {
    "Kurz": { min: 1000, max: 1500 },
    "Mittel": { min: 2000, max: 2500 },
    "Prüfung": { min: 3300, max: 3700 }
};

const TEXT_LENGTH_CONFIG: Record<InterpretingMode, { min: number, max: number }> = {
    "Vortragsdolmetschen": { min: 0, max: 0 }, // Handled by SPEECH_LENGTH_CONFIG
    "Simultandolmetschen": { min: 0, max: 0 }, // Handled by SPEECH_LENGTH_CONFIG
    "Shadowing": { min: 0, max: 0 }, // Handled by SPEECH_LENGTH_CONFIG
    "Stegreifübersetzen": { min: 1280, max: 1450 },
    "Gesprächsdolmetschen": { min: 0, max: 0 } // Not used for dialogue
};

const model = "gemini-2.5-flash";

// --- HELPER COMPONENTS ---
const ApiKeyErrorDisplay = () => (
  <div className="api-key-modal-overlay">
    <div className="api-key-modal">
      <h2>Konfiguration erforderlich</h2>
      <p>
        Der Google AI API-Schlüssel wurde nicht gefunden. Bitte stellen Sie sicher, dass die 
        <code> API_KEY</code> Umgebungsvariable in Ihrer Deployment-Umgebung (z.B. Vercel) korrekt gesetzt ist.
      </p>
    </div>
  </div>
);

// --- Google Cloud Text-to-Speech API Helper ---
const synthesizeSpeechGoogleCloud = async (text: string, language: Language, apiKey: string): Promise<string> => {
    const langCode = LANGUAGE_CODES[language];
    const voiceName = WAVENET_VOICES[language];

    const response = await fetch(`https://texttospeech.googleapis.com/v1/text:synthesize`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': apiKey,
        },
        body: JSON.stringify({
            input: { text },
            voice: {
                languageCode: langCode,
                name: voiceName
            },
            audioConfig: {
                audioEncoding: 'MP3'
            }
        })
    });

    if (!response.ok) {
        const errorData = await response.json();
        console.error("Google TTS API Error:", JSON.stringify(errorData, null, 2));
        const message = `Fehler bei der Sprachsynthese: ${errorData.error?.message || 'Unbekannter Fehler'}`;
        // Check for common API key-related errors to allow for graceful fallback.
        if (response.status === 403 || response.status === 400 || errorData.error?.message?.toLowerCase().includes('api key not valid')) {
             throw new TtsAuthError(message);
        }
        throw new Error(message);
    }

    const data = await response.json();
    if (!data.audioContent) {
        throw new Error("Keine Audiodaten von der TTS-API erhalten.");
    }

    return data.audioContent; // This is a base64 encoded string
};


// --- MAIN APP COMPONENT ---
const App = () => {
  const [ai] = useState<GoogleGenAI | null>(() => {
    if (!process.env.API_KEY) {
      console.error("API_KEY environment variable not set.");
      return null;
    }
    try {
      return new GoogleGenAI({ apiKey: process.env.API_KEY });
    } catch (e) {
      console.error("Failed to initialize GoogleGenAI", e);
      return null;
    }
  });

  const [settings, setSettings] = useState<Settings>({
    mode: 'Simultandolmetschen',
    sourceLang: 'Deutsch',
    targetLang: 'Russisch',
    sourceType: 'ai',
    topic: '',
    qaLength: '2-4 Sätze',
    speechLength: 'Mittel',
    voiceQuality: 'Standard',
  });
  const [isPremiumVoiceAvailable, setIsPremiumVoiceAvailable] = useState(true);

  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [error, setError] = useState<string | null>(null);
  
  const [originalText, setOriginalText] = useState('');
  const [userTranscript, setUserTranscript] = useState('');
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [exerciseStarted, setExerciseStarted] = useState(false);
  
  if (!ai) {
    return <ApiKeyErrorDisplay />;
  }

  const handlePremiumVoiceAuthError = () => {
    setIsPremiumVoiceAvailable(false);
  };

  const generateText = async (currentSettings: Settings) => {
    setIsLoading(true);
    setLoadingMessage('Generiere Übungstext...');
    setError(null);
    setOriginalText('');
    setFeedback(null);
    setUserTranscript('');

    try {
        if (currentSettings.mode === 'Gesprächsdolmetschen') {
            const prompt = `Erstelle ein Interview-Skript für eine Dolmetschübung zum Thema "${currentSettings.topic}". Das Interview soll zwischen zwei Personen stattfinden. Gib mir 6 Fragen und 6 Antworten. Die Sprache ist ${currentSettings.sourceLang}. Die Fragen und Antworten sollen jeweils eine Länge von "${currentSettings.qaLength}" haben. Formattiere die Ausgabe als Text, in dem jede Frage mit "Frage X:" und jede Antwort mit "Antwort X:" beginnt. Wechsle die Sprache zwischen den Runden ab, beginnend mit ${currentSettings.sourceLang} für die erste Frage, dann ${currentSettings.targetLang} für die erste Antwort, dann ${currentSettings.sourceLang} für die zweite Frage und so weiter.`;
            const response = await ai.models.generateContent({ model, contents: prompt });
            setOriginalText(response.text || '');
        } else {
            const isSpeechMode = ["Vortragsdolmetschen", "Simultandolmetschen", "Shadowing"].includes(currentSettings.mode);
            const { min, max } = isSpeechMode
                ? SPEECH_LENGTH_CONFIG[currentSettings.speechLength]
                : TEXT_LENGTH_CONFIG[currentSettings.mode];

            const promptText = currentSettings.mode === 'Stegreifübersetzen'
                ? `Schreibe einen sachlichen Text auf ${currentSettings.sourceLang} zum Thema "${currentSettings.topic}". Der Text soll für eine Stegreifübersetzungsübung geeignet sein und keine direkte Rede oder Anrede (wie "Sehr geehrte Damen und Herren") enthalten.`
                : `Schreibe einen Vortragstext auf ${currentSettings.sourceLang} zum Thema "${currentSettings.topic}". Der Text soll für eine Dolmetschübung geeignet sein.`;

            const initialPrompt = `${promptText} Die Ziellänge beträgt zwischen ${min} und ${max} Zeichen inklusive Leerzeichen. Gib nur den reinen Text ohne Titel oder Formatierung zurück.`;
            
            let currentText = '';
            let attempts = 0;
            
            setLoadingMessage('Generiere Text (Versuch 1)...');
            let response = await ai.models.generateContent({ model, contents: initialPrompt });
            currentText = response.text || ''; 

            while ((currentText.length < min || currentText.length > max) && attempts < 4) {
                attempts++;
                setLoadingMessage(`Passe Textlänge an (Versuch ${attempts + 1})...`);
                
                const lengthDifference = currentText.length < min ? min - currentText.length : currentText.length - max;
                const isLargeDifference = lengthDifference > 500;

                const adjustmentPrompt = currentText.length < min
                    ? `Der folgende Text ist mit ${currentText.length} Zeichen zu kurz. Bitte erweitere ihn, um eine Länge zwischen ${min} und ${max} Zeichen zu erreichen. ${isLargeDifference ? "Füge einen oder mehrere passende Absätze hinzu." : "Füge passende Sätze hinzu."} Gib nur den vollständigen, erweiterten Text zurück.\n\n${currentText}`
                    : `Der folgende Text ist mit ${currentText.length} Zeichen zu lang. Bitte kürze ihn, um eine Länge zwischen ${min} und ${max} Zeichen zu erreichen. ${isLargeDifference ? "Entferne einen oder mehrere Absätze." : "Entferne Sätze oder den letzten Absatz."} Gib nur den vollständigen, gekürzten Text zurück.\n\n${currentText}`;


                response = await ai.models.generateContent({ model, contents: adjustmentPrompt });
                currentText = response.text || '';
            }
            setOriginalText(currentText);
        }
        setExerciseStarted(true);
    } catch (err) {
        console.error(err);
        setError("Fehler bei der Generierung des Textes. Bitte versuchen Sie es erneut.");
    } finally {
        setIsLoading(false);
    }
  };
  
  const handleStart = (newSettings: Settings, fileContent?: string) => {
      setSettings(newSettings);
      setExerciseStarted(false);
      if (newSettings.sourceType === 'upload' && fileContent) {
          setOriginalText(fileContent);
          setExerciseStarted(true);
          setError(null);
          setFeedback(null);
          setUserTranscript('');
      } else if (newSettings.sourceType === 'ai') {
          generateText(newSettings);
      }
  };

  const handleRecordingFinished = async (rawTranscript: string) => {
    if (!rawTranscript.trim()) {
        setUserTranscript('');
        return;
    }
    setIsLoading(true);
    setLoadingMessage('Transkript wird verarbeitet...');
    setError(null);
    setFeedback(null);
    try {
        const prompt = `Füge dem folgenden Text eine korrekte Zeichensetzung und Groß-/Kleischreibung hinzu, um ihn lesbar zu machen. Ändere keine Wörter. Der Text ist ein Transkript einer gesprochenen Aufnahme.\n\nRoh-Transkript: "${rawTranscript}"\n\nGib nur den formatierten Text zurück.`;
        const response = await ai.models.generateContent({ model, contents: prompt });
        const punctuatedTranscript = response.text || rawTranscript;
        setUserTranscript(punctuatedTranscript);
    } catch (err) {
        console.error("Error processing transcript:", err);
        setError("Fehler bei der Verarbeitung des Transkripts. Zeige Roh-Version.");
        setUserTranscript(rawTranscript); // Fallback to raw transcript
    } finally {
        setIsLoading(false);
    }
  };

  const getFeedback = async (setActiveTab: (tab: PracticeAreaTab) => void) => {
      if (!userTranscript) {
          setError("Kein Transkript vorhanden, um Feedback zu erhalten.");
          return;
      }
      setIsLoading(true);
      setLoadingMessage('Analysiere und generiere Feedback...');
      setError(null);
      setFeedback(null);

      try {
        const prompt = `Du bist ein erfahrener Coach für Dolmetscher. Deine Aufgabe ist es, eine mündliche Verdolmetschung zu bewerten. Das Transkript des Nutzers ist eine automatische Spracherkennung und wurde bereits mit Zeichensetzung versehen.

**DEINE WICHTIGSTEN ANWEISUNGEN:**
1.  **Fokus auf Mündlichkeit**: Bewerte die Leistung als gesprochene Sprache, nicht als geschriebenen Text. Ignoriere ALLE Rechtschreib- und Zeichensetzungsfehler im Transkript.
2.  **Hohe Fehlertoleranz bei Akzenten**: Der Nutzer hat möglicherweise einen Akzent. Das Transkript kann Wörter falsch wiedergeben. Sei hier sehr tolerant. Die entscheidende Frage ist: **Hätte ein Muttersprachler verstanden, was gemeint war?**
3.  **Klangähnliche Wörter**: Wenn ein Wort im Transkript falsch ist, aber fast genauso klingt wie das korrekte Wort (Homophon/klangähnlich), gehe davon aus, dass der Nutzer das richtige Wort gesagt hat und markiere es NICHT als Fehler.
4.  **Selbstkorrekturen**: Dolmetscher korrigieren sich selbst. Wenn der Nutzer sich selbst korrigiert (z.B. sagt "... äh, ich meine ..."), **bewerte immer die letzte, korrigierte Version der Aussage**.
5.  **Transkriptionsartefakte**: Die Spracherkennung fügt manchmal fälschlicherweise kleine Wörter (Präpositionen, Artikel) ein. Wenn ein Wort im Transkript kontextuell unpassend erscheint, gehe davon aus, dass es ein Transkriptionsfehler ist und **ignoriere es**.
6.  **Verständlichkeit vor Perfektion**: Markiere Aussprachefehler nur dann, wenn sie die Verständlichkeit erheblich beeinträchtigen. Ein allgemeiner Hinweis auf deutlichere Aussprache in der Zusammenfassung ist in Ordnung, wenn es gehäuft vorkommt, aber vermeide es, jeden kleinen Aussprachefehler aufzulisten.

**AUFGABE:**
Bewerte die folgende Verdolmetschung basierend auf den obigen Anweisungen:
Originaltext (${settings.sourceLang}): "${originalText}"
Verdolmetschung des Nutzers (Transkript) (${settings.targetLang}): "${userTranscript}"

Gib dein Feedback als JSON-Objekt.
1.  **summary**: Gib eine kurze, konstruktive Zusammenfassung. Konzentriere dich auf mündliche Aspekte wie Sprechfluss, Füllwörter, Pausen und allgemeine Klarheit, unter Berücksichtigung der Akzenttoleranz.
2.  **ratings**: Bewerte die Verdolmetschung von 1 (schlecht) bis 10 (exzellent) in den folgenden Kategorien:
    -   **content**: Inhaltliche Korrektheit und Vollständigkeit.
    -   **expression**: Mündlicher Ausdruck, Stil und Flüssigkeit.
    -   **terminology**: Korrekte Fachterminologie.
3.  **errorAnalysis**: Erstelle eine Liste von bis zu 5 *signifikanten* inhaltlichen oder terminologischen Fehlern, die die Verständlichkeit beeinträchtigen.
    -   **original**: Der entsprechende Teil des Originaltextes.
    -   **interpretation**: Das Transkript des Nutzers. **Füge hier zur besseren Lesbarkeit Satzzeichen ein**, aber bewerte den Nutzer nicht danach.
    -   **suggestion**: Ein Verbesserungsvorschlag, der sich auf Inhalt, Ausdruck oder Terminologie bezieht, nicht auf die Schriftform oder kleine Ausspracheabweichungen.
`;
        
        const response = await ai.models.generateContent({
            model,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        summary: { type: Type.STRING },
                        ratings: {
                            type: Type.OBJECT,
                            properties: {
                                content: { type: Type.INTEGER },
                                expression: { type: Type.INTEGER },
                                terminology: { type: Type.INTEGER },
                            },
                        },
                        errorAnalysis: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.OBJECT,
                                properties: {
                                    original: { type: Type.STRING },
                                    interpretation: { type: Type.STRING },
                                    suggestion: { type: Type.STRING },
                                },
                            },
                        },
                    },
                },
            },
        });
        
        const jsonStr = (response.text || '').trim();
        if (!jsonStr) {
            throw new Error("Leere Antwort von der Feedback-API erhalten.");
        }
        const parsedFeedback = JSON.parse(jsonStr);
        setFeedback(parsedFeedback);
        setActiveTab('feedback');

      } catch (err) {
          console.error(err);
          setError("Fehler bei der Generierung des Feedbacks. Bitte versuchen Sie es erneut.");
      } finally {
          setIsLoading(false);
      }
  };

  return (
    <>
      <header className="app-header">
        <h1>Dolmetsch-Trainer Pro 2.0</h1>
        <p>Ihre KI-gestützte Übungsumgebung</p>
      </header>
      <main className="main-container">
        <SettingsPanel 
            settings={settings} 
            onStart={handleStart} 
            disabled={isLoading}
            isPremiumVoiceAvailable={isPremiumVoiceAvailable}
        />
        <PracticeArea 
            key={originalText} // Remount component when text changes
            isLoading={isLoading} 
            loadingMessage={loadingMessage}
            originalText={originalText}
            setOriginalText={setOriginalText}
            onRecordingFinished={handleRecordingFinished}
            getFeedback={getFeedback}
            userTranscript={userTranscript}
            setUserTranscript={setUserTranscript}
            feedback={feedback}
            error={error}
            settings={settings}
            exerciseStarted={exerciseStarted}
            onPremiumVoiceAuthError={handlePremiumVoiceAuthError}
            isPremiumVoiceAvailable={isPremiumVoiceAvailable}
         />
      </main>
    </>
  );
};

const SettingsPanel = ({ settings, onStart, disabled, isPremiumVoiceAvailable }: { 
    settings: Settings, 
    onStart: (settings: Settings, fileContent?: string) => void, 
    disabled: boolean,
    isPremiumVoiceAvailable: boolean 
}) => {
    const [currentSettings, setCurrentSettings] = useState(settings);
    const [uploadedFile, setUploadedFile] = useState<{ name: string; content: string } | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const speechModes: InterpretingMode[] = ["Vortragsdolmetschen", "Simultandolmetschen", "Shadowing"];

    const handleChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setCurrentSettings(prev => ({ ...prev, [name]: value }));
        if (name === 'sourceType' && value === 'ai') {
            setUploadedFile(null); // Clear uploaded file when switching back to AI
        }
    };
    
    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file && file.type === 'text/plain') {
            const reader = new FileReader();
            reader.onload = (event) => {
                const content = event.target?.result as string;
                setUploadedFile({ name: file.name, content: content });
            };
            reader.readAsText(file);
        } else {
            alert("Bitte laden Sie eine .txt-Datei hoch.");
            setUploadedFile(null);
        }
        if (e.target) e.target.value = ''; // Allow re-uploading the same file
    };

    const handleSubmit = () => {
        if (currentSettings.sourceType === 'ai') {
            if (!currentSettings.topic.trim()) {
                alert("Bitte geben Sie ein Thema an.");
                return;
            }
            onStart(currentSettings);
        } else { // 'upload'
            if (!uploadedFile) {
                alert("Bitte laden Sie zuerst eine Textdatei hoch, bevor Sie starten.");
                return;
            }
            onStart(currentSettings, uploadedFile.content);
        }
    };
    
    useEffect(() => {
        // When mode changes to shadowing, ensure target lang is same as source
        if(currentSettings.mode === 'Shadowing' && currentSettings.sourceLang !== currentSettings.targetLang) {
            setCurrentSettings(prev => ({...prev, targetLang: prev.sourceLang}));
        }
    }, [currentSettings.mode, currentSettings.sourceLang, currentSettings.targetLang]);

    useEffect(() => {
        if (!isPremiumVoiceAvailable && currentSettings.voiceQuality === 'Premium') {
            setCurrentSettings(prev => ({ ...prev, voiceQuality: 'Standard' }));
        }
    }, [isPremiumVoiceAvailable, currentSettings.voiceQuality]);


    return (
        <aside className="panel settings-panel">
            <div>
                <h2>Einstellungen</h2>
                <div className="form-group">
                    <label htmlFor="mode">Dolmetschmodus</label>
                    <select id="mode" name="mode" className="form-control" value={currentSettings.mode} onChange={handleChange} disabled={disabled}>
                        {MODES.map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                </div>
                {speechModes.includes(currentSettings.mode) && (
                    <div className="form-group">
                        <label htmlFor="voiceQuality">Stimmenqualität</label>
                        <select id="voiceQuality" name="voiceQuality" className="form-control" value={currentSettings.voiceQuality} onChange={handleChange} disabled={disabled}>
                            <option value="Standard">Standard (Browser)</option>
                            <option value="Premium" disabled={!isPremiumVoiceAvailable}>
                                Premium ({isPremiumVoiceAvailable ? 'Google WaveNet' : 'Nicht verfügbar'})
                            </option>
                        </select>
                        {!isPremiumVoiceAvailable && (
                            <p className="form-text-hint">
                                Premium-Stimme fehlgeschlagen. Prüfliste: (1) 'Cloud Text-to-Speech API' ist aktiviert. (2) Rechnungskonto ist verknüpft. (3) API-Schlüssel wurde im selben Projekt erstellt. (4) Der API-Schlüssel hat <strong>keine API-Einschränkungen</strong>, die den Dienst blockieren.
                            </p>
                        )}
                    </div>
                )}
                <div className="form-group">
                    <label htmlFor="sourceLang">Ausgangssprache</label>
                    <select id="sourceLang" name="sourceLang" className="form-control" value={currentSettings.sourceLang} onChange={handleChange} disabled={disabled}>
                        {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
                    </select>
                </div>
                <div className="form-group">
                    <label htmlFor="targetLang">Zielsprache</label>
                    <select id="targetLang" name="targetLang" className="form-control" value={currentSettings.targetLang} onChange={handleChange} disabled={disabled || currentSettings.mode === 'Shadowing'}>
                        {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
                    </select>
                </div>
                <div className="form-group">
                    <label htmlFor="sourceType">Quelltext-Art</label>
                    <select id="sourceType" name="sourceType" className="form-control" value={currentSettings.sourceType} onChange={handleChange} disabled={disabled}>
                        <option value="ai">
                            {currentSettings.mode === 'Gesprächsdolmetschen'
                                ? 'KI-generiertes Interview'
                                : currentSettings.mode === 'Stegreifübersetzen'
                                ? 'KI-generierter Text'
                                : 'KI-generierte Rede'}
                        </option>
                        <option value="upload">Textdatei-Upload (.txt)</option>
                    </select>
                     <input type="file" ref={fileInputRef} onChange={handleFileChange} style={{ display: 'none' }} accept=".txt" />
                </div>
                {currentSettings.sourceType === 'upload' && (
                    <div className="form-group upload-group">
                        <button className="btn btn-secondary" onClick={handleUploadClick} disabled={disabled}>
                            Datei auswählen
                        </button>
                        <span className="file-name" title={uploadedFile?.name}>
                            {uploadedFile ? uploadedFile.name : 'Keine Datei ausgewählt'}
                        </span>
                    </div>
                )}
                {currentSettings.sourceType === 'ai' && (
                    <div className="form-group">
                        <label htmlFor="topic">Thema und Schwierigkeitsgrad</label>
                        <input type="text" id="topic" name="topic" className="form-control" value={currentSettings.topic} onChange={handleChange} disabled={disabled} placeholder="z. B. Umweltschutz in kurzen Sätzen mit einfachem Wortschatz" />
                    </div>
                )}
                {speechModes.includes(currentSettings.mode) && currentSettings.sourceType === 'ai' && (
                    <div className="form-group">
                        <label htmlFor="speechLength">Redelänge</label>
                        <select id="speechLength" name="speechLength" className="form-control" value={currentSettings.speechLength} onChange={handleChange} disabled={disabled}>
                            {SPEECH_LENGTHS.map(l => <option key={l} value={l}>{l}</option>)}
                        </select>
                    </div>
                )}
                {currentSettings.mode === 'Gesprächsdolmetschen' && (
                    <div className="form-group">
                        <label htmlFor="qaLength">Frage- und Antwortlänge</label>
                        <select id="qaLength" name="qaLength" className="form-control" value={currentSettings.qaLength} onChange={handleChange} disabled={disabled}>
                            {QA_LENGTHS.map(l => <option key={l} value={l}>{l}</option>)}
                        </select>
                    </div>
                )}
            </div>
            <div className="settings-footer">
                <button className="btn btn-primary" onClick={handleSubmit} disabled={disabled}>Loslegen</button>
            </div>
        </aside>
    );
};

const splitTextIntoChunks = (text: string, maxLength = 250): string[] => {
    if (!text) return [];

    const chunks: string[] = [];
    let i = 0;
    while (i < text.length) {
        let chunkEnd = i + maxLength;
        if (chunkEnd >= text.length) {
            chunks.push(text.substring(i));
            break;
        }
        
        let lastPunctuation = -1;
        const punctuation = ['.', '!', '?', ':', ';', '\n'];
        for (const p of punctuation) {
            const index = text.lastIndexOf(p, chunkEnd);
            if (index > i && index > lastPunctuation) {
                lastPunctuation = index;
            }
        }
        
        if (lastPunctuation !== -1) {
            chunkEnd = lastPunctuation + 1;
        } else {
            const lastSpace = text.lastIndexOf(' ', chunkEnd);
            if (lastSpace > i) {
                chunkEnd = lastSpace + 1;
            }
        }
        
        chunks.push(text.substring(i, chunkEnd));
        i = chunkEnd;
    }
    return chunks.map(c => c.trim()).filter(c => c);
};

const parseDialogue = (text: string, sourceLang: Language, targetLang: Language): DialogueSegment[] => {
    const segments: DialogueSegment[] = [];
    const lines = text.split('\n').filter(line => line.trim() !== '');
    lines.forEach((line) => {
        const isQuestion = line.match(/^Frage\s*\d+:/i);
        const isAnswer = line.match(/^Antwort\s*\d+:/i);
        if (isQuestion) {
            segments.push({
                type: 'Frage',
                text: line.replace(/^Frage\s*\d+:/i, '').trim(),
                lang: sourceLang
            });
        } else if (isAnswer) {
            segments.push({
                type: 'Antwort',
                text: line.replace(/^Antwort\s*\d+:/i, '').trim(),
                lang: targetLang
            });
        }
    });
    return segments;
};


const PracticeArea = ({ 
    isLoading, loadingMessage, originalText, setOriginalText, onRecordingFinished, 
    getFeedback, userTranscript, setUserTranscript, feedback, error, settings, 
    exerciseStarted, onPremiumVoiceAuthError, isPremiumVoiceAvailable 
}: {
    isLoading: boolean; loadingMessage: string; originalText: string; setOriginalText: (text: string) => void;
    onRecordingFinished: (rawTranscript: string) => void; getFeedback: (setActiveTab: (tab: PracticeAreaTab) => void) => void; 
    userTranscript: string; setUserTranscript: (transcript: string) => void;
    feedback: Feedback | null; error: string | null; settings: Settings; exerciseStarted: boolean;
    onPremiumVoiceAuthError: () => void;
    isPremiumVoiceAvailable: boolean;
}) => {
    const [isEditing, setIsEditing] = useState(false);
    const [editedText, setEditedText] = useState(originalText);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [isSynthesizing, setIsSynthesizing] = useState(false);
    const [playbackTime, setPlaybackTime] = useState(0);
    const [recordingTime, setRecordingTime] = useState(0);
    const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
    const [voicesLoaded, setVoicesLoaded] = useState(false);
    const [playbackError, setPlaybackError] = useState<string | null>(null);
    const [recognitionError, setRecognitionError] = useState<string | null>(null);
    const [effectiveVoiceQuality, setEffectiveVoiceQuality] = useState(settings.voiceQuality);
    const [premiumAudioSrc, setPremiumAudioSrc] = useState<string | null>(null);

    const [isEditingTranscript, setIsEditingTranscript] = useState(false);
    const [editedTranscript, setEditedTranscript] = useState(userTranscript);
    const [activeTab, setActiveTab] = useState<PracticeAreaTab>('original');

    // State for Dialogue Interpreting mode
    const [dialogueSegments, setDialogueSegments] = useState<DialogueSegment[]>([]);
    const [currentSegmentIndex, setCurrentSegmentIndex] = useState(-1);
    const [isCurrentSegmentVisible, setIsCurrentSegmentVisible] = useState(false);
    const [dialogueState, setDialogueState] = useState<DialogueState>('idle');
    const [dialogueTranscripts, setDialogueTranscripts] = useState<string[]>([]);
    const [statusText, setStatusText] = useState('Klicken Sie auf ▶️, um das Interview zu starten.');
    const [premiumDialogueAudio, setPremiumDialogueAudio] = useState<Record<number, string>>({});


    const recognitionRef = useRef<SpeechRecognition | null>(null);
    const audioRef = useRef<HTMLAudioElement>(null);
    const timerRef = useRef<number | null>(null);

    const formatTime = (seconds: number) => new Date(seconds * 1000).toISOString().slice(14, 19);

    // --- LIFECYCLE & SETUP ---
    useEffect(() => {
        const handleVoicesChanged = () => {
            if (speechSynthesis.getVoices().length > 0) {
                setVoicesLoaded(true);
            }
        };

        speechSynthesis.onvoiceschanged = handleVoicesChanged;
        handleVoicesChanged(); // Initial check in case voices are already loaded

        // Cleanup on unmount
        return () => {
            speechSynthesis.onvoiceschanged = null;
            speechSynthesis.cancel();
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current.src = "";
            }
            if (recognitionRef.current) {
                (recognitionRef.current as any)._intentionalStop = true;
                recognitionRef.current.stop();
            }
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, []);


    useEffect(() => {
        setEditedText(originalText);
        setEditedTranscript(userTranscript);
        setPlaybackError(null);
        setRecognitionError(null);
        setActiveTab('original');
        setEffectiveVoiceQuality(settings.voiceQuality);
        setPremiumAudioSrc(null); // Reset cached audio
        
        // Also stop and clear any existing audio to prevent playing stale content
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.src = "";
        }
        setIsPlaying(false);

        if (settings.mode === 'Gesprächsdolmetschen' && originalText) {
            const segments = parseDialogue(originalText, settings.sourceLang, settings.targetLang);
            setDialogueSegments(segments);
            setDialogueState('idle');
            setCurrentSegmentIndex(-1);
            setDialogueTranscripts([]);
            setPremiumDialogueAudio({}); // Reset dialogue audio cache
            setStatusText('Klicken Sie auf ▶️, um das Interview zu starten.');
        }

    }, [originalText, userTranscript, settings.mode, settings.sourceLang, settings.targetLang, settings.voiceQuality]);
    
    useEffect(() => {
      if (dialogueState === 'finished' && settings.mode === 'Gesprächsdolmetschen') {
        const fullTranscript = dialogueTranscripts.join(' ');
        onRecordingFinished(fullTranscript);
        setActiveTab('transcript');
        setStatusText('Übung abgeschlossen. Ihr vollständiges Transkript wird verarbeitet.');
      }
    }, [dialogueState, dialogueTranscripts, onRecordingFinished, settings.mode]);

    // This effect ensures the playback speed is updated whenever the slider changes.
    useEffect(() => {
        if (audioRef.current) {
            audioRef.current.playbackRate = playbackSpeed;
        }
    }, [playbackSpeed]);


    // --- EDITING HANDLERS ---
    const handleEdit = () => { setIsEditing(true); setEditedText(originalText); };
    const handleSave = () => { setOriginalText(editedText); setIsEditing(false); };
    const handleEditTranscript = () => setIsEditingTranscript(true);
    const handleSaveTranscript = () => { setUserTranscript(editedTranscript); setIsEditingTranscript(false); };
    
    // --- STANDARD (BROWSER) VOICE PLAYBACK ---
    const playWithBrowserVoice = useCallback(() => {
        speechSynthesis.cancel();
        setPlaybackTime(0);

        const chunks = splitTextIntoChunks(originalText);
        if (chunks.length === 0) return;
        
        const targetLang = LANGUAGE_CODES[settings.sourceLang];
        const availableVoices = speechSynthesis.getVoices();
        const targetVoices = availableVoices.filter(v => v.lang === targetLang);
        let bestVoice: SpeechSynthesisVoice | undefined;
        if (targetVoices.length > 0) bestVoice = targetVoices.find(v => v.localService) || targetVoices[0];
        
        const utterances = chunks.map(chunk => {
            const utterance = new SpeechSynthesisUtterance(chunk);
            utterance.lang = targetLang;
            utterance.rate = playbackSpeed;
            if (bestVoice) utterance.voice = bestVoice;
            
            utterance.onerror = (e: SpeechSynthesisErrorEvent) => {
                if (e.error === 'interrupted') return;
                console.error(`Speech synthesis error: ${e.error}. For language ${utterance.lang}.`, e);
                let userMessage = 'Fehler bei der Sprachausgabe.';
                if (e.error === 'language-unavailable') userMessage = `Keine Stimme für die Sprache '${settings.sourceLang}' in Ihrem Browser verfügbar.`;
                setPlaybackError(userMessage);
                setIsPlaying(false);
                speechSynthesis.cancel();
            };
            return utterance;
        });

        if (utterances.length > 0) {
            utterances[utterances.length - 1].onend = () => setIsPlaying(false);
        }

        utterances.forEach(utterance => speechSynthesis.speak(utterance));
        setIsPlaying(true);
    }, [originalText, settings.sourceLang, playbackSpeed]);

    const playSegmentWithBrowserVoice = useCallback((index: number) => {
        if (index < 0 || index >= dialogueSegments.length) return;
        const segment = dialogueSegments[index];

        speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(segment.text);
        const targetLangCode = LANGUAGE_CODES[segment.lang];
        const availableVoices = speechSynthesis.getVoices();
        const targetVoices = availableVoices.filter(v => v.lang === targetLangCode);
        let bestVoice: SpeechSynthesisVoice | undefined;
        if (targetVoices.length > 0) {
            bestVoice = targetVoices.find(v => v.localService) || targetVoices[0];
        }
        
        utterance.lang = targetLangCode;
        if (bestVoice) utterance.voice = bestVoice;
        utterance.rate = playbackSpeed;

        utterance.onstart = () => {
            setIsPlaying(true);
            setDialogueState('playing');
            setStatusText(`${segment.type} ${Math.floor(index / 2) + 1} wird vorgelesen...`);
        };
        utterance.onend = () => {
          setIsPlaying(false);
          setDialogueState('waiting_for_record');
          setStatusText(`Drücken Sie 🎤, um Ihre Verdolmetschung aufzunehmen.`);
        };
        
        utterance.onerror = (e: SpeechSynthesisErrorEvent) => {
          console.error(`Speech synthesis error: ${e.error}. For language ${utterance.lang}.`, e);
          let userMessage = 'Fehler bei der Sprachausgabe.';
          if (e.error === 'language-unavailable') userMessage = `Keine Stimme für die Sprache '${segment.lang}' in Ihrem Browser verfügbar.`;
          setPlaybackError(userMessage);
          setIsPlaying(false);
          setDialogueState('idle');
        };

        speechSynthesis.speak(utterance);
    }, [dialogueSegments, playbackSpeed]);


    // --- DIALOGUE MODE LOGIC ---
    const playSegment = useCallback(async (index: number) => {
        if (index < 0 || index >= dialogueSegments.length) return;
        
        const segment = dialogueSegments[index];
        
        const playPremiumAudio = (src: string) => {
            if (audioRef.current) {
                audioRef.current.src = src;
                audioRef.current.playbackRate = playbackSpeed;
    
                audioRef.current.onended = () => {
                  setIsPlaying(false);
                  setDialogueState('waiting_for_record');
                  setStatusText(`Drücken Sie 🎤, um Ihre Verdolmetschung aufzunehmen.`);
                  if(audioRef.current) audioRef.current.onended = null;
                };
    
                audioRef.current.play().then(() => {
                    setIsPlaying(true);
                    setDialogueState('playing');
                    setStatusText(`${segment.type} ${Math.floor(index / 2) + 1} wird vorgelesen...`);
                }).catch(playError => {
                    console.error("Audio playback failed in dialogue mode:", playError);
                    setPlaybackError("Fehler beim Abspielen des Segments. Ihr Browser hat die Wiedergabe möglicherweise blockiert.");
                    setIsPlaying(false);
                    setDialogueState('idle');
                    if (audioRef.current) audioRef.current.onended = null;
                });
            }
        };

        if (effectiveVoiceQuality === 'Premium') {
            if (premiumDialogueAudio[index]) {
                playPremiumAudio(premiumDialogueAudio[index]);
                return;
            }
            try {
                setDialogueState('synthesizing');
                setStatusText(`Generiere Premium-Stimme für ${segment.type} ${Math.floor(index / 2) + 1}...`);
                const audioContent = await synthesizeSpeechGoogleCloud(segment.text, segment.lang, process.env.API_KEY!);
                const audioSrc = `data:audio/mp3;base64,${audioContent}`;
                setPremiumDialogueAudio(prev => ({ ...prev, [index]: audioSrc }));
                playPremiumAudio(audioSrc);
            } catch(err) {
                 console.error("Error playing premium segment:", err);
                 setDialogueState('idle');
                 if (err instanceof TtsAuthError) {
                    if (isPremiumVoiceAvailable) {
                        onPremiumVoiceAuthError();
                        setPlaybackError("Premium-Stimme ist mit dem API-Schlüssel nicht verfügbar. Die App wechselt zur Standard-Stimme.");
                    }
                    setEffectiveVoiceQuality('Standard');
                    playSegmentWithBrowserVoice(index);
                 } else {
                    setPlaybackError((err as Error).message);
                    setIsPlaying(false);
                 }
            }
        } else {
            playSegmentWithBrowserVoice(index);
        }

    }, [dialogueSegments, playbackSpeed, effectiveVoiceQuality, playSegmentWithBrowserVoice, isPremiumVoiceAvailable, onPremiumVoiceAuthError, premiumDialogueAudio]);

    const advanceToNextSegment = useCallback(() => {
      const nextIndex = currentSegmentIndex + 1;
      if (nextIndex >= dialogueSegments.length) {
          setDialogueState('finished');
      } else {
          setCurrentSegmentIndex(nextIndex);
          setIsCurrentSegmentVisible(false);
          playSegment(nextIndex);
      }
    }, [currentSegmentIndex, dialogueSegments, playSegment]);

    // --- GENERIC PLAYBACK & RECORDING ---
    const handlePlayPause = async () => {
        if (settings.mode === 'Gesprächsdolmetschen') {
            if (dialogueState === 'idle') {
                advanceToNextSegment();
            }
            return;
        }

        if (isPlaying) {
            speechSynthesis.cancel();
            if (audioRef.current) audioRef.current.pause();
            setIsPlaying(false);
        } else { // Not playing, so start playing
            setPlaybackError(null);
            
            if (effectiveVoiceQuality === 'Premium') {
                // If we have already synthesized the audio, just play it.
                if (premiumAudioSrc && audioRef.current) {
                    // Re-assigning the src is a robust way to ensure the audio element is ready to play again.
                    audioRef.current.src = premiumAudioSrc;
                    audioRef.current.currentTime = 0;
                    audioRef.current.play().then(() => {
                        setIsPlaying(true);
                    }).catch(playError => {
                        console.error("Audio playback failed:", playError);
                        setPlaybackError("Fehler beim Abspielen der Audiodatei. Ihr Browser hat die Wiedergabe möglicherweise blockiert.");
                        setIsPlaying(false);
                    });
                    return;
                }

                // Otherwise, synthesize for the first time
                try {
                    setIsSynthesizing(true);
                    const audioContent = await synthesizeSpeechGoogleCloud(originalText, settings.sourceLang, process.env.API_KEY!);
                    setIsSynthesizing(false);
                    const audioSrc = `data:audio/mp3;base64,${audioContent}`;
                    setPremiumAudioSrc(audioSrc); // Cache the audio source

                    if (audioRef.current) {
                        audioRef.current.src = audioSrc;
                        audioRef.current.playbackRate = playbackSpeed;
                        audioRef.current.play().then(() => {
                            setIsPlaying(true);
                        }).catch(playError => {
                            console.error("Audio playback failed:", playError);
                            setPlaybackError("Fehler beim Abspielen der Audiodatei. Ihr Browser hat die Wiedergabe möglicherweise blockiert.");
                            setIsPlaying(false);
                        });
                    }
                } catch (err) {
                    console.error("Premium playback error:", err);
                    if (err instanceof TtsAuthError) {
                        if (isPremiumVoiceAvailable) {
                            onPremiumVoiceAuthError();
                            setPlaybackError("Premium-Stimme ist mit dem API-Schlüssel nicht verfügbar. Die App wechselt zur Standard-Stimme.");
                        }
                        setEffectiveVoiceQuality('Standard');
                        playWithBrowserVoice();
                    } else {
                        setPlaybackError((err as Error).message || "Fehler bei der Generierung der Premium-Stimme.");
                    }
                    setIsSynthesizing(false);
                }
            } else {
                 playWithBrowserVoice();
            }
        }
    };
    
    const handleRecord = () => {
        if (isRecording) {
            if (recognitionRef.current) {
                (recognitionRef.current as any)._intentionalStop = true;
                recognitionRef.current.stop();
            }
        } else {
            setRecognitionError(null);
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                setRecognitionError("Speech Recognition wird von Ihrem Browser nicht unterstützt.");
                return;
            }
            
            const recognition = new SpeechRecognition();
            (recognition as any)._intentionalStop = false; // Custom flag to manage restarts

            const isDialogue = settings.mode === 'Gesprächsdolmetschen';
            const currentSegment = dialogueSegments[currentSegmentIndex];
            const targetLang = isDialogue 
              ? (currentSegment.lang === settings.sourceLang ? settings.targetLang : settings.sourceLang)
              : settings.targetLang;
            
            recognition.lang = LANGUAGE_CODES[targetLang];
            recognition.continuous = true;
            recognition.interimResults = false;
            
            let finalTranscript = '';
            recognition.onresult = (event) => {
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript + ' ';
                    }
                }
            };

            recognition.onstart = () => {
                setIsRecording(true);
                if(isDialogue) {
                    setDialogueState('recording');
                    setStatusText('Aufnahme läuft... Drücken Sie ⏹️ zum Beenden.');
                }
            };

            recognition.onend = () => {
                const wasIntentional = (recognition as any)._intentionalStop;
                if (wasIntentional) {
                    // User-initiated stop or an error occurred. Finalize.
                    setIsRecording(false);
                    if (isDialogue) {
                        setDialogueTranscripts(prev => [...prev, finalTranscript.trim()]);
                        advanceToNextSegment();
                    } else {
                        onRecordingFinished(finalTranscript);
                        setActiveTab('transcript');
                    }
                } else {
                    // Timeout due to silence. Restart automatically.
                    try {
                        recognition.start();
                    } catch (e) {
                        console.error("Failed to automatically restart speech recognition:", e);
                        setIsRecording(false); // Stop if restart fails
                    }
                }
            };

            recognition.onerror = (event) => {
                (recognition as any)._intentionalStop = true; // Prevent restart on error
                console.error("Speech recognition error:", event.error, event.message);
                let errorMessage;
                switch (event.error) {
                    case 'no-speech': errorMessage = "Es wurde keine Sprache erkannt. Die Aufnahme wurde beendet."; break;
                    case 'audio-capture': errorMessage = "Problem mit dem Mikrofon."; break;
                    case 'not-allowed': errorMessage = "Zugriff auf das Mikrofon verweigert."; break;
                    case 'network': errorMessage = "Netzwerkfehler bei der Spracherkennung."; break;
                    case 'language-not-supported': errorMessage = `Die Sprache wird nicht unterstützt.`; break;
                    default: errorMessage = `Ein unbekannter Fehler ist aufgetreten (${event.error}).`;
                }
                setRecognitionError(errorMessage);
                setIsRecording(false);
                if(isDialogue) setDialogueState('waiting_for_record');
            };
            
            recognition.start();
            recognitionRef.current = recognition;
        }
    };
    
    useEffect(() => {
        if (isPlaying || isRecording) {
            timerRef.current = window.setInterval(() => {
                if (isPlaying) setPlaybackTime(t => t + 1);
                if (isRecording) setRecordingTime(t => t + 1);
            }, 1000);
        } else {
            if(timerRef.current) clearInterval(timerRef.current);
        }
        return () => { if(timerRef.current) clearInterval(timerRef.current); }
    }, [isPlaying, isRecording]);

    const isSimultaneousMode = settings.mode === 'Simultandolmetschen' || settings.mode === 'Shadowing';
    
    // --- RENDER ---
    if (isLoading || isSynthesizing) {
        return (
            <section className="panel practice-area">
                <div className="loader-overlay">
                    <div className="spinner"></div>
                    <p>{isSynthesizing ? 'Generiere Premium-Stimme...' : loadingMessage}</p>
                </div>
            </section>
        );
    }

    if (!exerciseStarted) {
        return (
            <section className="panel practice-area">
                <div style={{textAlign: 'center', margin: 'auto'}}>
                    <p>Konfigurieren Sie Ihre Übungssitzung und klicken Sie auf "Loslegen".</p>
                </div>
            </section>
        );
    }
    

    if (settings.mode === 'Gesprächsdolmetschen') {
        const currentSegment = dialogueSegments[currentSegmentIndex];
        const isFinished = dialogueState === 'finished' || currentSegmentIndex >= dialogueSegments.length;
        
        return (
          <section className="panel practice-area">
            {!isFinished ? (
              <>
                <div className="dialogue-practice-container">
                    <div className="practice-header">
                        <h2>{settings.mode} Übung</h2>
                        {currentSegment && <span>{currentSegment.type} {Math.floor(currentSegmentIndex / 2) + 1} / {dialogueSegments.length / 2}</span>}
                    </div>
                    <div className="dialogue-status">{statusText}</div>

                    <div className="current-segment-display">
                        {dialogueState === 'idle' || !currentSegment ? (
                           <div className="segment-text-hidden">Das Interview beginnt in Kürze...</div>
                        ) : isCurrentSegmentVisible ? (
                            <p className="segment-text">{currentSegment.text}</p>
                        ) : (
                            <div className="segment-text-hidden">Der Text ist verborgen, um das Hörverstehen zu trainieren.</div>
                        )}
                    </div>
                    
                    <div className="segment-controls">
                        <button 
                            className="btn btn-secondary" 
                            onClick={() => setIsCurrentSegmentVisible(true)} 
                            disabled={!currentSegment || isCurrentSegmentVisible || dialogueState === 'idle'}
                        >
                            Text anzeigen
                        </button>
                    </div>
                </div>
                
                <div className="controls">
                    <button className="control-btn" onClick={handlePlayPause} disabled={dialogueState !== 'idle' || !voicesLoaded} title={voicesLoaded ? "Interview starten" : "Stimmen werden geladen..."}>
                        ▶️
                    </button>
                    <button className={`control-btn ${isRecording ? 'recording' : ''}`} onClick={handleRecord} disabled={dialogueState !== 'waiting_for_record' && !isRecording} title={isRecording ? 'Stop' : 'Aufnahme'}>
                        {isRecording ? '⏹️' : '🎤'}
                    </button>
                     <div className="timers">
                        <span>Aufnahme: {formatTime(recordingTime)}</span>
                    </div>
                </div>
              </>
            ) : (
                <div className="tab-content">
                    {/* Final transcript and feedback for dialogue mode */}
                    <div className="tab-pane-content">
                        {feedback ? (
                            <>
                                <div className="feedback-card" style={{ marginBottom: '1.5rem' }}>
                                    <h3>KI-Feedback: Zusammenfassung</h3>
                                    <p>{feedback.summary}</p>
                                </div>
                                <div className="feedback-card">
                                    <h3>Fehleranalyse</h3>
                                    {feedback.errorAnalysis?.length > 0 ? (
                                        <div className="error-analysis">
                                            <table>
                                                <thead>
                                                    <tr><th>Original</th><th>Ihre Version</th><th>Vorschlag</th></tr>
                                                </thead>
                                                <tbody>
                                                    {feedback.errorAnalysis.map((err, i) => (
                                                        <tr key={i}><td>{err.original}</td><td>{err.interpretation}</td><td>{err.suggestion}</td></tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    ) : <p style={{ paddingTop: '1rem' }}>Keine signifikanten Fehler gefunden.</p>}
                                </div>
                            </>
                        ) : (
                             <div className="feedback-section">
                                <div className="feedback-card">
                                  <h3>Ihre Verdolmetschung (Gesamtes Transkript)</h3>
                                  <textarea className="transcript-textarea" readOnly value={userTranscript} />
                                </div>
                                <div className="feedback-actions">
                                  <button className="btn btn-primary" onClick={() => getFeedback(setActiveTab)} disabled={isLoading}>Feedback anfordern</button>
                                </div>
                              </div>
                        )}
                    </div>
                </div>
            )}
            {error && <div className="error-message">{error}</div>}
            {playbackError && <div className="error-message">{playbackError}</div>}
            {recognitionError && <div className="error-message">{recognitionError}</div>}
            <audio ref={audioRef} onEnded={() => setIsPlaying(false)} />
          </section>
        );
    }
    
    // Special layout for Sight Translation (Stegreifübersetzen)
    if (settings.mode === 'Stegreifübersetzen') {
        return (
            <section className="panel practice-area">
                <div className="practice-header">
                    <h2>{settings.mode} Übung</h2>
                    <div>
                        {isEditing ? (
                            <button className="btn btn-primary" onClick={handleSave} style={{width:'auto', padding:'0.4rem 0.8rem'}}>Speichern</button>
                        ) : (
                            <button className="btn btn-secondary" onClick={handleEdit} style={{width:'auto', padding:'0.4rem 0.8rem'}}>Bearbeiten</button>
                        )}
                        <span className="char-count" style={{marginLeft: '1rem'}}>Zeichen: {isEditing ? editedText.length : originalText.length}</span>
                    </div>
                </div>
                <div className="original-text-container">
                  <textarea className="original-text" readOnly={!isEditing} value={isEditing ? editedText : originalText} onChange={(e) => setEditedText(e.target.value)} />
                </div>
                <div className="controls">
                    <button className={`control-btn ${isRecording ? 'recording' : ''}`} onClick={handleRecord} disabled={isEditing} title={isRecording ? 'Stop' : 'Aufnahme'}>
                        {isRecording ? '⏹️' : '🎤'}
                    </button>
                    <div className="timers">
                        <span>Aufnahme: {formatTime(recordingTime)}</span>
                    </div>
                </div>
                {error && <div className="error-message">{error}</div>}
                {recognitionError && <div className="error-message">{recognitionError}</div>}
                 {userTranscript && (
                    <div className="tab-pane-content" style={{borderTop: '1px solid var(--border-color)', marginTop: '1rem'}}>
                         {feedback ? (
                           <div className="feedback-card" style={{marginTop: '1.5rem'}}>
                                <h3>KI-Feedback</h3>
                                <p>{feedback.summary}</p>
                           </div>
                        ) : (
                             <div className="feedback-section" style={{paddingTop: '1rem'}}>
                                <div className="feedback-card">
                                  <h3>Ihre Verdolmetschung (Transkript)</h3>
                                  <textarea className="transcript-textarea" readOnly value={userTranscript} />
                                </div>
                                <div className="feedback-actions">
                                  <button className="btn btn-primary" onClick={() => getFeedback(setActiveTab)} disabled={isLoading}>Feedback anfordern</button>
                                </div>
                              </div>
                        )}
                    </div>
                 )}
            </section>
        );
    }
    
    // Tabbed layout for all other modes
    return (
        <section className="panel practice-area">
            <div className="practice-header">
                <h2>{settings.mode} Übung</h2>
                {activeTab === 'original' && (
                    <div>
                        {isEditing ? (
                            <button className="btn btn-primary" onClick={handleSave} style={{width:'auto', padding:'0.4rem 0.8rem'}}>Speichern</button>
                        ) : (
                            <button className="btn btn-secondary" onClick={handleEdit} style={{width:'auto', padding:'0.4rem 0.8rem'}}>Bearbeiten</button>
                        )}
                        <span className="char-count" style={{marginLeft: '1rem'}}>Zeichen: {isEditing ? editedText.length : originalText.length}</span>
                    </div>
                )}
            </div>

            <nav className="tab-nav">
                <button className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`} onClick={() => setActiveTab('original')}>
                    Originaltext
                </button>
                <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')} disabled={!userTranscript}>
                    Transkript
                </button>
                <button className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} onClick={() => setActiveTab('feedback')} disabled={!feedback}>
                    KI-Feedback
                </button>
                <button className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')} disabled={!feedback}>
                    Fehleranalyse
                </button>
            </nav>

            <div className="tab-content">
                {activeTab === 'original' && (
                    <div className="original-text-container">
                        <textarea className="original-text" readOnly={!isEditing} value={isEditing ? editedText : originalText} onChange={(e) => setEditedText(e.target.value)} />
                    </div>
                )}
                {activeTab === 'transcript' && userTranscript && (
                    <div className="tab-pane-content">
                        <div className="feedback-card" style={{flexGrow: 1, display: 'flex', flexDirection: 'column'}}>
                          <div className="transcript-header">
                            <h3>Ihre Verdolmetschung (Transkript)</h3>
                            {isEditingTranscript ? (
                              <button className="btn btn-primary" onClick={handleSaveTranscript}>Korrekturen übernehmen</button>
                            ) : (
                              <button className="btn btn-secondary" onClick={handleEditTranscript}>Transkript korrigieren</button>
                            )}
                          </div>
                          <textarea
                            className="transcript-textarea"
                            readOnly={!isEditingTranscript}
                            value={isEditingTranscript ? editedTranscript : userTranscript}
                            onChange={(e) => setEditedTranscript(e.target.value)}
                            aria-label="Transkript Ihrer Verdolmetschung"
                          />
                        </div>
                        <div className="feedback-actions">
                          <button className="btn btn-primary" onClick={() => getFeedback(setActiveTab)} disabled={isLoading || isEditingTranscript}>Feedback anfordern</button>
                        </div>
                    </div>
                )}
                {activeTab === 'feedback' && feedback && (
                     <div className="tab-pane-content">
                        <div className="feedback-card" style={{marginBottom: '1.5rem'}}>
                            <h3>KI-Feedback: Zusammenfassung</h3>
                            <p>{feedback.summary}</p>
                        </div>
                        <div className="feedback-card ratings-panel">
                             <h3>Bewertung</h3>
                             <div className="ratings">
                                <div className="rating-item">
                                    <span>{feedback.ratings.content}/10</span>
                                    <span>Inhalt</span>
                                </div>
                                <div className="rating-item">
                                    <span>{feedback.ratings.expression}/10</span>
                                    <span>Ausdruck</span>
                                </div>
                                <div className="rating-item">
                                    <span>{feedback.ratings.terminology}/10</span>
                                    <span>Terminologie</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
                {activeTab === 'analysis' && feedback && (
                    <div className="tab-pane-content">
                        <div className="feedback-card" style={{flexGrow: 1}}>
                            <h3>Fehleranalyse</h3>
                            {feedback.errorAnalysis?.length > 0 ? (
                                <div className="error-analysis">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Original</th>
                                            <th>Ihre Version</th>
                                            <th>Vorschlag</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {feedback.errorAnalysis.map((err, i) => (
                                            <tr key={i}>
                                                <td>{err.original}</td>
                                                <td>{err.interpretation}</td>
                                                <td>{err.suggestion}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                                </div>
                            ) : <p style={{paddingTop: '1rem'}}>Keine signifikanten Fehler gefunden.</p>}
                        </div>
                    </div>
                )}
            </div>
            
            {activeTab === 'original' && (
                <div className="controls">
                    <button className="control-btn" onClick={handlePlayPause} disabled={isEditing || (isRecording && !isSimultaneousMode) || (!voicesLoaded && settings.voiceQuality === 'Standard')} title={voicesLoaded || settings.voiceQuality === 'Premium' ? (isPlaying ? 'Stop' : 'Play') : 'Stimmen werden geladen...'}>
                        {isPlaying ? '⏹️' : '▶️'}
                    </button>
                    <button className={`control-btn ${isRecording ? 'recording' : ''}`} onClick={handleRecord} disabled={isEditing || (isPlaying && !isSimultaneousMode)} title={isRecording ? 'Stop' : 'Aufnahme'}>
                        {isRecording ? '⏹️' : '🎤'}
                    </button>
                    <div className="tempo-control">
                        <label htmlFor="tempo">Tempo:</label>
                        <input type="range" id="tempo" min="0.5" max="2" step="0.1" value={playbackSpeed} onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))} />
                        <span>{playbackSpeed.toFixed(1)}x</span>
                    </div>
                    <div className="timers">
                        <span>Lesezeit: {formatTime(playbackTime)}</span>
                        <span>Aufnahme: {formatTime(recordingTime)}</span>
                    </div>
                </div>
            )}
            
            <audio ref={audioRef} onEnded={() => setIsPlaying(false)} />
            {error && <div className="error-message">{error}</div>}
            {playbackError && <div className="error-message">{playbackError}</div>}
            {recognitionError && <div className="error-message">{recognitionError}</div>}
        </section>
    );
};

const container = document.getElementById('root');
const root = createRoot(container!);
root.render(<App />);