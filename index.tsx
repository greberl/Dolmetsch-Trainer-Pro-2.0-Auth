import React, { useState, useRef, useCallback, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";

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
type DialogueState = 'idle' | 'ready' | 'synthesizing' | 'playing' | 'waiting_for_record' | 'recording' | 'finished' | 'starting';
type PracticeAreaTab = 'original' | 'transcript' | 'feedback';


interface DialogueSegment {
    type: 'Frage' | 'Antwort';
    text: string;
    lang: Language;
}

interface StructuredDialogueResult {
  originalSegment: DialogueSegment;
  userInterpretation: string;
  interpretationLang: Language;
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

// --- HELPER HOOKS & COMPONENTS ---
/**
 * Custom hook to get the previous value of a prop or state.
 */
function usePrevious<T>(value: T): T | undefined {
    const ref = useRef<T>();
    useEffect(() => {
        ref.current = value;
    });
    return ref.current;
}

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

    // Fix: The response body can only be consumed once.
    // Parse the JSON response here and use the resulting object for both error and success handling.
    const data = await response.json();

    if (!response.ok) {
        console.error("Google TTS API Error:", JSON.stringify(data, null, 2));
        const message = `Fehler bei der Sprachsynthese: ${data.error?.message || 'Unbekannter Fehler'}`;
        // Check for common API key-related errors to allow for graceful fallback.
        if (response.status === 403 || response.status === 400 || data.error?.message?.toLowerCase().includes('api key not valid')) {
             throw new TtsAuthError(message);
        }
        throw new Error(message);
    }

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
  const [dialogueFinished, setDialogueFinished] = useState(false);
  const [structuredDialogueResults, setStructuredDialogueResults] = useState<StructuredDialogueResult[] | null>(null);
  // Fix: Use a lazy initializer for `useState` to resolve an error where `Date.now()` was not being accepted as an argument.
  const [exerciseId, setExerciseId] = useState(() => Date.now());
  
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
    setExerciseId(Date.now());

    try {
        if (currentSettings.mode === 'Gesprächsdolmetschen') {
            const prompt = `
Du bist ein Assistent zur Erstellung von Dolmetschübungen. Erstelle ein Interview-Skript zum Thema "${currentSettings.topic}".
Das Skript besteht aus genau 6 Fragen und 6 Antworten.

**WICHTIGE REGELN FÜR DIE SPRACHEN:**
- Alle 6 Fragen (Frage 1, Frage 2, etc.) MÜSSEN auf ${currentSettings.sourceLang} sein.
- Alle 6 Antworten (Antwort 1, Antwort 2, etc.) MÜSSEN auf ${currentSettings.targetLang} sein.
- Halte dich strikt an diesen Sprachenwechsel. Es darf keine Abweichungen geben.

Die Länge jeder Frage und jeder Antwort soll ungefähr dem Umfang von "${currentSettings.qaLength}" entsprechen.

**FORMATIERUNG (SEHR WICHTIG):**
- Jede Zeile MUSS mit "Frage X:" oder "Antwort X:" beginnen, gefolgt vom Text.
- Gib NUR das Skript zurück. Keinen einleitenden Text, keine Kommentare, keine Zusammenfassungen.

**Beispiel für ${currentSettings.sourceLang}=Deutsch und ${currentSettings.targetLang}=Englisch:**
Frage 1: Was sind die größten Herausforderungen beim Klimaschutz?
Antwort 1: The biggest challenges are the transition to renewable energy and international cooperation.
Frage 2: Welche Rolle spielt die Technologie dabei?
Antwort 2: Technology plays a crucial role, for example in developing more efficient solar panels.
Frage 3: Wie können einzelne Personen beitragen?
Antwort 3: Individuals can contribute by reducing their carbon footprint, for instance, through less consumption and more recycling.
`;
            const response = await ai.models.generateContent({ model, contents: prompt });
            setOriginalText(response.text || '');
        } else {
            const isSpeechMode = ["Vortragsdolmetschen", "Simultandolmetschen", "Shadowing"].includes(currentSettings.mode);
            const { min, max } = isSpeechMode
                ? SPEECH_LENGTH_CONFIG[currentSettings.speechLength]
                : TEXT_LENGTH_CONFIG[currentSettings.mode];

            const promptText = currentSettings.mode === 'Stegreifübersetzen'
                ? `Erstelle einen sachlichen Text zum Thema "${currentSettings.topic}". WICHTIG: Der Text muss vollständig auf ${currentSettings.sourceLang} verfasst sein. Der Text ist für eine Stegreifübersetzungsübung und sollte keine direkte Anrede (wie "Sehr geehrte Damen und Herren") oder Dialoge enthalten.`
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
    } catch (err) {
        console.error("Error generating text:", err);
        setError("Fehler bei der Texterstellung. Bitte versuchen Sie es erneut.");
    } finally {
        setIsLoading(false);
    }
  };

  const handleStartExercise = (currentSettings: Settings, text?: string) => {
    setError(null);
    setFeedback(null);
    setUserTranscript('');
    setDialogueFinished(false);
    setStructuredDialogueResults(null);
    setSettings(currentSettings); // Make sure settings are updated
    setExerciseStarted(true);
    setExerciseId(Date.now()); // Reset exercise state

    if (currentSettings.sourceType === 'upload' && text) {
        setOriginalText(text);
    } else {
        generateText(currentSettings);
    }
  };

  const handleBackToSettings = () => {
    setExerciseStarted(false);
    setOriginalText('');
    setUserTranscript('');
    setFeedback(null);
    setError(null);
    setDialogueFinished(false);
    setStructuredDialogueResults(null);
  };
  
  const handleDialogueFinish = (results: StructuredDialogueResult[]) => {
      setDialogueFinished(true);
      setStructuredDialogueResults(results);
  };

  const getFeedback = async () => {
    setIsLoading(true);
    setLoadingMessage('Analysiere deine Verdolmetschung...');
    setError(null);
    setFeedback(null);
    
    try {
      let prompt;
      if (settings.mode === 'Gesprächsdolmetschen' && structuredDialogueResults) {
          const dialogueSummary = structuredDialogueResults.map((res, index) => 
`Segment ${index + 1} (${res.originalSegment.type} auf ${res.originalSegment.lang}):
Original: "${res.originalSegment.text}"
Deine Verdolmetschung auf ${res.interpretationLang}: "${res.userInterpretation}"`
          ).join('\n\n');

          prompt = `
Du bist ein erfahrener Dolmetschlehrer. Analysiere die folgende Gesprächsdolmetsch-Übung.
Das Gespräch war zum Thema "${settings.topic}". 
Die Ausgangssprache für Fragen war ${settings.sourceLang}. Die Ausgangssprache für Antworten war ${settings.targetLang}.

Hier ist der gesamte Gesprächsverlauf mit den Verdolmetschungen des Nutzers:
${dialogueSummary}

Gib Feedback in drei Teilen:
1.  **Zusammenfassung:** Eine kurze, konstruktive Gesamtbewertung.
2.  **Bewertungen:** Bewerte Inhalt, Ausdruck und Terminologie auf einer Skala von 1 bis 5 Sternen.
3.  **Fehleranalyse:** Liste bis zu 5 konkrete und wichtige Fehler auf. Für jeden Fehler: Gib den Originaltext an, die fehlerhafte Verdolmetschung des Nutzers und einen besseren Vorschlag. Erkläre kurz, warum der Vorschlag besser ist (z.B. "präziserer Begriff", "idiomatischer", "grammatikalisch korrekt"). Wenn es keine oder nur wenige Fehler gibt, erwähne das positiv.

Stelle sicher, dass die Antwort ausschließlich im folgenden JSON-Format erfolgt:
{
  "summary": "...",
  "ratings": { "content": X, "expression": Y, "terminology": Z },
  "errorAnalysis": [
    { "original": "...", "interpretation": "...", "suggestion": "..." }
  ]
}
`;
      } else {
          prompt = `
Du bist ein erfahrener Dolmetschlehrer. Analysiere die folgende Dolmetschübung im Modus "${settings.mode}".
Thema: "${settings.topic}"
Ausgangssprache: ${settings.sourceLang}
Zielsprache: ${settings.targetLang}

Originaltext:
"${originalText}"

Verdolmetschung des Nutzers:
"${userTranscript}"

Gib Feedback in drei Teilen:
1.  **Zusammenfassung:** Eine kurze, konstruktive Gesamtbewertung.
2.  **Bewertungen:** Bewerte Inhalt, Ausdruck und Terminologie auf einer Skala von 1 bis 5 Sternen.
3.  **Fehleranalyse:** Liste bis zu 5 konkrete und wichtige Fehler auf. Für jeden Fehler: Gib den Originaltext-Ausschnitt an, die fehlerhafte Verdolmetschung des Nutzers und einen besseren Vorschlag. Erkläre kurz, warum der Vorschlag besser ist (z.B. "präziserer Begriff", "idiomatischer", "grammatikalisch korrekt"). Wenn es keine oder nur wenige Fehler gibt, erwähne das positiv.

Stelle sicher, dass die Antwort ausschließlich im folgenden JSON-Format erfolgt:
{
  "summary": "...",
  "ratings": { "content": X, "expression": Y, "terminology": Z },
  "errorAnalysis": [
    { "original": "...", "interpretation": "...", "suggestion": "..." }
  ]
}
`;
      }

      const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: { responseMimeType: "application/json" }
      });
      
      const parsedFeedback = JSON.parse(response.text || '{}');
      setFeedback(parsedFeedback);

    } catch (err) {
      console.error("Error getting feedback:", err);
      setError("Fehler bei der Feedback-Analyse. Bitte überprüfen Sie die Transkription und versuchen Sie es erneut.");
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <>
      <header className="app-header">
          <h1>Dolmetsch-Trainer Pro 2.0</h1>
          <p>KI-gestützte Trainingsumgebung für professionelle Dolmetscher</p>
      </header>
      <div className="main-container">
        <SettingsPanel
          settings={settings}
          onStartExercise={handleStartExercise}
          isExerciseInProgress={exerciseStarted}
          onBackToSettings={handleBackToSettings}
          isPremiumVoiceAvailable={isPremiumVoiceAvailable}
        />
        <PracticeArea
          key={exerciseId}
          settings={settings}
          isLoading={isLoading}
          loadingMessage={loadingMessage}
          error={error}
          originalText={originalText}
          userTranscript={userTranscript}
          onUserTranscriptChange={setUserTranscript}
          feedback={feedback}
          onGetFeedback={getFeedback}
          isExerciseStarted={exerciseStarted}
          dialogueFinished={dialogueFinished}
          structuredDialogueResults={structuredDialogueResults}
          onDialogueFinish={handleDialogueFinish}
          setStructuredDialogueResults={setStructuredDialogueResults}
          isPremiumVoiceAvailable={isPremiumVoiceAvailable}
          onPremiumVoiceAuthError={handlePremiumVoiceAuthError}
        />
      </div>
    </>
  );
};

// --- SETTINGS PANEL ---
interface SettingsPanelProps {
  settings: Settings;
  onStartExercise: (settings: Settings, text?: string) => void;
  isExerciseInProgress: boolean;
  onBackToSettings: () => void;
  isPremiumVoiceAvailable: boolean;
}

const SettingsPanel = ({ settings, onStartExercise, isExerciseInProgress, onBackToSettings, isPremiumVoiceAvailable }: SettingsPanelProps) => {
  const [currentSettings, setCurrentSettings] = useState<Settings>(settings);
  const [uploadedText, setUploadedText] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSettingChange = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setCurrentSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target?.result as string;
            setUploadedText(text);
            setFileName(file.name);
        };
        reader.readAsText(file, 'UTF-8');
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleSubmit = () => {
    if (currentSettings.sourceType === 'upload' && !uploadedText) {
        alert("Bitte laden Sie eine Textdatei hoch, bevor Sie die Übung starten.");
        return;
    }
    onStartExercise(currentSettings, uploadedText || undefined);
  };
  
  const showTopic = currentSettings.sourceType === 'ai';
  const showSpeechLength = ["Vortragsdolmetschen", "Simultandolmetschen", "Shadowing"].includes(currentSettings.mode);
  const showQALength = currentSettings.mode === 'Gesprächsdolmetschen';
  
  return (
    <div className="panel settings-panel">
        <div>
            {isExerciseInProgress ? (
                <>
                    <h2>Aktuelle Übung</h2>
                    <div className="form-group">
                        <p><strong>Modus:</strong> {currentSettings.mode}</p>
                        <p><strong>Ausgangssprache:</strong> {currentSettings.sourceLang}</p>
                        <p><strong>Zielsprache:</strong> {currentSettings.targetLang}</p>
                        {showTopic && <p><strong>Thema:</strong> {currentSettings.topic || 'Nicht angegeben'}</p>}
                    </div>
                </>
            ) : (
                <>
                <h2>Einstellungen</h2>
                 <div className="form-group">
                    <label htmlFor="mode">Modus</label>
                    <select id="mode" className="form-control" value={currentSettings.mode} onChange={e => handleSettingChange('mode', e.target.value as InterpretingMode)}>
                        {MODES.map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                </div>

                <div className="form-group">
                    <label htmlFor="sourceLang">Ausgangssprache</label>
                    <select id="sourceLang" className="form-control" value={currentSettings.sourceLang} onChange={e => handleSettingChange('sourceLang', e.target.value as Language)}>
                        {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
                    </select>
                </div>

                <div className="form-group">
                    <label htmlFor="targetLang">Zielsprache</label>
                    <select id="targetLang" className="form-control" value={currentSettings.targetLang} onChange={e => handleSettingChange('targetLang', e.target.value as Language)}>
                        {LANGUAGES.filter(l => l !== currentSettings.sourceLang).map(l => <option key={l} value={l}>{l}</option>)}
                    </select>
                </div>

                <div className="form-group">
                    <label htmlFor="sourceType">Textquelle</label>
                    <select id="sourceType" className="form-control" value={currentSettings.sourceType} onChange={e => {
                        const newSourceType = e.target.value as SourceTextType;
                        handleSettingChange('sourceType', newSourceType);
                        if (newSourceType === 'upload') {
                            // If switching to upload, make sure topic isn't required
                        }
                    }}>
                        <option value="ai">KI-generiert</option>
                        <option value="upload">Textdatei hochladen (.txt)</option>
                    </select>
                </div>
                
                {currentSettings.sourceType === 'upload' && (
                  <div className="form-group">
                    <div className="upload-group">
                      <button type="button" className="btn btn-secondary" onClick={handleUploadClick}>Datei wählen</button>
                      <span className="file-name">{fileName || 'Keine Datei ausgewählt'}</span>
                    </div>
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".txt" style={{ display: 'none' }} />
                  </div>
                )}


                {showTopic && (
                    <div className="form-group">
                        <label htmlFor="topic">Thema</label>
                        <input type="text" id="topic" className="form-control" value={currentSettings.topic} onChange={e => handleSettingChange('topic', e.target.value)} placeholder="z.B. Klimawandel, Technologie, Kunst" />
                    </div>
                )}

                {showSpeechLength && (
                    <div className="form-group">
                        <label htmlFor="speechLength">Länge des Vortrags</label>
                        <select id="speechLength" className="form-control" value={currentSettings.speechLength} onChange={e => handleSettingChange('speechLength', e.target.value as SpeechLength)}>
                           {SPEECH_LENGTHS.map(l => <option key={l} value={l}>{l}</option>)}
                        </select>
                    </div>
                )}
                
                {showQALength && (
                     <div className="form-group">
                        <label htmlFor="qaLength">Länge pro Redebeitrag</label>
                        <select id="qaLength" className="form-control" value={currentSettings.qaLength} onChange={e => handleSettingChange('qaLength', e.target.value as QALength)}>
                           {QA_LENGTHS.map(l => <option key={l} value={l}>{l}</option>)}
                        </select>
                    </div>
                )}

                {isPremiumVoiceAvailable && ["Vortragsdolmetschen", "Gesprächsdolmetschen"].includes(currentSettings.mode) && (
                  <div className="form-group">
                    <label>Stimmqualität</label>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                      <label><input type="radio" name="voiceQuality" value="Standard" checked={currentSettings.voiceQuality === 'Standard'} onChange={(e) => handleSettingChange('voiceQuality', e.target.value as VoiceQuality)} /> Standard (Browser)</label>
                      <label><input type="radio" name="voiceQuality" value="Premium" checked={currentSettings.voiceQuality === 'Premium'} onChange={(e) => handleSettingChange('voiceQuality', e.target.value as VoiceQuality)} /> Premium (Google Cloud)</label>
                    </div>
                    <p className="form-text-hint">Premium bietet natürlichere Stimmen, erfordert aber einen korrekt konfigurierten Google Cloud API-Schlüssel.</p>
                  </div>
                )}
                </>
            )}
        </div>
        <div className="settings-footer">
            {isExerciseInProgress ? (
                <button className="btn btn-secondary" onClick={onBackToSettings}>Übung beenden & zurück</button>
            ) : (
                <button className="btn btn-primary" onClick={handleSubmit} disabled={showTopic && !currentSettings.topic}>
                    Übung starten
                </button>
            )}
        </div>
    </div>
  );
};


// --- PRACTICE AREA ---
interface PracticeAreaProps {
  settings: Settings;
  isLoading: boolean;
  loadingMessage: string;
  error: string | null;
  originalText: string;
  userTranscript: string;
  onUserTranscriptChange: (transcript: string) => void;
  feedback: Feedback | null;
  onGetFeedback: () => void;
  isExerciseStarted: boolean;
  dialogueFinished: boolean;
  structuredDialogueResults: StructuredDialogueResult[] | null;
  onDialogueFinish: (results: StructuredDialogueResult[]) => void;
  setStructuredDialogueResults: (results: StructuredDialogueResult[] | null) => void;
  isPremiumVoiceAvailable: boolean;
  onPremiumVoiceAuthError: () => void;
}


const PracticeArea = ({
  settings,
  isLoading,
  loadingMessage,
  error,
  originalText,
  userTranscript,
  onUserTranscriptChange,
  feedback,
  onGetFeedback,
  isExerciseStarted,
  dialogueFinished,
  structuredDialogueResults,
  onDialogueFinish,
  setStructuredDialogueResults,
  isPremiumVoiceAvailable,
  onPremiumVoiceAuthError,
}: PracticeAreaProps) => {
  const [activeTab, setActiveTab] = useState<PracticeAreaTab>('original');
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const prevIsExerciseStarted = usePrevious(isExerciseStarted);
  const prevOriginalText = usePrevious(originalText);

  useEffect(() => {
    // When a new exercise starts, reset the tabs and feedback
    if (isExerciseStarted && !prevIsExerciseStarted) {
      setActiveTab('original');
    }
    // if original text has been populated, reset feedback.
    if(originalText && originalText !== prevOriginalText) {
        onUserTranscriptChange('');
    }
  }, [isExerciseStarted, prevIsExerciseStarted, originalText, prevOriginalText, onUserTranscriptChange]);

  const handleStartRecording = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech Recognition wird von diesem Browser nicht unterstützt.");
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    const recognition = recognitionRef.current;
    recognition.lang = LANGUAGE_CODES[settings.targetLang];
    recognition.continuous = true;
    recognition.interimResults = true;

    let finalTranscript = userTranscript ? userTranscript + ' ' : '';

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interimTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript + ' ';
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }
      onUserTranscriptChange(finalTranscript + interimTranscript);
    };
    
    recognition.onstart = () => {
        setIsRecording(true);
    };

    recognition.onend = () => {
      setIsRecording(false);
    };
    
    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error("Speech Recognition Error", event.error);
        setIsRecording(false);
    }

    recognition.start();
  };

  const handleStopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  };

  const handleRecordClick = () => {
    if (isRecording) {
      handleStopRecording();
    } else {
      handleStartRecording();
    }
  };

  const renderContent = () => {
    if (isLoading && !isExerciseStarted) return null; // Don't show overlay if it's just settings change
    
    if (!isExerciseStarted) {
      return (
        <div className="placeholder">
          <h2>Willkommen beim Dolmetsch-Trainer Pro</h2>
          <p>Passen Sie Ihre Einstellungen an und starten Sie eine Übung.</p>
        </div>
      );
    }

    if (!originalText) {
      return (
        <div className="placeholder">
          <h2>Übung wird vorbereitet...</h2>
          <p>Bitte haben Sie einen Moment Geduld.</p>
        </div>
      );
    }

    if (settings.mode === 'Gesprächsdolmetschen') {
      return (
        <DialoguePractice
          originalText={originalText}
          settings={settings}
          onFinish={onDialogueFinish}
          onUpdateResults={setStructuredDialogueResults}
          isPremiumVoiceAvailable={isPremiumVoiceAvailable}
          onAuthError={onPremiumVoiceAuthError}
        />
      );
    }
    
    const showTabs = isExerciseStarted && originalText;
    const showFeedbackButton = userTranscript.length > 0 && !feedback;
    
    return (
      <>
        {showTabs && (
          <div className="tabs">
            <button className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`} onClick={() => setActiveTab('original')}>Originaltext</button>
            <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')} disabled={!userTranscript && !isRecording}>Ihre Verdolmetschung</button>
            <button className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} onClick={() => setActiveTab('feedback')} disabled={!feedback}>Feedback</button>
          </div>
        )}
        <div className="tab-content">
          {activeTab === 'original' && (
            <>
                <SpeechPlayer text={originalText} lang={settings.sourceLang} mode={settings.mode} isPremiumVoiceAvailable={isPremiumVoiceAvailable} onAuthError={onPremiumVoiceAuthError} />
                <div className="text-area">
                    <p>{originalText}</p>
                </div>
            </>
          )}
          {activeTab === 'transcript' && (
             <div className="text-area">
                <textarea 
                    className="text-area-editor"
                    value={userTranscript}
                    onChange={(e) => onUserTranscriptChange(e.target.value)}
                    placeholder="Hier erscheint Ihre Verdolmetschung..."
                />
            </div>
          )}
          {activeTab === 'feedback' && feedback && <FeedbackDisplay feedback={feedback} />}
        </div>

        <div className="practice-footer">
          <p className="recording-status-text">
            {isRecording ? "Aufnahme läuft..." : (userTranscript ? "Aufnahme beendet." : "Bereit zur Aufnahme.")}
          </p>
          <button className={`btn-record ${isRecording ? 'recording' : ''}`} onClick={handleRecordClick} aria-label={isRecording ? 'Aufnahme stoppen' : 'Aufnahme starten'}>
            <div className="mic-icon"></div>
          </button>
          {showFeedbackButton && <button className="btn btn-primary" onClick={onGetFeedback} style={{marginTop: '1rem'}}>Feedback erhalten</button>}
        </div>
      </>
    );
  };
  
  const renderDialogueContent = () => {
    if (!isExerciseStarted || settings.mode !== 'Gesprächsdolmetschen') return null;
    
    if (dialogueFinished && structuredDialogueResults) {
       const showFeedbackButton = !feedback;
       return (
         <>
          <div className="tabs">
            <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>Gesprächsverlauf</button>
            <button className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} onClick={() => setActiveTab('feedback')} disabled={!feedback}>Feedback</button>
          </div>
          <div className="tab-content">
             {activeTab === 'transcript' && <StructuredTranscriptDisplay results={structuredDialogueResults} />}
             {activeTab === 'feedback' && feedback && <FeedbackDisplay feedback={feedback} />}
          </div>
          <div className="practice-footer">
            {showFeedbackButton && <button className="btn btn-primary" onClick={onGetFeedback} style={{marginTop: '1rem'}}>Feedback für Gespräch erhalten</button>}
          </div>
         </>
       )
    }
    
    return renderContent();
  }

  return (
    <div className="panel practice-area">
        {isLoading && (
            <div className="loading-overlay">
                <div className="spinner"></div>
                <p>{loadingMessage}</p>
            </div>
        )}
        {error && <div className="error-banner">{error}</div>}
        {settings.mode === 'Gesprächsdolmetschen' ? renderDialogueContent() : renderContent()}
    </div>
  );
};


// --- SPEECH PLAYER ---
interface SpeechPlayerProps {
    text: string;
    lang: Language;
    mode: InterpretingMode;
    isPremiumVoiceAvailable: boolean;
    onAuthError: () => void;
}

const SpeechPlayer = ({ text, lang, mode, isPremiumVoiceAvailable, onAuthError }: SpeechPlayerProps) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    const usePremium = isPremiumVoiceAvailable && ["Vortragsdolmetschen", "Gesprächsdolmetschen"].includes(mode);

    useEffect(() => {
        // Cleanup speech synthesis on component unmount or when text changes
        return () => {
            if (speechSynthesis.speaking) {
                speechSynthesis.cancel();
            }
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current = null;
            }
        };
    }, [text]);

    const handlePlayPause = async () => {
        if (isLoading) return;

        if (usePremium) {
            if (audioRef.current) {
                if (isPlaying) {
                    audioRef.current.pause();
                    setIsPlaying(false);
                } else {
                    audioRef.current.play();
                    setIsPlaying(true);
                }
            } else {
                setIsLoading(true);
                try {
                    const audioContent = await synthesizeSpeechGoogleCloud(text, lang, process.env.API_KEY);
                    const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioRef.current = new Audio(audioUrl);
                    audioRef.current.play();
                    audioRef.current.onended = () => setIsPlaying(false);
                    setIsPlaying(true);
                } catch (error) {
                    console.error("Error synthesizing premium speech:", error);
                    if (error instanceof TtsAuthError) {
                        onAuthError();
                    }
                    // Fallback or show error
                } finally {
                    setIsLoading(false);
                }
            }
        } else {
            // Standard browser synthesis
            if (isPlaying) {
                speechSynthesis.pause();
                setIsPaused(true);
                setIsPlaying(false);
            } else {
                if (isPaused) {
                    speechSynthesis.resume();
                } else {
                    if (speechSynthesis.speaking) {
                        speechSynthesis.cancel();
                    }
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.lang = LANGUAGE_CODES[lang];
                    utterance.onend = () => {
                        setIsPlaying(false);
                        setIsPaused(false);
                        utteranceRef.current = null;
                    };
                    utteranceRef.current = utterance;
                    speechSynthesis.speak(utterance);
                }
                setIsPlaying(true);
            }
        }
    };
    
    // Effect to get voices and attach listener
    useEffect(() => {
        const handleVoicesChanged = () => {
            // The voices are now loaded
        };
        speechSynthesis.addEventListener('voiceschanged', handleVoicesChanged);
        // Initial check in case voices are already loaded
        if (speechSynthesis.getVoices().length > 0) {
            // voices loaded
        }
        return () => {
            speechSynthesis.removeEventListener('voiceschanged', handleVoicesChanged);
            speechSynthesis.cancel(); // Clean up on unmount
        };
    }, []);


    return (
        <div className="controls-bar">
            <button onClick={handlePlayPause} className="btn-play-pause" disabled={isLoading}>
                {isLoading ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
                ) : isPlaying ? (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                )}
            </button>
            <p>
                {mode === 'Stegreifübersetzen' ? "Text zur Ansicht bereit" : "Rede zur Wiedergabe bereit"}
            </p>
        </div>
    );
};


// --- DIALOGUE PRACTICE ---
interface DialoguePracticeProps {
    originalText: string;
    settings: Settings;
    onFinish: (results: StructuredDialogueResult[]) => void;
    onUpdateResults: (results: StructuredDialogueResult[]) => void;
    isPremiumVoiceAvailable: boolean;
    onAuthError: () => void;
}

const DialoguePractice = ({ originalText, settings, onFinish, onUpdateResults, isPremiumVoiceAvailable, onAuthError }: DialoguePracticeProps) => {
  const [segments, setSegments] = useState<DialogueSegment[]>([]);
  const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);
  const [dialogueState, setDialogueState] = useState<DialogueState>('idle');
  const [results, setResults] = useState<StructuredDialogueResult[]>([]);
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [isCurrentTextVisible, setIsCurrentTextVisible] = useState(false);
  
  const recognition = useRef<SpeechRecognition | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // FIX: Create a ref to hold the current dialogue state to avoid stale closures in callbacks.
  const dialogueStateRef = useRef(dialogueState);
  useEffect(() => {
    dialogueStateRef.current = dialogueState;
  }, [dialogueState]);

  useEffect(() => {
    const lines = originalText.trim().split('\n').filter(line => line.length > 0);
    const parsedSegments: DialogueSegment[] = lines.map(line => {
        const isQuestion = line.startsWith('Frage');
        const text = line.replace(/^(Frage \d:|Antwort \d:)\s*/, '');
        return {
            type: isQuestion ? 'Frage' : 'Antwort',
            text,
            lang: isQuestion ? settings.sourceLang : settings.targetLang,
        };
    });
    setSegments(parsedSegments);
    setDialogueState('ready');
  }, [originalText, settings.sourceLang, settings.targetLang]);
  
  useEffect(() => {
    if (dialogueState === 'ready' && segments.length > 0) {
        setDialogueState('starting');
    }
  }, [dialogueState, segments]);

  useEffect(() => {
    const playCurrentSegment = async () => {
        if (dialogueState === 'starting' && segments.length > 0) {
            const segment = segments[currentSegmentIndex];
            if (!segment) return;

            setDialogueState('synthesizing');
            
            try {
                let audioSrc = '';
                if (settings.voiceQuality === 'Premium' && isPremiumVoiceAvailable) {
                    try {
                        const audioContent = await synthesizeSpeechGoogleCloud(segment.text, segment.lang, process.env.API_KEY);
                        const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
                        audioSrc = URL.createObjectURL(audioBlob);
                    } catch (err) {
                        if (err instanceof TtsAuthError) {
                            onAuthError();
                        }
                        // Fallback to standard voice on error
                        const utterance = new SpeechSynthesisUtterance(segment.text);
                        utterance.lang = LANGUAGE_CODES[segment.lang];
                        speechSynthesis.speak(utterance);
                        setDialogueState('playing');
                        utterance.onend = () => setDialogueState('waiting_for_record');
                        return;
                    }
                }

                if (audioSrc) {
                    audioRef.current = new Audio(audioSrc);
                    setDialogueState('playing');
                    audioRef.current.play();
                    audioRef.current.onended = () => setDialogueState('waiting_for_record');
                } else {
                    // Standard voice
                    const utterance = new SpeechSynthesisUtterance(segment.text);
                    utterance.lang = LANGUAGE_CODES[segment.lang];
                    speechSynthesis.speak(utterance);
                    setDialogueState('playing');
                    utterance.onend = () => setDialogueState('waiting_for_record');
                }
            } catch (err) {
                console.error("Error during speech synthesis:", err);
                setDialogueState('waiting_for_record'); // Move on even if TTS fails
            }
        }
    };
    playCurrentSegment();

    return () => {
        if (speechSynthesis.speaking) speechSynthesis.cancel();
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current = null;
        }
    };
    // Fix: Removed non-existent property `settings.lang` from the dependency array.
}, [dialogueState, currentSegmentIndex, segments, isPremiumVoiceAvailable, onAuthError, settings.voiceQuality, settings.sourceLang, settings.targetLang]);

  const handleNextSegment = useCallback(() => {
    const interpretationLang = segments[currentSegmentIndex].type === 'Frage' ? settings.targetLang : settings.sourceLang;
    const newResult: StructuredDialogueResult = {
        originalSegment: segments[currentSegmentIndex],
        userInterpretation: currentTranscript.trim(),
        interpretationLang: interpretationLang
    };
    const updatedResults = [...results, newResult];
    setResults(updatedResults);
    onUpdateResults(updatedResults);
    setCurrentTranscript('');
    setIsCurrentTextVisible(false);

    if (currentSegmentIndex < segments.length - 1) {
        setCurrentSegmentIndex(prev => prev + 1);
        setDialogueState('starting');
    } else {
        setDialogueState('finished');
        onFinish(updatedResults);
    }
  }, [currentSegmentIndex, segments, onFinish, results, onUpdateResults, currentTranscript, settings.targetLang, settings.sourceLang]);

  useEffect(() => {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) {
      setDialogueState('finished');
      console.error("Speech Recognition not supported in this browser.");
      return;
    }

    if (dialogueState === 'recording') {
      recognition.current = new SpeechRecognitionAPI();
      const rec = recognition.current;
      const interpretationLang = segments[currentSegmentIndex].type === 'Frage' ? settings.targetLang : settings.sourceLang;
      rec.lang = LANGUAGE_CODES[interpretationLang];
      rec.continuous = true;
      rec.interimResults = true;
      
      let finalTranscriptForSegment = ''; // Closure variable to accumulate final transcript

      rec.onresult = (event: SpeechRecognitionEvent) => {
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscriptForSegment += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        setCurrentTranscript(finalTranscriptForSegment + interimTranscript);
      };

      rec.onend = () => {
        // FIX: Use the state ref to check the most up-to-date state.
        if (dialogueStateRef.current === 'recording') {
          handleNextSegment();
        }
      };
      
      rec.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error('Speech recognition error', event.error);
        handleNextSegment();
      };
      
      rec.start();
    } else if (recognition.current) {
      recognition.current.stop();
    }

    return () => {
      if (recognition.current) {
        recognition.current.onresult = () => {};
        recognition.current.onend = () => {};
        recognition.current.onerror = () => {};
        recognition.current.stop();
        recognition.current = null;
      }
    };
  }, [dialogueState, handleNextSegment, settings.targetLang, settings.sourceLang, segments, currentSegmentIndex]);


  const handleRecordClick = () => {
    if (audioRef.current && !audioRef.current.paused) {
      audioRef.current.pause();
    }
    
    if (dialogueState === 'recording') {
      setDialogueState('waiting_for_record');
    } else if (dialogueState === 'waiting_for_record') {
      setCurrentTranscript('');
      setDialogueState('recording');
    }
  };
  
  const getStatusText = () => {
    switch(dialogueState) {
        case 'idle':
        case 'ready':
            return "Übung wird geladen...";
        case 'synthesizing':
            return "Audio wird vorbereitet...";
        case 'playing':
            return `Teilnehmer ${segments[currentSegmentIndex]?.type === 'Frage' ? 'A' : 'B'} spricht...`;
        case 'waiting_for_record':
            return "Sie sind dran. Drücken Sie den Aufnahme-Button.";
        case 'recording':
            return "Ihre Verdolmetschung wird aufgenommen...";
        case 'finished':
            return "Gespräch beendet.";
        case 'starting':
             return `Beginne Segment ${currentSegmentIndex + 1}/${segments.length}...`;
        default:
            return "Bereit";
    }
  };
  
  const currentSegment = segments[currentSegmentIndex];

  return (
    <div className="dialogue-practice-container">
        <div className="dialogue-status">
            {getStatusText()}
        </div>
        <div className="current-segment-display">
            {currentSegment && (
              <div className="dialogue-text-container">
                {isCurrentTextVisible ? (
                  <p className="segment-text">{currentSegment.text}</p>
                ) : (
                  <p className="segment-text-hidden">
                    {`Der Text für ${currentSegment.type} ${currentSegmentIndex + 1} (${currentSegment.lang}) ist ausgeblendet.`}
                  </p>
                )}
                <button className="btn btn-secondary btn-show-text" onClick={() => setIsCurrentTextVisible(!isCurrentTextVisible)}>
                  {isCurrentTextVisible ? "Text ausblenden" : "Text anzeigen"}
                </button>
              </div>
            )}
        </div>
        <div className="practice-footer">
          <p className="recording-status-text">
            {dialogueState === 'recording' ? currentTranscript : ''}
          </p>
          <button 
            className={`btn-record ${dialogueState === 'recording' ? 'recording' : ''}`} 
            onClick={handleRecordClick} 
            disabled={!['waiting_for_record', 'recording'].includes(dialogueState)}
            aria-label={dialogueState === 'recording' ? 'Aufnahme stoppen' : 'Aufnahme starten'}
          >
            <div className="mic-icon"></div>
          </button>
        </div>
    </div>
  );
};


// --- FEEDBACK & RESULTS DISPLAYS ---
const FeedbackDisplay = ({ feedback }: { feedback: Feedback }) => {
  const renderStars = (rating: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <span key={i} className={`star ${i < rating ? 'filled' : ''}`}>&#9733;</span>
    ));
  };

  return (
    <div className="feedback-content">
      <h3>Zusammenfassung</h3>
      <p>{feedback.summary}</p>

      <h3>Detaillierte Bewertung</h3>
      <table className="ratings-table">
        <tbody>
          <tr>
            <td>Inhaltliche Korrektheit</td>
            <td>{renderStars(feedback.ratings.content)}</td>
          </tr>
          <tr>
            <td>Stil & Ausdruck</td>
            <td>{renderStars(feedback.ratings.expression)}</td>
          </tr>
          <tr>
            <td>Terminologie</td>
            <td>{renderStars(feedback.ratings.terminology)}</td>
          </tr>
        </tbody>
      </table>

      <h3>Fehleranalyse & Vorschläge</h3>
      {feedback.errorAnalysis && feedback.errorAnalysis.length > 0 ? (
        <ul className="error-analysis-list">
          {feedback.errorAnalysis.map((item, index) => (
            <li key={index}>
              <p><strong>Original:</strong> {item.original}</p>
              <p><strong>Ihre Version:</strong> {item.interpretation}</p>
              <p><strong>Vorschlag:</strong> {item.suggestion}</p>
            </li>
          ))}
        </ul>
      ) : (
        <p>Sehr gute Arbeit! Es wurden keine wesentlichen Fehler gefunden.</p>
      )}
    </div>
  );
};

const StructuredTranscriptDisplay = ({ results }: { results: StructuredDialogueResult[] }) => (
    <div className="dialogue-results-wrapper text-area">
        <div className="structured-transcript">
            {results.map((result, index) => (
                <div key={index} className="transcript-segment">
                    <div className="transcript-segment-header">
                        <h4>{`${result.originalSegment.type} ${index + 1} (${result.originalSegment.lang})`}</h4>
                    </div>
                    <p className="transcript-segment-original">
                        {result.originalSegment.text}
                    </p>
                    <p className="transcript-segment-user">
                        {result.userInterpretation ? result.userInterpretation : <em>Keine Verdolmetschung aufgenommen.</em>}
                    </p>
                </div>
            ))}
        </div>
    </div>
);


const container = document.getElementById('root');
if(container) {
    const root = createRoot(container);
    root.render(<App />);
}