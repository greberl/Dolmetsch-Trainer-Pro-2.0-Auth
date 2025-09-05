import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
// Fix: Import `Type` to be used for defining a response schema.
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
type DialogueState = 'synthesizing' | 'playing' | 'waiting_for_record' | 'recording' | 'finished';
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
    explanation: string;
    type: string;
}

interface Feedback {
    clarity: number;
    accuracy: number;
    completeness: number;
    style: number;
    terminology: number;
    overall: number;
    summary: string;
    errorAnalysis: ErrorAnalysisItem[];
}

// --- CONSTANTS & UTILS ---
const LANGUAGES: Language[] = ["Deutsch", "Englisch", "Russisch", "Spanisch", "Französisch"];
const MODES: InterpretingMode[] = ["Vortragsdolmetschen", "Simultandolmetschen", "Shadowing", "Gesprächsdolmetschen", "Stegreifübersetzen"];
const QA_LENGTHS: QALength[] = ["1-3 Sätze", "2-4 Sätze", "3-5 Sätze", "4-6 Sätze"];
const SPEECH_LENGTHS: SpeechLength[] = ["Kurz", "Mittel", "Prüfung"];

const LANG_MAP: Record<Language, string> = {
  "Deutsch": "de-DE",
  "Englisch": "en-US",
  "Russisch": "ru-RU",
  "Spanisch": "es-ES",
  "Französisch": "fr-FR"
};
const VOICE_MAP: Record<Language, Record<VoiceQuality, string>> = {
    "Deutsch": { "Standard": "de-DE-Standard-A", "Premium": "de-DE-Wavenet-F" },
    "Englisch": { "Standard": "en-US-Standard-C", "Premium": "en-US-Wavenet-F" },
    "Russisch": { "Standard": "ru-RU-Standard-A", "Premium": "ru-RU-Wavenet-D" },
    "Spanisch": { "Standard": "es-ES-Standard-A", "Premium": "es-ES-Wavenet-B" },
    "Französisch": { "Standard": "fr-FR-Standard-A", "Premium": "fr-FR-Wavenet-E" }
};
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// --- API HELPERS ---
const generateContentWithRetry = async (prompt: string, retries = 3, delay = 1000) => {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt,
            });
            return response.text;
        } catch (error) {
            console.error(`Attempt ${i + 1} failed:`, error);
            if (i === retries - 1) throw error;
            await new Promise(res => setTimeout(res, delay));
        }
    }
    throw new Error("Failed to generate content after multiple retries.");
};

const synthesizeSpeechGoogleCloud = async (text: string, lang: Language, quality: VoiceQuality) => {
    const languageCode = LANG_MAP[lang];
    const voiceName = VOICE_MAP[lang][quality];
    const url = `https://texttospeech.googleapis.com/v1/text:synthesize?key=${process.env.API_KEY}`;

    const body = JSON.stringify({
        input: { text: text },
        voice: { languageCode: languageCode, name: voiceName },
        audioConfig: { audioEncoding: 'MP3' },
    });
    try {
        const response = await fetch(url, { method: 'POST', body });
        if (!response.ok) {
            const errorText = await response.text();
            console.error("Speech synthesis failed:", response.status, errorText);
            throw new Error(`Speech synthesis failed: ${response.status} ${errorText}`);
        }
        const data = await response.json();
        return data.audioContent;
    } catch (error) {
        console.error("Error calling Google Cloud TTS API:", error);
        throw error;
    }
};

// --- REACT COMPONENTS ---
const App = () => {
  const [settings, setSettings] = useState<Settings>({
    mode: "Vortragsdolmetschen",
    sourceLang: "Deutsch",
    targetLang: "Englisch",
    sourceType: "ai",
    topic: "Erneuerbare Energien",
    qaLength: "2-4 Sätze",
    speechLength: "Kurz",
    voiceQuality: "Premium",
  });
  const [exerciseState, setExerciseState] = useState<'idle' | 'generating' | 'ready' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [originalText, setOriginalText] = useState<string>('');
  const [exerciseId, setExerciseId] = useState<number>(Date.now());
  const [dialogue, setDialogue] = useState<DialogueSegment[]>([]);

  const handleStart = async () => {
    setExerciseState('generating');
    setErrorMessage('');
    try {
      let prompt = '';
      if (settings.mode === 'Gesprächsdolmetschen' || settings.mode === 'Stegreifübersetzen') {
          prompt = `Erstelle einen realistischen Dialog zwischen zwei Personen (A und B) zum Thema "${settings.topic}". Der Dialog soll im Frage-Antwort-Format sein. Person A (${settings.sourceLang}) stellt Fragen, Person B (${settings.targetLang}) antwortet. Der Dialog soll insgesamt 4 Segmente haben (A-Frage, B-Antwort, A-Frage, B-Antwort). Jedes Segment soll eine Länge von "${settings.qaLength}" haben. Gib nur den reinen Dialog aus, ohne zusätzliche Erklärungen, formatiert als JSON-Array mit Objekten, die "type", "text" und "lang" enthalten. Beispiel: [{"type": "Frage", "text": "...", "lang": "${settings.sourceLang}"}, ...]`;
      } else {
          prompt = `Erstelle einen Text zum Thema "${settings.topic}" für eine Dolmetschübung im Modus "${settings.mode}". Die Sprache des Textes soll ${settings.sourceLang} sein. Die Länge soll dem Level "${settings.speechLength}" entsprechen. Gib nur den reinen Text aus, ohne Titel oder zusätzliche Kommentare.`;
      }
      const text = await generateContentWithRetry(prompt);

      if (settings.mode === 'Gesprächsdolmetschen' || settings.mode === 'Stegreifübersetzen') {
          const parsedDialogue = JSON.parse(text);
          setDialogue(parsedDialogue);
      } else {
          setOriginalText(text);
      }
      setExerciseState('ready');
      setExerciseId(Date.now()); // Reset exercise with new ID
    } catch (error) {
      console.error(error);
      setErrorMessage("Fehler bei der Erstellung der Übung. Bitte versuchen Sie es erneut.");
      setExerciseState('error');
    }
  };

  const handleFileUpload = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result;
      if (typeof result === 'string') {
        setOriginalText(result);
        setExerciseState('ready');
        setExerciseId(Date.now()); // Reset exercise with new ID
      } else {
        console.error("Failed to read file as text.");
        setErrorMessage("Datei konnte nicht als Text gelesen werden.");
        setExerciseState('error');
      }
    };
    reader.readAsText(file);
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
          setSettings={setSettings}
          onStart={handleStart}
          onFileUpload={handleFileUpload}
          isLoading={exerciseState === 'generating'}
        />
        <PracticeArea
          key={exerciseId}
          settings={settings}
          exerciseState={exerciseState}
          originalText={originalText}
          dialogue={dialogue}
          errorMessage={errorMessage}
        />
      </div>
    </>
  );
};

const SettingsPanel = ({ settings, setSettings, onStart, onFileUpload, isLoading }: {
  settings: Settings;
  setSettings: (settings: Settings) => void;
  onStart: () => void;
  onFileUpload: (file: File) => void;
  isLoading: boolean;
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState('');

  const handleSettingChange = (field: keyof Settings, value: string) => {
    // Basic validation to prevent incompatible language settings
    if (field === 'sourceLang' && value === settings.targetLang) {
      // Find a new target language that is not the selected source language
      const newTargetLang = LANGUAGES.find(l => l !== value) || LANGUAGES[1];
      setSettings({ ...settings, [field]: value, targetLang: newTargetLang });
    } else if (field === 'targetLang' && value === settings.sourceLang) {
       // Find a new source language that is not the selected target language
      const newSourceLang = LANGUAGES.find(l => l !== value) || LANGUAGES[0];
      setSettings({ ...settings, [field]: value, sourceLang: newSourceLang });
    } else {
      setSettings({ ...settings, [field]: value });
    }
  };

  const handleFileSelect = () => {
    fileInputRef.current?.click();
  };
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setFileName(file.name);
      onFileUpload(file);
    }
  };

  const renderSourceTypeOptions = () => {
    if (settings.mode === "Gesprächsdolmetschen" || settings.mode === "Stegreifübersetzen") {
        return null; // Hide source type for these modes as they always use AI
    }
    return (
        <div className="form-group">
            <label htmlFor="sourceType">Quelle</label>
            <select id="sourceType" className="form-control" value={settings.sourceType} onChange={e => handleSettingChange('sourceType', e.target.value)}>
                <option value="ai">KI-generiert</option>
                <option value="upload">Text hochladen</option>
            </select>
        </div>
    );
  };

  const renderAiOptions = () => {
      if (settings.sourceType === 'upload' && settings.mode !== 'Gesprächsdolmetschen' && settings.mode !== 'Stegreifübersetzen') return null;
      return (
          <>
              <div className="form-group">
                  <label htmlFor="topic">Thema</label>
                  <input type="text" id="topic" className="form-control" value={settings.topic} onChange={e => handleSettingChange('topic', e.target.value)} />
              </div>
              {settings.mode === 'Gesprächsdolmetschen' || settings.mode === 'Stegreifübersetzen' ? (
                  <div className="form-group">
                      <label htmlFor="qaLength">Satzlänge</label>
                      <select id="qaLength" className="form-control" value={settings.qaLength} onChange={e => handleSettingChange('qaLength', e.target.value)}>
                          {QA_LENGTHS.map(len => <option key={len} value={len}>{len}</option>)}
                      </select>
                  </div>
              ) : (
                  <div className="form-group">
                      <label htmlFor="speechLength">Redelänge</label>
                      <select id="speechLength" className="form-control" value={settings.speechLength} onChange={e => handleSettingChange('speechLength', e.target.value)}>
                          {SPEECH_LENGTHS.map(len => <option key={len} value={len}>{len}</option>)}
                      </select>
                  </div>
              )}
          </>
      );
  };

  return (
    <div className="panel settings-panel">
      <div>
        <h2>Einstellungen</h2>
        <div className="form-group">
          <label htmlFor="mode">Modus</label>
          <select id="mode" className="form-control" value={settings.mode} onChange={e => handleSettingChange('mode', e.target.value)}>
            {MODES.map(mode => <option key={mode} value={mode}>{mode}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="sourceLang">Ausgangssprache</label>
          <select id="sourceLang" className="form-control" value={settings.sourceLang} onChange={e => handleSettingChange('sourceLang', e.target.value)}>
            {LANGUAGES.map(lang => <option key={lang} value={lang} disabled={lang === settings.targetLang}>{lang}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="targetLang">Zielsprache</label>
          <select id="targetLang" className="form-control" value={settings.targetLang} onChange={e => handleSettingChange('targetLang', e.target.value)}>
            {LANGUAGES.map(lang => <option key={lang} value={lang} disabled={lang === settings.sourceLang}>{lang}</option>)}
          </select>
        </div>

        {renderSourceTypeOptions()}

        {settings.sourceType === 'upload' && settings.mode !== 'Gesprächsdolmetschen' && settings.mode !== 'Stegreifübersetzen' && (
             <div className="form-group">
                <label>Datei hochladen</label>
                <div className="upload-group">
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".txt" style={{ display: 'none' }} />
                    <button className="btn btn-secondary" onClick={handleFileSelect}>Datei wählen</button>
                    {fileName && <span className="file-name">{fileName}</span>}
                </div>
            </div>
        )}

        {renderAiOptions()}

        <div className="form-group">
             <label htmlFor="voiceQuality">Stimmqualität</label>
             <select id="voiceQuality" className="form-control" value={settings.voiceQuality} onChange={e => handleSettingChange('voiceQuality', e.target.value)}>
                 <option value="Standard">Standard</option>
                 <option value="Premium">Premium</option>
             </select>
             <p className="form-text-hint">Premium-Stimmen (Wavenet) bieten eine höhere Qualität, können aber zusätzliche Kosten verursachen.</p>
        </div>
      </div>

      <div className="settings-footer">
          <button className="btn btn-primary btn-large" onClick={onStart} disabled={isLoading}>
            {isLoading ? 'Wird erstellt...' : 'Übung starten'}
          </button>
      </div>
    </div>
  );
};

const PracticeArea = ({ settings, exerciseState, originalText, dialogue, errorMessage }: {
  settings: Settings;
  exerciseState: 'idle' | 'generating' | 'ready' | 'error';
  originalText: string;
  dialogue: DialogueSegment[];
  errorMessage: string;
}) => {
    if (exerciseState === 'idle' || exerciseState === 'generating') {
        return (
            <div className="panel practice-area">
                <div className="placeholder">
                    <h2>Ihre Übung erscheint hier</h2>
                    <p>Konfigurieren Sie Ihre Einstellungen und klicken Sie auf "Übung starten".</p>
                </div>
                {exerciseState === 'generating' && (
                    <div className="loading-overlay">
                        <div className="spinner"></div>
                        <p>KI-Magie im Gange... Ihre Übung wird vorbereitet.</p>
                    </div>
                )}
            </div>
        );
    }
     if (exerciseState === 'error') {
        return (
            <div className="panel practice-area">
                <div className="error-banner">{errorMessage}</div>
            </div>
        );
    }
    // Render specific practice component based on mode
    switch (settings.mode) {
        case "Gesprächsdolmetschen":
        case "Stegreifübersetzen":
            return <DialoguePractice settings={settings} dialogue={dialogue} />;
        case "Vortragsdolmetschen":
        case "Simultandolmetschen":
        case "Shadowing":
            return <MonologuePractice settings={settings} originalText={originalText} />;
        default:
            return <div className="panel practice-area"><p>Modus nicht gefunden.</p></div>;
    }
};

const MonologuePractice = ({ settings, originalText }: {
    settings: Settings;
    originalText: string;
}) => {
    const [activeTab, setActiveTab] = useState<PracticeAreaTab>('original');
    const [transcript, setTranscript] = useState<string>('');
    const [isRecording, setIsRecording] = useState<boolean>(false);
    const [isAudioPlaying, setIsAudioPlaying] = useState<boolean>(false);
    const [feedback, setFeedback] = useState<Feedback | null>(null);
    const [isLoadingFeedback, setIsLoadingFeedback] = useState<boolean>(false);
    const [editedTranscript, setEditedTranscript] = useState<string>('');

    const recognitionRef = useRef<SpeechRecognition | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const synthesisInitiated = useRef(false);

    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = true;
            recognitionRef.current.interimResults = true;
            recognitionRef.current.lang = LANG_MAP[settings.mode === 'Shadowing' ? settings.sourceLang : settings.targetLang];

            recognitionRef.current.onresult = (event: SpeechRecognitionEvent) => {
                let interimTranscript = '';
                let finalTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                // Always append to the existing transcript state
                setTranscript(prev => prev + finalTranscript);
                // We don't display the interim transcript in this setup, but this is how you'd get it.
            };

            recognitionRef.current.onend = () => {
                if(isRecording) { // If it stops unexpectedly, restart it.
                   recognitionRef.current?.start();
                }
            };
        }
    }, [settings.targetLang, settings.sourceLang, settings.mode, isRecording]);

    useEffect(() => {
        // Reset edited transcript when the main transcript changes
        setEditedTranscript(transcript);
    }, [transcript]);


    const handlePlayPause = async () => {
        if (isAudioPlaying) {
            audioRef.current?.pause();
            setIsAudioPlaying(false);
            return;
        }

        setIsAudioPlaying(true);
        if (audioRef.current && audioRef.current.src) {
            audioRef.current.play();
        } else if (!synthesisInitiated.current) {
            synthesisInitiated.current = true;
            try {
                const audioContent = await synthesizeSpeechGoogleCloud(originalText, settings.sourceLang, settings.voiceQuality);
                const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
                const url = URL.createObjectURL(audioBlob);
                audioRef.current = new Audio(url);
                audioRef.current.play();
                audioRef.current.onended = () => setIsAudioPlaying(false);
            } catch (error) {
                console.error("Error synthesizing audio:", error);
                alert("Audio-Synthese fehlgeschlagen.");
                setIsAudioPlaying(false);
            }
        }
    };


    const toggleRecording = () => {
        if (isRecording) {
            recognitionRef.current?.stop();
            setIsRecording(false);
        } else {
            // Clear previous transcript before starting a new recording
            setTranscript('');
            recognitionRef.current?.start();
            setIsRecording(true);
        }
    };

    const getFeedback = async () => {
        setIsLoadingFeedback(true);
        setFeedback(null);
        setActiveTab('feedback');

        const langForFeedback = settings.mode === 'Shadowing' ? settings.sourceLang : settings.targetLang;

        // Fix: Use a response schema to ensure structured JSON output.
        const schema = {
            type: Type.OBJECT,
            properties: {
                clarity: { type: Type.INTEGER, description: "Klarheit und Aussprache (1-5 Sterne)" },
                accuracy: { type: Type.INTEGER, description: "Inhaltliche Genauigkeit (1-5 Sterne)" },
                completeness: { type: Type.INTEGER, description: "Vollständigkeit (1-5 Sterne)" },
                style: { type: Type.INTEGER, description: "Stil und Register (1-5 Sterne)" },
                terminology: { type: Type.INTEGER, description: "Terminologie (1-5 Sterne)" },
                overall: { type: Type.INTEGER, description: "Gesamteindruck (1-5 Sterne)" },
                summary: { type: Type.STRING, description: "Eine konstruktive Zusammenfassung der Leistung in 2-3 Sätzen." },
                errorAnalysis: {
                    type: Type.ARRAY,
                    description: "Liste von spezifischen Fehlern.",
                    items: {
                        type: Type.OBJECT,
                        properties: {
                            original: { type: Type.STRING, description: "Der relevante Satzteil aus dem Originaltext." },
                            interpretation: { type: Type.STRING, description: "Der entsprechende Satzteil aus der Verdolmetschung des Nutzers." },
                            suggestion: { type: Type.STRING, description: "Ein Korrekturvorschlag." },
                            explanation: { type: Type.STRING, description: "Eine kurze Erklärung des Fehlers." },
                            type: { type: Type.STRING, description: "Fehlertyp (z.B. 'Terminologie', 'Grammatik', 'Auslassung')." }
                        },
                        required: ["original", "interpretation", "suggestion", "explanation", "type"]
                    }
                }
            },
            required: ["clarity", "accuracy", "completeness", "style", "terminology", "overall", "summary", "errorAnalysis"]
        };


        const prompt = `
            Aufgabe: Analysiere die Leistung eines Dolmetschers.
            Modus: ${settings.mode}
            Ausgangssprache: ${settings.sourceLang}
            Zielsprache: ${langForFeedback}

            Originaltext:
            ---
            ${originalText}
            ---

            Verdolmetschung des Nutzers:
            ---
            ${editedTranscript}
            ---

            Analyseanweisungen:
            1.  Bewerte die Verdolmetschung anhand der folgenden Kriterien auf einer Skala von 1 bis 5 Sternen: Klarheit/Aussprache, Inhaltliche Genauigkeit, Vollständigkeit, Stil/Register, Terminologie und Gesamteindruck.
            2.  Schreibe eine kurze, konstruktive Zusammenfassung der Leistung.
            3.  Führe eine detaillierte Fehleranalyse durch. Identifiziere bis zu 5 signifikante Fehler. Gib für jeden Fehler den Originalteil, den entsprechenden Teil der Verdolmetschung, einen Korrekturvorschlag, eine kurze Erklärung und den Fehlertyp an. Wenn keine Fehler vorhanden sind, gib ein leeres Array für die Fehleranalyse zurück.
            4.  Gib die Ausgabe ausschließlich als JSON-Objekt zurück, das der folgenden Struktur entspricht. Gib keinen Markdown oder einleitenden Text aus.
        `;

        try {
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt,
                // Fix: Apply the response schema to the model config.
                config: {
                    responseMimeType: "application/json",
                    responseSchema: schema,
                },
            });

            const feedbackData = JSON.parse(response.text);
            setFeedback(feedbackData);
        } catch (error) {
            console.error("Error getting feedback:", error);
            alert("Fehler bei der Feedback-Analyse. Bitte versuchen Sie es erneut.");
        } finally {
            setIsLoadingFeedback(false);
        }
    };


    return (
        <div className="panel practice-area">
            <div className="tabs">
                <button className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`} onClick={() => setActiveTab('original')}>Originaltext</button>
                <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')} disabled={!transcript}>Transkript</button>
                <button className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} onClick={() => setActiveTab('feedback')} disabled={!feedback && !isLoadingFeedback}>Feedback</button>
            </div>

            <div className="tab-content">
                {activeTab === 'original' && (
                    <>
                        <div className="controls-bar">
                             <button onClick={handlePlayPause} className="btn-play-pause" disabled={synthesisInitiated.current && !audioRef.current?.src}>
                                {isAudioPlaying ? (
                                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>
                                ) : (
                                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"></path></svg>
                                )}
                            </button>
                            <p>Originaltext in {settings.sourceLang} abspielen.</p>
                        </div>
                        <div className="text-area">
                            <p>{originalText}</p>
                        </div>
                    </>
                )}
                {activeTab === 'transcript' && (
                     <>
                        <div className="controls-bar">
                            <p>Hier können Sie das Transkript vor der Feedback-Analyse korrigieren.</p>
                        </div>
                        <div className="text-area">
                           <textarea
                                className="text-area-editor"
                                value={editedTranscript}
                                onChange={(e) => setEditedTranscript(e.target.value)}
                            />
                        </div>
                    </>
                )}
                {activeTab === 'feedback' && (
                     <div className="text-area">
                        {isLoadingFeedback ? (
                            <div className="loading-overlay" style={{position: 'absolute', backgroundColor: 'var(--surface-color)'}}>
                                <div className="spinner"></div>
                                <p>Feedback wird analysiert...</p>
                            </div>
                        ) : (
                            feedback && <FeedbackDisplay feedback={feedback} />
                        )}
                    </div>
                )}
            </div>

            <div className="practice-footer">
                <span className="recording-status-text">
                    {isRecording ? `Aufnahme in ${settings.mode === 'Shadowing' ? settings.sourceLang : settings.targetLang}...` : 'Aufnahme starten'}
                </span>
                <button onClick={toggleRecording} className={`btn-record ${isRecording ? 'recording' : ''}`}>
                    <div className="mic-icon"></div>
                </button>
                 {activeTab === 'transcript' && transcript && (
                    <button className="btn btn-primary" style={{marginTop: '1rem'}} onClick={getFeedback} disabled={isLoadingFeedback}>
                       Feedback anfordern
                    </button>
                )}
            </div>
        </div>
    );
};

const DialoguePractice = ({ settings, dialogue }: {
    settings: Settings;
    dialogue: DialogueSegment[];
}) => {
    const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);
    const [dialogueState, setDialogueState] = useState<DialogueState>('synthesizing');
    const [showText, setShowText] = useState(false);
    const [userInterpretations, setUserInterpretations] = useState<string[]>([]);
    const [currentTranscript, setCurrentTranscript] = useState("");

    const audioRef = useRef<HTMLAudioElement | null>(null);
    const recognition = useRef<SpeechRecognition | null>(null);

    const currentSegment = dialogue[currentSegmentIndex];
    const isStegreif = settings.mode === 'Stegreifübersetzen';

    useEffect(() => {
        if (!currentSegment) return; // Guard against undefined segment

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            const rec = new SpeechRecognition();
            rec.continuous = true;
            rec.interimResults = true;
            rec.lang = LANG_MAP[currentSegment.lang === settings.sourceLang ? settings.targetLang : settings.sourceLang];
            recognition.current = rec;
        }
    }, [currentSegment]); // Re-init when segment changes

    const handleNextSegment = () => {
        setShowText(false);
        setCurrentTranscript("");
        if (currentSegmentIndex < dialogue.length - 1) {
            setCurrentSegmentIndex(prev => prev + 1);
            setDialogueState('synthesizing');
        } else {
            setDialogueState('finished');
        }
    };

    useEffect(() => {
        if (!currentSegment) return;

        const synthesizeAndPlay = async () => {
            try {
                const audioContent = await synthesizeSpeechGoogleCloud(currentSegment.text, currentSegment.lang, settings.voiceQuality);
                const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
                const url = URL.createObjectURL(audioBlob);
                audioRef.current = new Audio(url);
                audioRef.current.play();
                setDialogueState('playing');
                audioRef.current.onended = () => {
                    setDialogueState('waiting_for_record');
                };
            } catch (error) {
                console.error("Audio synthesis failed:", error);
                alert("Audio-Synthese fehlgeschlagen. Wechsle zum nächsten Segment.");
                handleNextSegment();
            }
        };

        if (dialogueState === 'synthesizing') {
            synthesizeAndPlay();
        }

    }, [currentSegment, dialogueState]); // Effect runs when segment or state changes

    useEffect(() => {
        const rec = recognition.current;
        if (!rec) return;

        rec.onresult = (event: SpeechRecognitionEvent) => {
             let finalTranscript = '';
             for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                }
            }
            if (finalTranscript) {
                setCurrentTranscript(prev => prev + " " + finalTranscript);
            }
        };

        rec.onend = () => {
             // Only move to next segment if recording ended naturally, not by manual stop
            if (dialogueState === 'recording') {
                setUserInterpretations(prev => [...prev, currentTranscript.trim()]);
                handleNextSegment();
            }
        };
        
        if (dialogueState === 'recording') {
            rec.start();
        } else {
            rec.stop();
        }

        return () => { // Cleanup
            rec.onresult = () => {};
            rec.onend = () => {};
            rec.onerror = () => {};
            if (dialogueState !== 'recording') {
                 rec.stop();
            }
        };

    }, [dialogueState, currentTranscript]);

    const handleRecordClick = () => {
        if (dialogueState === 'waiting_for_record') {
            setDialogueState('recording');
        } else if (dialogueState === 'recording') {
            recognition.current?.stop();
             // Manually trigger the state transition after stopping
            setUserInterpretations(prev => [...prev, currentTranscript.trim()]);
            handleNextSegment();
        }
    };

    const getStatusText = () => {
        if (!currentSegment) return 'Übung wird geladen...';
        switch (dialogueState) {
            case 'synthesizing':
                return `Segment ${currentSegmentIndex + 1}/${dialogue.length}: Audio wird vorbereitet...`;
            case 'playing':
                return `Segment ${currentSegmentIndex + 1}/${dialogue.length}: Teilnehmer ${currentSegment.type === 'Frage' ? 'A' : 'B'} (${currentSegment.lang}) spricht...`;
            case 'waiting_for_record':
                return `Segment ${currentSegmentIndex + 1}/${dialogue.length}: Sie sind dran. Klicken Sie auf das Mikrofon, um Ihre Verdolmetschung aufzunehmen.`;
            case 'recording':
                return `Segment ${currentSegmentIndex + 1}/${dialogue.length}: Aufnahme läuft... Klicken Sie zum Beenden.`;
            case 'finished':
                return 'Übung abgeschlossen.';
        }
    };

    if (dialogueState === 'finished') {
        const results: StructuredDialogueResult[] = dialogue.map((segment, index) => ({
            originalSegment: segment,
            userInterpretation: userInterpretations[index] || "Keine Aufnahme",
            interpretationLang: segment.lang === settings.sourceLang ? settings.targetLang : settings.sourceLang
        }));
        return <DialogueResults results={results} settings={settings} />;
    }

    if (!currentSegment) {
        return (
            <div className="panel practice-area">
                <div className="placeholder">
                    <h2>Dialog wird vorbereitet...</h2>
                </div>
            </div>
        );
    }

    return (
        <div className="panel practice-area dialogue-practice-container">
            <div className="dialogue-status">
                {getStatusText()}
            </div>
            <div className="current-segment-display">
                <div className="dialogue-text-container">
                    {isStegreif && !showText && (
                        <>
                            <p className="segment-text-hidden">Der Text ist für die Stegreifübersetzung ausgeblendet.</p>
                            <button className="btn btn-secondary btn-show-text" onClick={() => setShowText(true)}>Text anzeigen</button>
                        </>
                    )}
                    {(showText || !isStegreif) && <p className="segment-text">{currentSegment?.text}</p>}
                </div>
            </div>
             <div className="practice-footer">
                <span className="recording-status-text">
                    {dialogueState === 'recording' ? 'Aufnahme läuft...' :
                     dialogueState === 'waiting_for_record' ? 'Bereit zur Aufnahme' :
                     ''}
                </span>
                <button
                    onClick={handleRecordClick}
                    className={`btn-record ${dialogueState === 'recording' ? 'recording' : ''}`}
                    disabled={dialogueState !== 'waiting_for_record' && dialogueState !== 'recording'}
                >
                    <div className="mic-icon"></div>
                </button>
            </div>
        </div>
    );
};

const DialogueResults = ({ results, settings }: { results: StructuredDialogueResult[], settings: Settings }) => {
    const [activeTab, setActiveTab] = useState<'transcript' | 'feedback'>('transcript');
    const [feedback, setFeedback] = useState<Feedback | null>(null);
    const [isLoadingFeedback, setIsLoadingFeedback] = useState<boolean>(false);
     // State to hold potentially edited interpretations
    const [editedInterpretations, setEditedInterpretations] = useState<string[]>(results.map(r => r.userInterpretation));

    const handleInterpretationChange = (index: number, newText: string) => {
        const newEdits = [...editedInterpretations];
        newEdits[index] = newText;
        setEditedInterpretations(newEdits);
    };

    const getFeedback = async () => {
        setIsLoadingFeedback(true);
        setFeedback(null);
        setActiveTab('feedback');

        const formattedDialogue = results.map((result, index) => `
            Segment ${index + 1} (Original in ${result.originalSegment.lang}):
            ${result.originalSegment.text}

            Segment ${index + 1} (Verdolmetschung in ${result.interpretationLang}):
            ${editedInterpretations[index]}
        `).join('\n---\n');

         // Fix: Use a response schema to ensure structured JSON output.
        const schema = {
            type: Type.OBJECT,
            properties: {
                clarity: { type: Type.INTEGER, description: "Klarheit und Aussprache (1-5 Sterne)" },
                accuracy: { type: Type.INTEGER, description: "Inhaltliche Genauigkeit (1-5 Sterne)" },
                completeness: { type: Type.INTEGER, description: "Vollständigkeit (1-5 Sterne)" },
                style: { type: Type.INTEGER, description: "Stil und Register (1-5 Sterne)" },
                terminology: { type: Type.INTEGER, description: "Terminologie (1-5 Sterne)" },
                overall: { type: Type.INTEGER, description: "Gesamteindruck (1-5 Sterne)" },
                summary: { type: Type.STRING, description: "Eine konstruktive Zusammenfassung der Leistung in 2-3 Sätzen." },
                errorAnalysis: {
                    type: Type.ARRAY,
                    description: "Liste von spezifischen Fehlern.",
                    items: {
                        type: Type.OBJECT,
                        properties: {
                            original: { type: Type.STRING, description: "Der relevante Satzteil aus dem Originaltext." },
                            interpretation: { type: Type.STRING, description: "Der entsprechende Satzteil aus der Verdolmetschung des Nutzers." },
                            suggestion: { type: Type.STRING, description: "Ein Korrekturvorschlag." },
                            explanation: { type: Type.STRING, description: "Eine kurze Erklärung des Fehlers." },
                            type: { type: Type.STRING, description: "Fehlertyp (z.B. 'Terminologie', 'Grammatik', 'Auslassung')." }
                        },
                        required: ["original", "interpretation", "suggestion", "explanation", "type"]
                    }
                }
            },
            required: ["clarity", "accuracy", "completeness", "style", "terminology", "overall", "summary", "errorAnalysis"]
        };


        const prompt = `
            Aufgabe: Analysiere die Leistung eines Dolmetschers in einem Dialog.
            Modus: ${settings.mode}
            Dialogkontext:
            ---
            ${formattedDialogue}
            ---

            Analyseanweisungen:
            1.  Bewerte die Gesamtleistung im Dialog anhand der folgenden Kriterien auf einer Skala von 1 bis 5 Sternen: Klarheit/Aussprache, Inhaltliche Genauigkeit, Vollständigkeit, Stil/Register, Terminologie und Gesamteindruck.
            2.  Schreibe eine kurze, konstruktive Zusammenfassung der Leistung über den gesamten Dialog.
            3.  Führe eine detaillierte Fehleranalyse durch. Identifiziere bis zu 5 signifikante Fehler aus dem gesamten Dialog. Gib für jeden Fehler den Originalteil, den entsprechenden Teil der Verdolmetschung, einen Korrekturvorschlag, eine kurze Erklärung und den Fehlertyp an. Wenn keine Fehler vorhanden sind, gib ein leeres Array für die Fehleranalyse zurück.
            4.  Gib die Ausgabe ausschließlich als JSON-Objekt zurück, das der oben definierten Struktur entspricht. Gib keinen Markdown oder einleitenden Text aus.
        `;

        try {
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt,
                 // Fix: Apply the response schema to the model config.
                config: {
                    responseMimeType: "application/json",
                    responseSchema: schema,
                },
            });

            const feedbackData = JSON.parse(response.text);
            setFeedback(feedbackData);
        } catch (error) {
            console.error("Error getting feedback:", error);
            alert("Fehler bei der Feedback-Analyse. Bitte versuchen Sie es erneut.");
        } finally {
            setIsLoadingFeedback(false);
        }
    };

    return (
        <div className="panel practice-area dialogue-results-wrapper">
             <div className="tabs">
                <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>Transkript</button>
                <button className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} onClick={() => setActiveTab('feedback')} disabled={!feedback && !isLoadingFeedback}>Feedback</button>
            </div>
             <div className="tab-content">
                {activeTab === 'transcript' && (
                    <div className="text-area structured-transcript">
                        {results.map((result, index) => (
                            <div key={index} className="transcript-segment">
                                <div className="transcript-segment-header">
                                    <h4>Segment {index + 1}: {result.originalSegment.type} ({result.originalSegment.lang})</h4>
                                </div>
                                <p className="transcript-segment-original">{result.originalSegment.text}</p>
                                 <textarea
                                    className="transcript-segment-user text-area-editor"
                                    value={editedInterpretations[index]}
                                    onChange={(e) => handleInterpretationChange(index, e.target.value)}
                                />
                            </div>
                        ))}
                    </div>
                )}
                 {activeTab === 'feedback' && (
                     <div className="text-area">
                        {isLoadingFeedback ? (
                            <div className="loading-overlay" style={{position: 'absolute', backgroundColor: 'var(--surface-color)'}}>
                                <div className="spinner"></div>
                                <p>Feedback wird analysiert...</p>
                            </div>
                        ) : (
                            feedback && <FeedbackDisplay feedback={feedback} />
                        )}
                    </div>
                )}
             </div>
             <div className="practice-footer">
                <button className="btn btn-primary" onClick={getFeedback} disabled={isLoadingFeedback}>
                    {isLoadingFeedback ? "Wird analysiert..." : "Gesamt-Feedback anfordern"}
                </button>
            </div>
        </div>
    );
};

const FeedbackDisplay = ({ feedback }: { feedback: Feedback }) => {
    const renderStars = (rating: number) => {
        return Array.from({ length: 5 }, (_, i) => (
            <span key={i} className={`star ${i < rating ? 'filled' : ''}`}>★</span>
        ));
    };

    return (
        <div className="feedback-content">
            <h3>Feedback-Analyse</h3>
            <table className="ratings-table">
                <tbody>
                    <tr><td>Klarheit</td><td>{renderStars(feedback.clarity)}</td></tr>
                    <tr><td>Genauigkeit</td><td>{renderStars(feedback.accuracy)}</td></tr>
                    <tr><td>Vollständigkeit</td><td>{renderStars(feedback.completeness)}</td></tr>
                    <tr><td>Stil</td><td>{renderStars(feedback.style)}</td></tr>
                    <tr><td>Terminologie</td><td>{renderStars(feedback.terminology)}</td></tr>
                    <tr><td><b>Gesamteindruck</b></td><td><b>{renderStars(feedback.overall)}</b></td></tr>
                </tbody>
            </table>
            <h3>Zusammenfassung</h3>
            <p>{feedback.summary}</p>
            {feedback.errorAnalysis && feedback.errorAnalysis.length > 0 && (
                <>
                    <h3>Fehleranalyse</h3>
                    <ul className="error-analysis-list">
                        {feedback.errorAnalysis.map((item, index) => (
                            <li key={index}>
                                <p><strong>Typ:</strong> {item.type}</p>
                                <p><strong>Original:</strong> "{item.original}"</p>
                                <p><strong>Ihre Version:</strong> "{item.interpretation}"</p>
                                <p><strong>Vorschlag:</strong> "{item.suggestion}"</p>
                                <p><strong>Erklärung:</strong> {item.explanation}</p>
                            </li>
                        ))}
                    </ul>
                </>
            )}
        </div>
    );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);