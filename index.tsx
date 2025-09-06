import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";

// --- WEB SPEECH API TYPES ---
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
type DialogueState = 'synthesizing' | 'playing' | 'waiting_for_record' | 'recording' | 'processing_result' | 'finished';
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

const SPEECH_LENGTH_TARGETS: Record<SpeechLength, { min: number; max: number }> = {
  "Kurz": { min: 1200, max: 1600 },
  "Mittel": { min: 1800, max: 2200 },
  "Prüfung": { min: 3300, max: 3700 },
};

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
const generateContentWithRetry = async (prompt: string, retries = 3, delay = 1000): Promise<string> => {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt,
            });
            const text = response.text;
            if (text === undefined) {
                throw new Error("API response did not contain text.");
            }
            return text;
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

const adjustTextLength = async (initialText: string, settings: Settings): Promise<string> => {
    const { speechLength, topic } = settings;
    let target: { min: number; max: number; };

    if (settings.mode === 'Stegreifübersetzen') {
        target = { min: 1280, max: 1420 };
    } else {
        target = SPEECH_LENGTH_TARGETS[speechLength];
    }

    let currentText = initialText;
    let attempts = 0;
    const MAX_ADJUSTMENT_ATTEMPTS = 3;

    while (
        (currentText.length < target.min || currentText.length > target.max) &&
        attempts < MAX_ADJUSTMENT_ATTEMPTS
    ) {
        attempts++;
        let adjustmentPrompt = '';

        if (currentText.length < target.min) {
            const diff = target.min - currentText.length;
            const paragraphsToAdd = diff > 800 ? 2 : 1;
            adjustmentPrompt = `Der folgende Text zum Thema "${topic}" ist zu kurz. Aktuelle Länge: ${currentText.length} Zeichen. Ziel ist ${target.min}-${target.max} Zeichen. Bitte füge ${paragraphsToAdd} sinnvollen Absatz/Absätze hinzu, um den Text zu verlängern, aber bleibe im Stil des Originaltextes. Gib NUR den vollständigen, neuen Text aus.
            
            Originaltext:
            """
            ${currentText}
            """`;
        } else { // currentText.length > target.max
            const diff = currentText.length - target.max;
            const paragraphsToRemove = diff > 800 ? 2 : 1;
            adjustmentPrompt = `Der folgende Text zum Thema "${topic}" ist zu lang. Aktuelle Länge: ${currentText.length} Zeichen. Ziel ist ${target.min}-${target.max} Zeichen. Bitte kürze den Text um ${paragraphsToRemove} Absatz/Absätze, ohne den Kerninhalt zu verlieren. Gib NUR den vollständigen, gekürzten Text aus.

            Originaltext:
            """
            ${currentText}
            """`;
        }

        console.log(`Adjustment attempt ${attempts}: Current length ${currentText.length}, target ${target.min}-${target.max}. Adjusting...`);
        currentText = await generateContentWithRetry(adjustmentPrompt);
    }

    if (attempts >= MAX_ADJUSTMENT_ATTEMPTS) {
        console.warn(`Could not adjust text to target length after ${MAX_ADJUSTMENT_ATTEMPTS} attempts. Using last generated text.`);
    }

    return currentText;
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
        const isMonologueMode = settings.mode === 'Vortragsdolmetschen' || settings.mode === 'Simultandolmetschen' || settings.mode === 'Shadowing';
        const isSightTranslationMode = settings.mode === 'Stegreifübersetzen';

        if (isMonologueMode) {
            prompt = `Erstelle einen Text zum Thema "${settings.topic}" für eine Dolmetschübung im Modus "${settings.mode}". Die Sprache des Textes soll ${settings.sourceLang} sein. Die Länge soll dem Level "${settings.speechLength}" entsprechen. Gib nur den reinen Text aus, ohne Titel oder zusätzliche Kommentare.`;
        } else if (isSightTranslationMode) {
            prompt = `Erstelle einen zusammenhängenden Text zum Thema "${settings.topic}" in ${settings.sourceLang} für eine Stegreifübersetzungs-Übung. Der Text soll eine Länge zwischen 1280 und 1420 Zeichen haben. Gib nur den reinen Text aus, ohne Titel oder zusätzliche Kommentare.`;
        } else { // Gesprächsdolmetschen
            prompt = `Erstelle einen realistischen Dialog zwischen zwei Personen (A und B) zum Thema "${settings.topic}". Der Dialog soll im Frage-Antwort-Format sein. Person A (${settings.sourceLang}) stellt Fragen, Person B (${settings.targetLang}) antwortet. Der Dialog soll insgesamt 4 Segmente haben (A-Frage, B-Antwort, A-Frage, B-Antwort). Jedes Segment soll eine Länge von "${settings.qaLength}" haben. Gib nur den reinen Dialog aus, ohne zusätzliche Erklärungen, formatiert als JSON-Array mit Objekten, die "type", "text" und "lang" enthalten. Beispiel: [{"type": "Frage", "text": "...", "lang": "${settings.sourceLang}"}, ...]`;
        }
  
        const generatedContent = await generateContentWithRetry(prompt);

        if (isMonologueMode) {
            const adjustedText = await adjustTextLength(generatedContent, settings);
            setOriginalText(adjustedText);
        } else if (isSightTranslationMode) {
            const adjustedText = await adjustTextLength(generatedContent, settings);
            const sightTranslationDialogue: DialogueSegment[] = [{
                type: 'Frage', // Type is arbitrary for sight translation, just to match the data structure
                text: adjustedText,
                lang: settings.sourceLang,
            }];
            setDialogue(sightTranslationDialogue);
        } else { // Gesprächsdolmetschen
            const parsedDialogue = JSON.parse(generatedContent);
            setDialogue(parsedDialogue);
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
      if (result && typeof result === 'string') {
        setOriginalText(result);
        setExerciseState('ready');
        setExerciseId(Date.now());
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
      
      const isMonologue = settings.mode === 'Vortragsdolmetschen' || settings.mode === 'Simultandolmetschen' || settings.mode === 'Shadowing';

      return (
          <>
              <div className="form-group">
                  <label htmlFor="topic">Thema</label>
                  <input type="text" id="topic" className="form-control" value={settings.topic} onChange={e => handleSettingChange('topic', e.target.value)} />
              </div>
              {settings.mode === 'Gesprächsdolmetschen' ? (
                  <div className="form-group">
                      <label htmlFor="qaLength">Satzlänge</label>
                      <select id="qaLength" className="form-control" value={settings.qaLength} onChange={e => handleSettingChange('qaLength', e.target.value)}>
                          {QA_LENGTHS.map(len => <option key={len} value={len}>{len}</option>)}
                      </select>
                  </div>
              ) : isMonologue ? (
                  <div className="form-group">
                      <label htmlFor="speechLength">Textlänge</label>
                      <select id="speechLength" className="form-control" value={settings.speechLength} onChange={e => handleSettingChange('speechLength', e.target.value)}>
                          {SPEECH_LENGTHS.map(len => <option key={len} value={len}>{len}</option>)}
                      </select>
                  </div>
              ) : null}
          </>
      );
  };
  
  const renderUploadOptions = () => {
      if (settings.sourceType !== 'upload' || settings.mode === "Gesprächsdolmetschen" || settings.mode === "Stegreifübersetzen") return null;
      return (
        <div className="form-group">
          <label>Eigener Text</label>
          <div className="upload-group">
            <button className="btn btn-secondary" onClick={handleFileSelect}>Datei wählen</button>
            <span className="file-name" title={fileName}>{fileName || "Keine Datei gewählt"}</span>
          </div>
           <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".txt"
            style={{ display: 'none' }}
          />
        </div>
      );
  };

  return (
    <div className="panel settings-panel">
        <div>
            <h2>Einstellungen</h2>
            <div className="form-group">
                <label htmlFor="mode">Modus</label>
                <select id="mode" className="form-control" value={settings.mode} onChange={e => handleSettingChange('mode', e.target.value as InterpretingMode)}>
                    {MODES.map(mode => <option key={mode} value={mode}>{mode}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label htmlFor="sourceLang">Ausgangssprache</label>
                <select id="sourceLang" className="form-control" value={settings.sourceLang} onChange={e => handleSettingChange('sourceLang', e.target.value as Language)}>
                    {LANGUAGES.map(lang => <option key={lang} value={lang}>{lang}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label htmlFor="targetLang">Zielsprache</label>
                <select id="targetLang" className="form-control" value={settings.targetLang} onChange={e => handleSettingChange('targetLang', e.target.value as Language)}>
                    {LANGUAGES.map(lang => <option key={lang} value={lang}>{lang}</option>)}
                </select>
            </div>
            {renderSourceTypeOptions()}
            {renderAiOptions()}
            {renderUploadOptions()}
            {settings.mode !== 'Stegreifübersetzen' && (
             <div className="form-group">
                <label htmlFor="voiceQuality">Stimmqualität</label>
                <select id="voiceQuality" className="form-control" value={settings.voiceQuality} onChange={e => handleSettingChange('voiceQuality', e.target.value as VoiceQuality)}>
                    <option value="Standard">Standard</option>
                    <option value="Premium">Premium (bessere Qualität)</option>
                </select>
            </div>
            )}
        </div>
        <div className="settings-footer">
            <button className="btn btn-primary btn-large" onClick={onStart} disabled={isLoading}>
                {isLoading ? 'Wird erstellt...' : 'Übung erstellen'}
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
  if (exerciseState === 'idle') {
    return (
      <div className="panel practice-area">
        <div className="placeholder">
          <h2>Willkommen beim Dolmetsch-Trainer Pro</h2>
          <p>Passen Sie die Einstellungen links an und erstellen Sie Ihre erste Übung.</p>
        </div>
      </div>
    );
  }
  if (exerciseState === 'generating') {
     return (
        <div className="panel practice-area">
            <div className="loading-overlay">
                <div className="spinner"></div>
                <p>KI-Übung wird für Sie erstellt...</p>
            </div>
        </div>
    );
  }
  if (exerciseState === 'error') {
     return (
        <div className="panel practice-area">
            <div className="placeholder">
                <h2>Fehler</h2>
                <p>{errorMessage}</p>
            </div>
        </div>
    );
  }
  // Render the correct practice component based on the mode
  switch(settings.mode) {
      case "Vortragsdolmetschen":
          return <MonologuePractice settings={settings} originalText={originalText} mode="consecutive" />;
      case "Simultandolmetschen":
          return <MonologuePractice settings={settings} originalText={originalText} mode="simultaneous" />;
      case "Shadowing":
          return <MonologuePractice settings={settings} originalText={originalText} mode="shadowing" />;
      case "Gesprächsdolmetschen":
          return <DialoguePractice settings={settings} dialogue={dialogue} />;
      case "Stegreifübersetzen":
          return <SightTranslationPractice settings={settings} dialogue={dialogue} />;
      default:
          return <div className="panel practice-area"><p>Modus nicht gefunden.</p></div>;
  }
};

const MonologuePractice = ({ settings, originalText: initialText, mode }: {
  settings: Settings;
  originalText: string;
  mode: 'consecutive' | 'simultaneous' | 'shadowing';
}) => {
  const [activeTab, setActiveTab] = useState<PracticeAreaTab>('original');
  const [originalText, setOriginalText] = useState(initialText);
  const [transcript, setTranscript] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [isGeneratingFeedback, setIsGeneratingFeedback] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const recognition = useRef<SpeechRecognition | null>(null);
  const [isEditingOriginalText, setIsEditingOriginalText] = useState(false);

  const targetLang = mode === 'shadowing' ? settings.sourceLang : settings.targetLang;

  useEffect(() => {
    // Setup SpeechRecognition
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognitionAPI) {
        recognition.current = new SpeechRecognitionAPI();
        recognition.current.continuous = true;
        recognition.current.interimResults = true;
        recognition.current.lang = LANG_MAP[targetLang];

        recognition.current.onresult = (event: SpeechRecognitionEvent) => {
            let interimTranscript = '';
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
             setTranscript(prev => prev + finalTranscript + interimTranscript);
        };
        recognition.current.onerror = (event: SpeechRecognitionErrorEvent) => {
            console.error("Speech recognition error", event.error, event.message);
            setIsRecording(false);
        };
        recognition.current.onend = () => {
             // Do not stop recording automatically unless told to.
             if (isRecording) {
                 recognition.current?.start();
             }
        };

    } else {
        console.warn("Speech Recognition API not supported in this browser.");
    }
  }, [targetLang, isRecording]);

  const handlePlayPause = async () => {
    if (isPlaying) {
      audioRef.current?.pause();
      setIsPlaying(false);
      return;
    }
    
    if (!audioRef.current) {
        try {
            const audioContent = await synthesizeSpeechGoogleCloud(originalText, settings.sourceLang, settings.voiceQuality);
            const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
            const url = URL.createObjectURL(audioBlob);
            audioRef.current = new Audio(url);
            audioRef.current.onended = () => setIsPlaying(false);
        } catch(error) {
            console.error("Failed to synthesize speech:", error);
            return;
        }
    }
    audioRef.current.play();
    setIsPlaying(true);

    if (mode === 'simultaneous' || mode === 'shadowing') {
        // Automatically start recording when playback starts for these modes
        handleRecord();
    }
  };

  const handleRecord = () => {
    if (isRecording) {
      recognition.current?.stop();
      setIsRecording(false);
    } else {
      if (mode === 'consecutive') {
          setTranscript(''); // Clear previous transcript for new recording
      }
      recognition.current?.start();
      setIsRecording(true);
    }
  };
  
  const handleEditToggle = () => {
    setIsEditingOriginalText(prev => !prev);
  };
  
  const getFeedback = async () => {
      setIsGeneratingFeedback(true);
      setFeedback(null);
      setActiveTab('feedback');

      const prompt = `
        Kontext: Eine Dolmetschübung im Modus "${settings.mode}".
        Ausgangssprache: ${settings.sourceLang}
        Zielsprache: ${targetLang}
        
        Originaltext:
        """
        ${originalText}
        """

        Verdolmetschung des Benutzers:
        """
        ${transcript}
        """

        Aufgabe: Analysiere die Verdolmetschung. Bewerte sie in den folgenden Kategorien von 1 (sehr schlecht) bis 5 (ausgezeichnet): Klarheit/Verständlichkeit, Genauigkeit, Vollständigkeit, Stil/Register, Terminologie. Gib auch eine Gesamtbewertung (1-5). Erstelle eine kurze Zusammenfassung (2-3 Sätze) der Leistung.
        Erstelle dann eine detaillierte Fehleranalyse. Identifiziere bis zu 5 signifikante Fehler oder verbesserungswürdige Stellen. Für jeden Punkt, gib den originalen Teil, die Interpretation des Nutzers, einen Korrekturvorschlag und eine kurze Erklärung an.
        
        Gib deine Antwort NUR als JSON-Objekt im folgenden Format aus. Keine zusätzlichen Texte oder Erklärungen.

        {
          "clarity": number,
          "accuracy": number,
          "completeness": number,
          "style": number,
          "terminology": number,
          "overall": number,
          "summary": "string",
          "errorAnalysis": [
            {
              "original": "string",
              "interpretation": "string",
              "suggestion": "string",
              "explanation": "string",
              "type": "string (z.B. 'Auslassung', 'Terminologiefehler', 'Grammatikfehler', 'Stilfehler')"
            }
          ]
        }
      `;
      try {
          const feedbackText = await generateContentWithRetry(prompt);
          const feedbackJson = JSON.parse(feedbackText);
          setFeedback(feedbackJson);
      } catch (error) {
          console.error("Error getting feedback:", error);
      } finally {
          setIsGeneratingFeedback(false);
      }
  };

  return (
    <div className="panel practice-area">
      <div className="tabs">
        <button className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`} onClick={() => setActiveTab('original')}>Originaltext</button>
        <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>Meine Verdolmetschung</button>
        <button className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} onClick={() => setActiveTab('feedback')} disabled={!transcript}>Feedback</button>
      </div>
      <div className="tab-content">
        {activeTab === 'original' && (
          <>
            <div className="controls-bar">
                 <button onClick={handlePlayPause} className="btn-play-pause" disabled={!originalText}>
                    {isPlaying ? (
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>
                    ) : (
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"></path></svg>
                    )}
                 </button>
                 <p>Originaltext anhören</p>
                 <button className="btn btn-secondary" onClick={handleEditToggle} style={{ marginLeft: 'auto' }}>
                    {isEditingOriginalText ? 'Speichern' : 'Bearbeiten'}
                 </button>
            </div>
            <div className="text-area">
                <textarea
                    className={`text-area-editor ${isEditingOriginalText ? 'is-editing' : ''}`}
                    value={originalText}
                    onChange={(e) => {
                        setOriginalText(e.target.value);
                        // Invalidate existing audio if text changes
                        if (audioRef.current) {
                            audioRef.current.pause();
                            setIsPlaying(false);
                            audioRef.current = null;
                        }
                    }}
                    readOnly={!isEditingOriginalText}
                />
            </div>
          </>
        )}
        {activeTab === 'transcript' && (
          <div className="text-area">
            <textarea className="text-area-editor" value={transcript} onChange={(e) => setTranscript(e.target.value)} placeholder="Hier erscheint Ihre Verdolmetschung..." />
          </div>
        )}
        {activeTab === 'feedback' && (
             <FeedbackDisplay feedback={feedback} isLoading={isGeneratingFeedback} onGenerate={getFeedback} transcriptProvided={!!transcript} />
        )}
      </div>
      <div className="practice-footer">
          <p className="recording-status-text">{isRecording ? `Aufnahme in ${targetLang}...` : "Bereit zur Aufnahme"}</p>
          <button className={`btn-record ${isRecording ? 'recording' : ''}`} onClick={handleRecord}>
            <div className="mic-icon"></div>
          </button>
      </div>
    </div>
  );
};


const DialoguePractice = ({ settings, dialogue }: {
  settings: Settings;
  dialogue: DialogueSegment[];
}) => {
    const [segmentIndex, setSegmentIndex] = useState(0);
    const [dialogueState, setDialogueState] = useState<DialogueState>('synthesizing');
    const [dialogueResults, setDialogueResults] = useState<StructuredDialogueResult[]>([]);
    const [currentTranscript, setCurrentTranscript] = useState('');
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const recognition = useRef<SpeechRecognition | null>(null);
    const [activeTab, setActiveTab] = useState<'practice' | 'results'>('practice');

    const currentSegment = dialogue[segmentIndex];
    const targetLang = currentSegment?.lang === settings.sourceLang ? settings.targetLang : settings.sourceLang;
    const isUserTurn = currentSegment?.lang === settings.targetLang;

    useEffect(() => {
        // Automatically start the first segment synthesis
        if (segmentIndex === 0 && dialogueState === 'synthesizing') {
            synthesizeAndPlayCurrentSegment();
        }
    }, [segmentIndex, dialogueState, dialogue]);
    
    // Speech Recognition Setup
    useEffect(() => {
        if (!currentSegment) return; // Guard against running when dialogue is finished

        const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognitionAPI) {
            console.warn("Speech Recognition not supported.");
            return;
        }

        const rec = new SpeechRecognitionAPI();
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = LANG_MAP[targetLang];

        rec.onresult = (event: SpeechRecognitionEvent) => {
            let interim = '';
            let final = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    final += event.results[i][0].transcript;
                } else {
                    interim += event.results[i][0].transcript;
                }
            }
            setCurrentTranscript(prev => prev + final + interim);
        };
        
        rec.onerror = (e) => console.error("Recognition error:", e);
        rec.onend = () => {
            if (dialogueState === 'recording') {
                handleNextSegment();
            }
        };

        recognition.current = rec;

        return () => {
            rec.onresult = () => {};
            rec.onend = () => {};
            rec.onerror = () => {};
            rec.stop();
        };

    }, [targetLang, dialogueState, currentSegment]);

    // Main state machine for dialogue flow
    useEffect(() => {
        if (dialogueState === 'recording' && recognition.current) {
            setCurrentTranscript('');
            recognition.current.start();
        }
    }, [dialogueState]);


    const synthesizeAndPlayCurrentSegment = async () => {
        if (!currentSegment) return;
        
        setDialogueState('synthesizing');
        try {
            const audioContent = await synthesizeSpeechGoogleCloud(currentSegment.text, currentSegment.lang, settings.voiceQuality);
            const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
            const url = URL.createObjectURL(audioBlob);
            audioRef.current = new Audio(url);
            audioRef.current.onended = () => {
                // After participant B speaks, it's user's turn
                 if (isUserTurn) {
                     setDialogueState('waiting_for_record');
                 } else { // After participant A speaks, it's user's turn
                     setDialogueState('recording');
                 }
            };
            setDialogueState('playing');
            audioRef.current.play();
        } catch (error) {
            console.error("Speech synthesis failed:", error);
            // Decide how to handle this error - maybe try again or show an error message
        }
    };
    
    const handleNextSegment = () => {
        if (!currentSegment) return;
         setDialogueResults(prev => [...prev, {
            originalSegment: currentSegment,
            userInterpretation: currentTranscript,
            interpretationLang: targetLang
        }]);

        const nextIndex = segmentIndex + 1;
        if (nextIndex < dialogue.length) {
            setSegmentIndex(nextIndex);
            setDialogueState('synthesizing'); // Explicitly set state for next segment
            synthesizeAndPlayCurrentSegment();
        } else {
            setDialogueState('finished');
            setActiveTab('results');
        }
    };
    
    const handleRecordClick = () => {
        if (dialogueState === 'recording') {
             if (recognition.current) {
                recognition.current.stop();
            }
            // onEnd will trigger handleNextSegment
        } else if (dialogueState === 'waiting_for_record') {
            setDialogueState('recording');
        }
    };
    
    const getStatusText = () => {
        if (!currentSegment) return "Übung beendet.";
        const participant = currentSegment.lang === settings.sourceLang ? 'A' : 'B';
        switch (dialogueState) {
            case 'synthesizing': return `Teilnehmer ${participant} (${currentSegment.lang}) wird vorbereitet...`;
            case 'playing': return `Teilnehmer ${participant} (${currentSegment.lang}) spricht...`;
            case 'waiting_for_record': return `Sie sind dran. Klicken Sie auf Aufnahme, um als Teilnehmer ${participant} (${targetLang}) zu antworten.`;
            case 'recording': return `Aufnahme läuft... (Sie als Teilnehmer ${participant} in ${targetLang})`;
            case 'finished': return "Dialog beendet. Sehen Sie sich die Ergebnisse an.";
            default: return "Bereit.";
        }
    };

    if (activeTab === 'results') {
        return (
             <div className="panel practice-area">
                <div className="tabs">
                    <button className="tab-btn" onClick={() => setActiveTab('practice')}>Übung</button>
                    <button className="tab-btn active">Ergebnisse</button>
                </div>
                <div className="tab-content">
                    <StructuredTranscript results={dialogueResults} />
                </div>
            </div>
        )
    }

    return (
        <div className="panel practice-area dialogue-practice-container">
            <div className="dialogue-status">
                {getStatusText()}
            </div>
             <div className="current-segment-display">
                {dialogueState === 'playing' ? (
                     <p className="segment-text">{currentSegment?.text}</p>
                ) : dialogueState === 'recording' || dialogueState === 'waiting_for_record' ? (
                    <p className="segment-text">{currentTranscript || "..."}</p>
                ) : (
                    <div className="spinner"></div>
                )}
            </div>
            <div className="practice-footer">
                <p className="recording-status-text">
                    {dialogueState === 'recording' ? `Aufnahme in ${targetLang}...` : ''}
                </p>
                <button
                    className={`btn-record ${dialogueState === 'recording' ? 'recording' : ''}`}
                    onClick={handleRecordClick}
                    disabled={dialogueState !== 'recording' && dialogueState !== 'waiting_for_record'}
                >
                    <div className="mic-icon"></div>
                </button>
            </div>
        </div>
    );
};

const SightTranslationPractice = ({ settings, dialogue }: {
  settings: Settings;
  dialogue: DialogueSegment[];
}) => {
    const [editableDialogue, setEditableDialogue] = useState(dialogue);
    const [isEditing, setIsEditing] = useState(false);
    const [segmentIndex, setSegmentIndex] = useState(0);
    const [dialogueResults, setDialogueResults] = useState<StructuredDialogueResult[]>([]);
    const [currentTranscript, setCurrentTranscript] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const recognition = useRef<SpeechRecognition | null>(null);
    const [activeTab, setActiveTab] = useState<'practice' | 'results'>('practice');

    const currentSegment = editableDialogue[segmentIndex];
    const targetLang = currentSegment?.lang === settings.sourceLang ? settings.targetLang : settings.sourceLang;
    
    // Handlers for editing
    const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const newText = e.target.value;
        const updatedDialogue = [...editableDialogue];
        updatedDialogue[segmentIndex] = { ...updatedDialogue[segmentIndex], text: newText };
        setEditableDialogue(updatedDialogue);
    };

    const handleEditToggle = () => {
        setIsEditing(prev => !prev);
    };

    // Speech Recognition Setup
    useEffect(() => {
        if (!currentSegment) return; // Guard against running when dialogue is finished

        const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognitionAPI) {
            console.warn("Speech Recognition not supported.");
            return;
        }

        const rec = new SpeechRecognitionAPI();
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = LANG_MAP[targetLang];

        rec.onresult = (event: SpeechRecognitionEvent) => {
            let interim = '';
            let final = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    final += event.results[i][0].transcript;
                } else {
                    interim += event.results[i][0].transcript;
                }
            }
            setCurrentTranscript(prev => prev + final + interim);
        };
        
        rec.onerror = (e) => {
            console.error("Recognition error:", e);
            setIsRecording(false);
        };
        rec.onend = () => {
             if (isRecordingRef.current) { // If it stops unexpectedly, restart it
                rec.start();
             }
        };

        recognition.current = rec;
        
        return () => {
            rec.onresult = () => {};
            rec.onend = () => {};
            rec.onerror = () => {};
            if (rec) {
                isRecordingRef.current = false; // Prevent restart on manual stop
                rec.stop();
            }
        };

    }, [targetLang, currentSegment]);
    
    // useRef to get the latest isRecording state in the onend callback
    const isRecordingRef = useRef(isRecording);
    useEffect(() => {
        isRecordingRef.current = isRecording;
    }, [isRecording]);


    const handleNextSegment = () => {
        if (isRecording) { // Stop recording before moving on
            handleRecordClick();
        }
        
        if (!currentSegment) return;
         setDialogueResults(prev => [...prev, {
            originalSegment: currentSegment,
            userInterpretation: currentTranscript,
            interpretationLang: targetLang
        }]);
        setCurrentTranscript('');

        const nextIndex = segmentIndex + 1;
        if (nextIndex < dialogue.length) {
            setSegmentIndex(nextIndex);
        } else {
            setActiveTab('results');
        }
    };
    
    const handleRecordClick = () => {
       if (isRecording) {
           isRecordingRef.current = false; // Signal that this is a manual stop
           recognition.current?.stop();
           setIsRecording(false);
       } else {
            setCurrentTranscript('');
            recognition.current?.start();
            setIsRecording(true);
       }
    };

    if (!currentSegment) {
         return (
             <div className="panel practice-area">
                <div className="tabs">
                    <button className="tab-btn" onClick={() => setActiveTab('practice')}>Übung</button>
                    <button className="tab-btn active">Ergebnisse</button>
                </div>
                <div className="tab-content">
                    <StructuredTranscript results={dialogueResults} />
                </div>
            </div>
        )
    }

    if (activeTab === 'results') {
        return (
             <div className="panel practice-area">
                <div className="tabs">
                    <button className="tab-btn" onClick={() => setActiveTab('practice')}>Übung</button>
                    <button className="tab-btn active">Ergebnisse</button>
                </div>
                <div className="tab-content">
                    <StructuredTranscript results={dialogueResults} />
                </div>
            </div>
        )
    }

    return (
        <div className="panel practice-area dialogue-practice-container">
            <div className="dialogue-status">
                Segment {segmentIndex + 1} / {dialogue.length} ({currentSegment.lang} {'->'} {targetLang})
                <button className="btn btn-secondary" onClick={handleEditToggle} style={{ marginLeft: 'auto', padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}>
                    {isEditing ? 'Speichern' : 'Bearbeiten'}
                </button>
            </div>
            <div className="text-area" style={{ flexGrow: 1, border: '1px solid var(--border-color)', borderRadius: 'var(--border-radius)', padding: '1rem' }}>
                <textarea
                    className={`text-area-editor ${isEditing ? 'is-editing' : ''}`}
                    value={currentSegment.text}
                    onChange={handleTextChange}
                    readOnly={!isEditing}
                    style={{ height: '100%' }}
                />
            </div>
            <div className="text-area" style={{ height: '150px', marginTop: '1rem', border: '1px solid var(--border-color)', borderRadius: 'var(--border-radius)', padding: '1rem', backgroundColor: '#f8f9fa' }}>
                <p>{currentTranscript || "..."}</p>
            </div>
            
            <div className="practice-footer">
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                     <button
                        className={`btn-record ${isRecording ? 'recording' : ''}`}
                        onClick={handleRecordClick}
                    >
                        <div className="mic-icon"></div>
                    </button>
                    <button className="btn btn-primary" onClick={handleNextSegment}>
                        {segmentIndex === dialogue.length - 1 ? "Ergebnisse anzeigen" : "Nächstes Segment"}
                    </button>
                </div>
            </div>
        </div>
    );
};


const FeedbackDisplay = ({ feedback, isLoading, onGenerate, transcriptProvided }: {
  feedback: Feedback | null;
  isLoading: boolean;
  onGenerate: () => void;
  transcriptProvided: boolean;
}) => {
  if (isLoading) {
    return (
      <div className="loading-overlay" style={{ position: 'relative', background: 'transparent' }}>
        <div className="spinner"></div>
        <p>Feedback wird analysiert...</p>
      </div>
    );
  }

  if (!feedback) {
    return (
      <div className="placeholder">
        <h2>Feedback</h2>
        <p>Nachdem Sie eine Verdolmetschung aufgenommen haben, können Sie hier eine detaillierte KI-Analyse anfordern.</p>
        <button className="btn btn-primary btn-large" onClick={onGenerate} disabled={!transcriptProvided}>
          Feedback generieren
        </button>
      </div>
    );
  }

  const StarRating = ({ score }: { score: number }) => (
    <span>
      {[...Array(5)].map((_, i) => (
        <span key={i} className={`star ${i < score ? 'filled' : ''}`}>★</span>
      ))}
    </span>
  );

  return (
    <div className="feedback-content">
        <h3>Zusammenfassung</h3>
        <p>{feedback.summary}</p>
        
        <table className="ratings-table">
            <tbody>
                <tr><td>Klarheit/Verständlichkeit</td><td><StarRating score={feedback.clarity} /></td></tr>
                <tr><td>Genauigkeit</td><td><StarRating score={feedback.accuracy} /></td></tr>
                <tr><td>Vollständigkeit</td><td><StarRating score={feedback.completeness} /></td></tr>
                <tr><td>Stil/Register</td><td><StarRating score={feedback.style} /></td></tr>
                <tr><td>Terminologie</td><td><StarRating score={feedback.terminology} /></td></tr>
                <tr><td><strong>Gesamt</strong></td><td><strong><StarRating score={feedback.overall} /></strong></td></tr>
            </tbody>
        </table>

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
        <button className="btn btn-secondary" onClick={onGenerate} style={{marginTop: '1rem'}}>
          Feedback erneut generieren
        </button>
    </div>
  );
};

const StructuredTranscript = ({ results }: { results: StructuredDialogueResult[] }) => {
    if (results.length === 0) {
        return <p>Noch keine Ergebnisse vorhanden.</p>;
    }

    return (
        <div className="structured-transcript text-area">
            {results.map((result, index) => (
                <div key={index} className="transcript-segment">
                     <div className="transcript-segment-header">
                        <h4>Segment {index + 1}: {result.originalSegment.type} ({result.originalSegment.lang})</h4>
                    </div>
                    <p className="transcript-segment-original">{result.originalSegment.text}</p>
                    <p className="transcript-segment-user">
                        <strong>Ihre Verdolmetschung ({result.interpretationLang}):</strong><br />
                        {result.userInterpretation || <em>Keine Aufnahme für dieses Segment.</em>}
                    </p>
                </div>
            ))}
        </div>
    );
}

const root = createRoot(document.getElementById('root')!);
root.render(<App />);