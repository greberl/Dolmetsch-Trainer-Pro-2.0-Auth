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
    type: "Inhalt" | "Sprache";
}

interface Feedback {
    contentRating: number;
    languageRating: number;
    contentSummary: string;
    languageSummary: string;
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

const punctuateTextWithAI = async (rawText: string, lang: Language): Promise<string> => {
    if (!rawText || rawText.trim() === '') {
        return rawText;
    }
    const prompt = `Füge dem folgenden Text auf ${lang} die korrekte Zeichensetzung (Punkte, Kommas, Fragezeichen usw.) hinzu. Ändere nicht die Wörter oder die Reihenfolge der Sätze. Gib NUR den vollständigen Text mit Zeichensetzung zurück.

Originaltext:
"""
${rawText}
"""`;

    const punctuatedText = await generateContentWithRetry(prompt);
    return punctuatedText.trim().replace(/^["']|["']$/g, '');
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
    const { speechLength, topic, sourceLang, mode } = settings;
    const isMonologueMode = mode === 'Vortragsdolmetschen' || mode === 'Simultandolmetschen' || mode === 'Shadowing';
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
            let actionPrompt = `Bitte füge ${paragraphsToAdd} sinnvollen Absatz/Absätze hinzu, um den Text zu verlängern.`;
            if (isMonologueMode) {
                actionPrompt = `Bitte füge ${paragraphsToAdd} sinnvollen Absatz/Absätze in der Mitte des Textes hinzu, um ihn zu verlängern. Die Anrede am Anfang und die Schlussformel am Ende müssen unbedingt erhalten bleiben.`;
            }
            adjustmentPrompt = `Der folgende ${isMonologueMode ? 'Vortrag' : 'Text'} zum Thema "${topic}" in der Sprache ${sourceLang} ist zu kurz. Aktuelle Länge: ${currentText.length} Zeichen. Ziel ist ${target.min}-${target.max} Zeichen. ${actionPrompt} Bleibe dabei unbedingt im Stil und in der Sprache (${sourceLang}) des Originaltextes. Gib NUR den vollständigen, neuen Text aus.
            
            Originaltext:
            """
            ${currentText}
            """`;
        } else { // currentText.length > target.max
            const diff = currentText.length - target.max;
            const paragraphsToRemove = diff > 800 ? 2 : 1;
            let actionPrompt = `Bitte kürze den Text um ${paragraphsToRemove} Absatz/Absätze, ohne den Kerninhalt zu verlieren.`;
            if (isMonologueMode) {
                actionPrompt = `Bitte kürze den Text um ${paragraphsToRemove} Absatz/Absätze aus der Mitte des Textes, ohne den Kerninhalt zu verlieren. Die Anrede am Anfang und die Schlussformel am Ende müssen unbedingt erhalten bleiben.`;
            }
            adjustmentPrompt = `Der folgende ${isMonologueMode ? 'Vortrag' : 'Text'} zum Thema "${topic}" in der Sprache ${sourceLang} ist zu lang. Aktuelle Länge: ${currentText.length} Zeichen. Ziel ist ${target.min}-${target.max} Zeichen. ${actionPrompt} Stelle sicher, dass der gekürzte Text in der Sprache ${sourceLang} bleibt. Gib NUR den vollständigen, gekürzten Text aus.

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
            prompt = `Erstelle einen Vortrag zum Thema "${settings.topic}" für eine Dolmetschübung im Modus "${settings.mode}". Die Sprache des Vortrags soll ${settings.sourceLang} sein. Die Länge soll dem Level "${settings.speechLength}" entsprechen. Der Vortrag muss mit einer passenden Anrede für das Publikum beginnen (z.B. "Sehr geehrte Damen und Herren", "Liebe Freunde", "Verehrte Gäste") und mit einer Schlussformel enden (z.B. "Vielen Dank für Ihre Aufmerksamkeit"). Gib nur den reinen Vortragstext aus, ohne Titel oder zusätzliche Kommentare.`;
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
    const newSettings = { ...settings, [field]: value };

    // Rule 1: If mode is set to Shadowing, force targetLang to match sourceLang
    if (field === 'mode' && value === 'Shadowing') {
        newSettings.targetLang = newSettings.sourceLang;
    } 
    // Rule 2: If sourceLang is changed while in Shadowing mode, force targetLang to match
    else if (field === 'sourceLang' && newSettings.mode === 'Shadowing') {
        newSettings.targetLang = value as Language;
    }
    // Rule 3: Prevent source and target from being the same in non-Shadowing modes
    else if (field === 'sourceLang' && value === newSettings.targetLang && newSettings.mode !== 'Shadowing') {
        const newTargetLang = LANGUAGES.find(l => l !== value) || LANGUAGES[1];
        newSettings.targetLang = newTargetLang;
    } 
    else if (field === 'targetLang' && value === newSettings.sourceLang && newSettings.mode !== 'Shadowing') {
        const newSourceLang = LANGUAGES.find(l => l !== value) || LANGUAGES[0];
        newSettings.sourceLang = newSourceLang;
    }
    setSettings(newSettings);
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
      const isDialogue = settings.mode === 'Gesprächsdolmetschen';

      return (
          <>
              <div className="form-group">
                  <label htmlFor="topic">Thema</label>
                  <input type="text" id="topic" className="form-control" value={settings.topic} onChange={e => handleSettingChange('topic', e.target.value)} />
              </div>
              {isDialogue ? (
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
                <select id="targetLang" className="form-control" value={settings.targetLang} onChange={e => handleSettingChange('targetLang', e.target.value as Language)} disabled={settings.mode === 'Shadowing'}>
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
  const [rawTranscript, setRawTranscript] = useState('');
  const [displayTranscript, setDisplayTranscript] = useState<string | null>(null);
  const [isPunctuating, setIsPunctuating] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [isGeneratingFeedback, setIsGeneratingFeedback] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const recognition = useRef<SpeechRecognition | null>(null);
  const [isEditingOriginalText, setIsEditingOriginalText] = useState(false);
  const [isEditingTranscript, setIsEditingTranscript] = useState(false);

  const targetLang = mode === 'shadowing' ? settings.sourceLang : settings.targetLang;

  useEffect(() => {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognitionAPI) {
        recognition.current = new SpeechRecognitionAPI();
        recognition.current.continuous = true;
        recognition.current.interimResults = true;
        recognition.current.lang = LANG_MAP[targetLang];

        recognition.current.onresult = (event: SpeechRecognitionEvent) => {
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                }
            }
            setRawTranscript(prev => prev + finalTranscript);
        };
        recognition.current.onerror = (event: SpeechRecognitionErrorEvent) => {
            console.error("Speech recognition error", event.error, event.message);
            setIsRecording(false);
        };
        recognition.current.onend = () => {
             if (isRecording) {
                 recognition.current?.start();
             }
        };

    } else {
        console.warn("Speech Recognition API not supported in this browser.");
    }
  }, [targetLang, isRecording]);

  const handleTabChange = async (tab: PracticeAreaTab) => {
    setActiveTab(tab);
    if (tab === 'transcript' && rawTranscript && displayTranscript === null && !isPunctuating) {
        setIsPunctuating(true);
        try {
            const result = await punctuateTextWithAI(rawTranscript, targetLang);
            setDisplayTranscript(result);
        } catch (e) {
            console.error("Punctuation failed:", e);
            setDisplayTranscript(rawTranscript); // Fallback to raw transcript on error
        } finally {
            setIsPunctuating(false);
        }
    }
  };

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
        handleRecord();
    }
  };

  const handleRecord = () => {
    if (isRecording) {
      recognition.current?.stop();
      setIsRecording(false);
    } else {
      setRawTranscript('');
      setDisplayTranscript(null);
      setFeedback(null);
      recognition.current?.start();
      setIsRecording(true);
    }
  };
  
  const handleEditOriginalTextToggle = () => {
    setIsEditingOriginalText(prev => !prev);
  };

  const handleEditTranscriptToggle = () => {
    setIsEditingTranscript(prev => !prev);
  };
  
  const getFeedback = async () => {
      const transcriptForFeedback = displayTranscript ?? rawTranscript;
      if (!transcriptForFeedback) return;
      
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
        ${transcriptForFeedback}
        """

        Aufgabe:
        Analysiere die Verdolmetschung. Gib dein Feedback in zwei Hauptkategorien: "Inhaltliche Richtigkeit" und "Sprachliche Richtigkeit".

        1. Inhaltliche Richtigkeit:
           - Bewerte auf einer Skala von 1 (sehr schlecht) bis 10 (ausgezeichnet), wie genau der Inhalt wiedergegeben wurde. Berücksichtige Auslassungen, Hinzufügungen oder inhaltliche Fehler.
           - Gib eine kurze Zusammenfassung (2-3 Sätze) der inhaltlichen Leistung.

        2. Sprachliche Richtigkeit:
           - Bewerte auf einer Skala von 1 (sehr schlecht) bis 10 (ausgezeichnet), wie korrekt die Zielsprache verwendet wurde. Berücksichtige Grammatik, Terminologie, Stil und Register.
           - Gib eine kurze Zusammenfassung (2-3 Sätze) der sprachlichen Leistung.

        3. Fehleranalyse:
           - Identifiziere bis zu 5 signifikante Fehler oder verbesserungswürdige Stellen.
           - Klassifiziere jeden Fehler als "Inhalt" oder "Sprache".
           - Gib für jeden Fehler den originalen Teil, die Interpretation des Nutzers, einen Korrekturvorschlag und eine kurze Erklärung an.
        
        Gib deine Antwort NUR als JSON-Objekt im folgenden Format aus. Keine zusätzlichen Texte oder Erklärungen.

        {
          "contentRating": number,
          "languageRating": number,
          "contentSummary": "string",
          "languageSummary": "string",
          "errorAnalysis": [
            {
              "original": "string",
              "interpretation": "string",
              "suggestion": "string",
              "explanation": "string",
              "type": "'Inhalt' or 'Sprache'"
            }
          ]
        }
      `;
      try {
          const feedbackText = await generateContentWithRetry(prompt);
          const jsonStart = feedbackText.indexOf('{');
          const jsonEnd = feedbackText.lastIndexOf('}');
          if (jsonStart === -1 || jsonEnd === -1) {
              throw new Error("AI response does not contain a JSON object.");
          }
          const jsonString = feedbackText.substring(jsonStart, jsonEnd + 1);
          const feedbackJson = JSON.parse(jsonString);
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
        <button className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`} onClick={() => handleTabChange('original')}>Originaltext</button>
        <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => handleTabChange('transcript')}>Meine Verdolmetschung</button>
        <button className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} onClick={() => handleTabChange('feedback')} disabled={!rawTranscript}>Feedback</button>
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
                 <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <span className="char-counter">{originalText.length} Zeichen</span>
                    <button className="btn btn-secondary" onClick={handleEditOriginalTextToggle}>
                        {isEditingOriginalText ? 'Speichern' : 'Bearbeiten'}
                    </button>
                 </div>
            </div>
            <div className="text-area">
                <textarea
                    className={`text-area-editor ${isEditingOriginalText ? 'is-editing' : ''}`}
                    value={originalText}
                    onChange={(e) => {
                        setOriginalText(e.target.value);
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
            isPunctuating ? (
                 <div className="loading-overlay" style={{ position: 'relative', background: 'transparent' }}>
                    <div className="spinner"></div>
                    <p>Transkript wird erstellt...</p>
                 </div>
            ) : (
                 <>
                    <div className="controls-bar">
                        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <span className="char-counter">{(displayTranscript ?? '').length} Zeichen</span>
                            <button className="btn btn-secondary" onClick={handleEditTranscriptToggle}>
                                {isEditingTranscript ? 'Speichern' : 'Bearbeiten'}
                            </button>
                        </div>
                    </div>
                    <div className="text-area">
                       <textarea 
                            className={`text-area-editor ${isEditingTranscript ? 'is-editing' : ''}`}
                            value={displayTranscript ?? ''} 
                            onChange={(e) => setDisplayTranscript(e.target.value)} 
                            readOnly={!isEditingTranscript}
                            placeholder="Hier erscheint Ihre Verdolmetschung nach der Aufnahme..." 
                        />
                    </div>
                 </>
            )
        )}
        {activeTab === 'feedback' && (
             <FeedbackDisplay feedback={feedback} isLoading={isGeneratingFeedback} onGenerate={getFeedback} transcriptProvided={!!(displayTranscript ?? rawTranscript)} />
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
    const [processedResults, setProcessedResults] = useState<StructuredDialogueResult[] | null>(null);
    const [isProcessingResults, setIsProcessingResults] = useState(false);
    const [currentTranscript, setCurrentTranscript] = useState('');
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const recognition = useRef<SpeechRecognition | null>(null);
    const [activeTab, setActiveTab] = useState<'practice' | 'results'>('practice');
    const [feedback, setFeedback] = useState<Feedback | null>(null);
    const [isGeneratingFeedback, setIsGeneratingFeedback] = useState(false);

    const currentSegment = dialogue[segmentIndex];
    const targetLang = currentSegment?.lang === settings.sourceLang ? settings.targetLang : settings.sourceLang;
    const isUserTurn = currentSegment?.lang === settings.targetLang;

    useEffect(() => {
        if (segmentIndex === 0 && dialogueState === 'synthesizing') {
            synthesizeAndPlayCurrentSegment();
        }
    }, [segmentIndex, dialogueState, dialogue]);
    
    useEffect(() => {
        if (!currentSegment) return;
        const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognitionAPI) { console.warn("Speech Recognition not supported."); return; }
        const rec = new SpeechRecognitionAPI();
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = LANG_MAP[targetLang];
        rec.onresult = (event: SpeechRecognitionEvent) => {
            let final = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) { final += event.results[i][0].transcript; }
            }
            setCurrentTranscript(prev => prev + final);
        };
        rec.onerror = (e) => console.error("Recognition error:", e);
        rec.onend = () => { if (dialogueState === 'recording') { handleNextSegment(); } };
        recognition.current = rec;
        return () => { rec.stop(); };
    }, [targetLang, dialogueState, currentSegment]);

    useEffect(() => {
        if (dialogueState === 'recording' && recognition.current) {
            setCurrentTranscript('');
            recognition.current.start();
        }
    }, [dialogueState]);
    
     useEffect(() => {
        const processResults = async () => {
            if (dialogueResults.length === 0 || processedResults) return;
            setIsProcessingResults(true);
            try {
                const promises = dialogueResults.map(r => punctuateTextWithAI(r.userInterpretation, r.interpretationLang));
                const punctuatedTexts = await Promise.all(promises);
                const newResults = dialogueResults.map((result, index) => ({
                    ...result,
                    userInterpretation: punctuatedTexts[index]
                }));
                setProcessedResults(newResults);
            } catch (error)
            {
                console.error("Failed to punctuate dialogue results:", error);
                setProcessedResults(dialogueResults); // Fallback to raw results
            } finally {
                setIsProcessingResults(false);
            }
        };

        if (activeTab === 'results') {
            processResults();
        }
    }, [activeTab, dialogueResults, processedResults]);

    const handleUpdateResult = (index: number, newText: string) => {
        setProcessedResults(prev => {
            if (!prev) return null;
            const updated = [...prev];
            updated[index] = { ...updated[index], userInterpretation: newText };
            return updated;
        });
    };

    const synthesizeAndPlayCurrentSegment = async () => {
        if (!currentSegment) return;
        setDialogueState('synthesizing');
        try {
            const audioContent = await synthesizeSpeechGoogleCloud(currentSegment.text, currentSegment.lang, settings.voiceQuality);
            const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
            const url = URL.createObjectURL(audioBlob);
            audioRef.current = new Audio(url);
            audioRef.current.onended = () => {
                 if (isUserTurn) { setDialogueState('waiting_for_record'); } 
                 else { setDialogueState('recording'); }
            };
            setDialogueState('playing');
            audioRef.current.play();
        } catch (error) { console.error("Speech synthesis failed:", error); }
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
            setDialogueState('synthesizing');
            synthesizeAndPlayCurrentSegment();
        } else {
            setDialogueState('finished');
            setActiveTab('results');
        }
    };
    
    const handleRecordClick = () => {
        if (dialogueState === 'recording') {
             if (recognition.current) { recognition.current.stop(); }
        } else if (dialogueState === 'waiting_for_record') {
            setDialogueState('recording');
        }
    };
    
    const getFeedbackForDialogue = async () => {
        const resultsForFeedback = processedResults || dialogueResults;
        if (!resultsForFeedback || resultsForFeedback.length === 0) return;
        
        setIsGeneratingFeedback(true);
        setFeedback(null);

        const resultsText = resultsForFeedback.map((r, i) => `
            --- Segment ${i + 1} ---
            Original (${r.originalSegment.lang}): "${r.originalSegment.text}"
            Verdolmetschung des Benutzers (${r.interpretationLang}): "${r.userInterpretation}"
        `).join('');

        const prompt = `
            Kontext: Eine Gesprächsdolmetsch-Übung. Person A spricht ${settings.sourceLang}, Person B spricht ${settings.targetLang}. Der Benutzer dolmetscht in die jeweils andere Sprache.

            Dialogverlauf und Verdolmetschung:
            ${resultsText}

            Aufgabe:
            Analysiere die GESAMTE Verdolmetschungsleistung des Benutzers über alle Segmente hinweg. Gib dein Feedback in zwei Hauptkategorien: "Inhaltliche Richtigkeit" und "Sprachliche Richtigkeit".

            1. Inhaltliche Richtigkeit:
               - Bewerte auf einer Skala von 1 (sehr schlecht) bis 10 (ausgezeichnet), wie genau der Inhalt wiedergegeben wurde.
               - Gib eine kurze Zusammenfassung (2-3 Sätze) der inhaltlichen Leistung.

            2. Sprachliche Richtigkeit:
               - Bewerte auf einer Skala von 1 (sehr schlecht) bis 10 (ausgezeichnet), wie korrekt die Zielsprache verwendet wurde (Grammatik, Terminologie, Stil).
               - Gib eine kurze Zusammenfassung (2-3 Sätze) der sprachlichen Leistung.

            3. Fehleranalyse:
               - Identifiziere bis zu 5 signifikante Fehler. Klassifiziere jeden als "Inhalt" oder "Sprache".
               - Gib für jeden Fehler den originalen Teil, die Interpretation des Nutzers, einen Korrekturvorschlag und eine kurze Erklärung an.
            
            Gib deine Antwort NUR als JSON-Objekt im folgenden Format aus:
            {
              "contentRating": number, "languageRating": number, "contentSummary": "string", "languageSummary": "string",
              "errorAnalysis": [{"original": "string", "interpretation": "string", "suggestion": "string", "explanation": "string", "type": "'Inhalt' or 'Sprache'"}]
            }
        `;
        try {
            const feedbackText = await generateContentWithRetry(prompt);
            const jsonStart = feedbackText.indexOf('{');
            const jsonEnd = feedbackText.lastIndexOf('}');
            if (jsonStart === -1 || jsonEnd === -1) {
                throw new Error("AI response does not contain a JSON object.");
            }
            const jsonString = feedbackText.substring(jsonStart, jsonEnd + 1);
            const feedbackJson = JSON.parse(jsonString);
            setFeedback(feedbackJson);
        } catch (error) {
            console.error("Error getting feedback for dialogue:", error);
        } finally {
            setIsGeneratingFeedback(false);
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
                <div className="tab-content" style={{gap: '1.5rem'}}>
                    {isProcessingResults ? (
                        <div className="loading-overlay" style={{ position: 'relative', background: 'transparent' }}>
                            <div className="spinner"></div>
                            <p>Ergebnisse werden aufbereitet...</p>
                        </div>
                    ) : (
                        <StructuredTranscript results={processedResults || []} onUpdateResult={handleUpdateResult} />
                    )}
                    <FeedbackDisplay
                        feedback={feedback}
                        isLoading={isGeneratingFeedback}
                        onGenerate={getFeedbackForDialogue}
                        transcriptProvided={(processedResults || dialogueResults).length > 0}
                    />
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
                     <p className="segment-text">{currentTranscript ? `"${currentTranscript}"` : "..."}</p>
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
    const [processedResults, setProcessedResults] = useState<StructuredDialogueResult[] | null>(null);
    const [isProcessingResults, setIsProcessingResults] = useState(false);
    const [currentTranscript, setCurrentTranscript] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const recognition = useRef<SpeechRecognition | null>(null);
    const [activeTab, setActiveTab] = useState<'practice' | 'results'>('practice');
    const [feedback, setFeedback] = useState<Feedback | null>(null);
    const [isGeneratingFeedback, setIsGeneratingFeedback] = useState(false);

    const currentSegment = editableDialogue[segmentIndex];
    const targetLang = currentSegment?.lang === settings.sourceLang ? settings.targetLang : settings.sourceLang;

    const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const newText = e.target.value;
        const updatedDialogue = [...editableDialogue];
        updatedDialogue[segmentIndex] = { ...updatedDialogue[segmentIndex], text: newText };
        setEditableDialogue(updatedDialogue);
    };

    const handleEditToggle = () => setIsEditing(prev => !prev);

    useEffect(() => {
        if (!currentSegment) return;
        const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognitionAPI) { console.warn("Speech Recognition not supported."); return; }
        const rec = new SpeechRecognitionAPI();
        rec.continuous = true;
        rec.interimResults = false;
        rec.lang = LANG_MAP[targetLang];
        rec.onresult = (event: SpeechRecognitionEvent) => {
            let final = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) { final += event.results[i][0].transcript; }
            }
            setCurrentTranscript(prev => prev + final);
        };
        rec.onerror = (e) => { console.error("Recognition error:", e); setIsRecording(false); };
        rec.onend = () => { if (isRecordingRef.current) { rec.start(); } };
        recognition.current = rec;
        return () => { isRecordingRef.current = false; rec.stop(); };
    }, [targetLang, currentSegment]);
    
    const isRecordingRef = useRef(isRecording);
    useEffect(() => { isRecordingRef.current = isRecording; }, [isRecording]);

     useEffect(() => {
        const processResults = async () => {
            if (dialogueResults.length === 0 || processedResults) return;
            setIsProcessingResults(true);
            try {
                const promises = dialogueResults.map(r => punctuateTextWithAI(r.userInterpretation, r.interpretationLang));
                const punctuatedTexts = await Promise.all(promises);
                const newResults = dialogueResults.map((result, index) => ({ ...result, userInterpretation: punctuatedTexts[index] }));
                setProcessedResults(newResults);
            } catch (error) {
                console.error("Failed to punctuate sight translation results:", error);
                setProcessedResults(dialogueResults); // Fallback
            } finally {
                setIsProcessingResults(false);
            }
        };

        if (activeTab === 'results') {
            processResults();
        }
    }, [activeTab, dialogueResults, processedResults]);

    const handleUpdateResult = (index: number, newText: string) => {
        setProcessedResults(prev => {
            if (!prev) return null;
            const updated = [...prev];
            updated[index] = { ...updated[index], userInterpretation: newText };
            return updated;
        });
    };

    const handleNextSegment = () => {
        if (isRecording) { handleRecordClick(); }
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
           isRecordingRef.current = false;
           recognition.current?.stop();
           setIsRecording(false);
       } else {
            setCurrentTranscript('');
            recognition.current?.start();
            setIsRecording(true);
       }
    };
    
    const getFeedbackForSightTranslation = async () => {
        const resultsForFeedback = processedResults || dialogueResults;
        if (!resultsForFeedback || resultsForFeedback.length === 0) return;
        
        setIsGeneratingFeedback(true);
        setFeedback(null);

        const resultsText = resultsForFeedback.map((r, i) => `
            --- Segment ${i + 1} ---
            Original (${r.originalSegment.lang}): "${r.originalSegment.text}"
            Verdolmetschung des Benutzers (${r.interpretationLang}): "${r.userInterpretation}"
        `).join('');

        const prompt = `
            Kontext: Eine Stegreifübersetzungs-Übung von ${settings.sourceLang} nach ${targetLang}.

            Originaltext und Verdolmetschung:
            ${resultsText}

            Aufgabe:
            Analysiere die Verdolmetschungsleistung. Gib dein Feedback in zwei Hauptkategorien: "Inhaltliche Richtigkeit" und "Sprachliche Richtigkeit".

            1. Inhaltliche Richtigkeit:
               - Bewerte auf einer Skala von 1 (sehr schlecht) bis 10 (ausgezeichnet), wie genau der Inhalt wiedergegeben wurde.
               - Gib eine kurze Zusammenfassung (2-3 Sätze) der inhaltlichen Leistung.

            2. Sprachliche Richtigkeit:
               - Bewerte auf einer Skala von 1 (sehr schlecht) bis 10 (ausgezeichnet), wie korrekt die Zielsprache verwendet wurde (Grammatik, Terminologie, Stil).
               - Gib eine kurze Zusammenfassung (2-3 Sätze) der sprachlichen Leistung.

            3. Fehleranalyse:
               - Identifiziere bis zu 5 signifikante Fehler. Klassifiziere jeden als "Inhalt" oder "Sprache".
               - Gib für jeden Fehler den originalen Teil, die Interpretation des Nutzers, einen Korrekturvorschlag und eine kurze Erklärung an.
            
            Gib deine Antwort NUR als JSON-Objekt im folgenden Format aus:
            {
              "contentRating": number, "languageRating": number, "contentSummary": "string", "languageSummary": "string",
              "errorAnalysis": [{"original": "string", "interpretation": "string", "suggestion": "string", "explanation": "string", "type": "'Inhalt' or 'Sprache'"}]
            }
        `;
        try {
            const feedbackText = await generateContentWithRetry(prompt);
            const jsonStart = feedbackText.indexOf('{');
            const jsonEnd = feedbackText.lastIndexOf('}');
            if (jsonStart === -1 || jsonEnd === -1) {
                throw new Error("AI response does not contain a JSON object.");
            }
            const jsonString = feedbackText.substring(jsonStart, jsonEnd + 1);
            const feedbackJson = JSON.parse(jsonString);
            setFeedback(feedbackJson);
        } catch (error) {
            console.error("Error getting feedback for sight translation:", error);
        } finally {
            setIsGeneratingFeedback(false);
        }
    };

    const renderResults = () => (
        <div className="panel practice-area">
            <div className="tabs">
                <button className="tab-btn" onClick={() => setActiveTab('practice')}>Übung</button>
                <button className="tab-btn active">Ergebnisse</button>
            </div>
            <div className="tab-content" style={{gap: '1.5rem'}}>
                {isProcessingResults ? (
                    <div className="loading-overlay" style={{ position: 'relative', background: 'transparent' }}>
                        <div className="spinner"></div>
                        <p>Ergebnisse werden aufbereitet...</p>
                    </div>
                ) : (
                    <StructuredTranscript results={processedResults || dialogueResults} onUpdateResult={handleUpdateResult} />
                )}
                <FeedbackDisplay
                    feedback={feedback}
                    isLoading={isGeneratingFeedback}
                    onGenerate={getFeedbackForSightTranslation}
                    transcriptProvided={(processedResults || dialogueResults).length > 0}
                />
            </div>
        </div>
    );

    if (!currentSegment || activeTab === 'results') {
        return renderResults();
    }

    return (
        <div className="panel practice-area dialogue-practice-container">
            <div className="dialogue-status">
                Segment {segmentIndex + 1} / {dialogue.length} ({currentSegment.lang} {'->'} {targetLang})
                <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <span className="char-counter">{currentSegment.text.length} Zeichen</span>
                    <button className="btn btn-secondary" onClick={handleEditToggle} style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}>
                        {isEditing ? 'Speichern' : 'Bearbeiten'}
                    </button>
                </div>
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
                <p>{currentTranscript ? `"${currentTranscript}"` : "..."}</p>
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

const StarRating = ({ score, maxScore = 10 }: { score: number, maxScore?: number }) => (
    <span className="star-rating">
      {[...Array(maxScore)].map((_, i) => (
        <span key={i} className={`star ${i < score ? 'filled' : ''}`}>★</span>
      ))}
    </span>
);

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
      <div className="placeholder" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.5rem', marginTop: '1.5rem' }}>
        <h2>Feedback</h2>
        <p>Nachdem Sie eine Verdolmetschung aufgenommen haben, können Sie hier eine detaillierte KI-Analyse anfordern.</p>
        <button className="btn btn-primary btn-large" onClick={onGenerate} disabled={!transcriptProvided}>
          Feedback generieren
        </button>
      </div>
    );
  }

  return (
    <div className="feedback-content">
        <h3>Bewertung</h3>
        <div className="ratings-container">
            <div className="rating-item">
                <div className="rating-header">
                    <h4>Inhaltliche Richtigkeit</h4>
                    <span><StarRating score={feedback.contentRating} maxScore={10} /> ({feedback.contentRating}/10)</span>
                </div>
                <p className="rating-summary">{feedback.contentSummary}</p>
            </div>
            <div className="rating-item">
                <div className="rating-header">
                    <h4>Sprachliche Richtigkeit</h4>
                    <span><StarRating score={feedback.languageRating} maxScore={10} /> ({feedback.languageRating}/10)</span>
                </div>
                <p className="rating-summary">{feedback.languageSummary}</p>
            </div>
        </div>

        <h3>Fehleranalyse</h3>
        <ul className="error-analysis-list">
            {feedback.errorAnalysis.length > 0 ? feedback.errorAnalysis.map((item, index) => (
                <li key={index}>
                    <p>
                      <span className={`error-type ${item.type.toLowerCase() === 'inhalt' ? 'inhalt' : 'sprache'}`}>{item.type}</span>
                      <strong>Original:</strong> "{item.original}"
                    </p>
                    <p><strong>Ihre Version:</strong> "{item.interpretation}"</p>
                    <p><strong>Vorschlag:</strong> "{item.suggestion}"</p>
                    <p><strong>Erklärung:</strong> {item.explanation}</p>
                </li>
            )) : <p>Keine spezifischen Fehler gefunden. Gute Arbeit!</p>}
        </ul>
        <button className="btn btn-secondary" onClick={onGenerate} style={{marginTop: '1rem'}}>
          Feedback erneut generieren
        </button>
    </div>
  );
};

const StructuredTranscript = ({ results, onUpdateResult }: { results: StructuredDialogueResult[]; onUpdateResult: (index: number, newText: string) => void; }) => {
    const [editingState, setEditingState] = useState<{ index: number | null; text: string }>({ index: null, text: '' });

    if (results.length === 0) {
        return <p>Noch keine Ergebnisse vorhanden.</p>;
    }

    const handleEdit = (index: number, currentText: string) => {
        setEditingState({ index, text: currentText });
    };

    const handleSave = () => {
        if (editingState.index !== null) {
            onUpdateResult(editingState.index, editingState.text);
            setEditingState({ index: null, text: '' });
        }
    };

    const handleCancel = () => {
        setEditingState({ index: null, text: '' });
    };

    const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setEditingState(prev => ({ ...prev, text: e.target.value }));
    };

    return (
        <div className="structured-transcript text-area">
            {results.map((result, index) => (
                <div key={index} className="transcript-segment">
                     <div className="transcript-segment-header">
                        <h4>Segment {index + 1}: {result.originalSegment.type} ({result.originalSegment.lang})</h4>
                        {editingState.index === index ? (
                            <>
                                <span className="char-counter">{editingState.text.length} Zeichen</span>
                                <button className="btn btn-secondary" onClick={handleSave} style={{padding: '0.25rem 0.5rem', fontSize: '0.8rem'}}>Speichern</button>
                                <button className="btn btn-secondary" onClick={handleCancel} style={{padding: '0.25rem 0.5rem', fontSize: '0.8rem'}}>Abbrechen</button>
                            </>
                        ) : (
                            <>
                                <span className="char-counter">{(result.userInterpretation || '').length} Zeichen</span>
                                <button className="btn btn-secondary" onClick={() => handleEdit(index, result.userInterpretation)} style={{padding: '0.25rem 0.5rem', fontSize: '0.8rem'}}>Bearbeiten</button>
                            </>
                        )}
                    </div>
                    <p className="transcript-segment-original">{result.originalSegment.text}</p>
                    <div className="transcript-segment-user">
                        <strong>Ihre Verdolmetschung ({result.interpretationLang}):</strong>
                        {editingState.index === index ? (
                             <textarea
                                className="text-area-editor is-editing"
                                value={editingState.text}
                                onChange={handleTextChange}
                                style={{ width: '100%', minHeight: '80px', marginTop: '0.5rem' }}
                            />
                        ) : (
                             <p style={{whiteSpace: 'pre-wrap', marginTop: '0.5rem'}}>
                                {result.userInterpretation || <em>Keine Aufnahme für dieses Segment.</em>}
                             </p>
                        )}
                    </div>
                </div>
            ))}
        </div>
    );
}

const root = createRoot(document.getElementById('root')!);
root.render(<App />);