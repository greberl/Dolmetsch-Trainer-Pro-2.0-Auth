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
type DialogueState = 'idle' | 'ready' | 'synthesizing' | 'playing' | 'waiting_for_record' | 'recording' | 'finished' | 'starting';
type PracticeAreaTab = 'original' | 'transcript' | 'feedback';


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
  const [dialogueFinished, setDialogueFinished] = useState(false);
  
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
      setDialogueFinished(false);
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

  const handleRecordingFinished = useCallback(async (rawTranscript: string) => {
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
        if (!ai) throw new Error("AI client not initialized");
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
  }, [ai]);

  const getFeedback = useCallback(async () => {
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
        if (!ai) throw new Error("AI client not initialized");
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

      } catch (err) {
          console.error(err);
          setError("Fehler bei der Generierung des Feedbacks. Bitte versuchen Sie es erneut.");
      } finally {
          setIsLoading(false);
      }
  }, [ai, userTranscript, originalText, settings]);

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
         {settings.mode === 'Gesprächsdolmetschen' && dialogueFinished ? (
              <DialogueResults
                originalText={originalText}
                userTranscript={userTranscript}
                feedback={feedback}
                getFeedback={getFeedback}
                isLoading={isLoading}
                loadingMessage={loadingMessage}
                error={error}
              />
            ) : (
              <PracticeArea
                  key={originalText} // Remount component when text changes
                  isLoading={isLoading} 
                  loadingMessage={loadingMessage}
                  originalText={originalText}
                  onRecordingFinished={handleRecordingFinished}
                  getFeedback={getFeedback}
                  userTranscript={userTranscript}
                  feedback={feedback}
                  error={error}
                  settings={settings}
                  exerciseStarted={exerciseStarted}
                  onPremiumVoiceAuthError={handlePremiumVoiceAuthError}
                  onDialogueFinished={() => setDialogueFinished(true)}
              />
            )}
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

        if (lastPunctuation === -1) {
            lastPunctuation = text.lastIndexOf(' ', chunkEnd);
        }

        if (lastPunctuation === -1 || lastPunctuation <= i) {
            chunkEnd = i + maxLength;
        } else {
            chunkEnd = lastPunctuation + 1;
        }
        
        chunks.push(text.substring(i, chunkEnd));
        i = chunkEnd;
    }
    return chunks;
};

const FeedbackDisplay = ({ feedback, getFeedback, userTranscript }: { feedback: Feedback | null, getFeedback: () => void, userTranscript: string }) => {
    if (!feedback) {
      return (
        <div className="controls-bar">
          <button onClick={getFeedback} disabled={!userTranscript.trim()}>Feedback erhalten</button>
          <p>{userTranscript ? '' : 'Noch kein Transkript vorhanden. Erstellen Sie eine Aufnahme und klicken Sie auf "Feedback erhalten".'}</p>
        </div>
      );
    }

    const renderStars = (rating: number) => {
        return Array.from({ length: 10 }, (_, i) => (
            <span key={i} className={`star ${i < rating ? 'filled' : ''}`}>&#9733;</span>
        ));
    };

    return (
      <div className="feedback-content">
        <h4>Zusammenfassung</h4>
        <p>{feedback.summary}</p>
        <h4>Bewertungen</h4>
        <table className="ratings-table">
          <tbody>
            <tr>
              <td>Inhalt</td>
              <td>{renderStars(feedback.ratings.content)}</td>
              <td>{feedback.ratings.content}/10</td>
            </tr>
            <tr>
              <td>Ausdruck</td>
              <td>{renderStars(feedback.ratings.expression)}</td>
              <td>{feedback.ratings.expression}/10</td>
            </tr>
             <tr>
              <td>Terminologie</td>
              <td>{renderStars(feedback.ratings.terminology)}</td>
              <td>{feedback.ratings.terminology}/10</td>
            </tr>
          </tbody>
        </table>
        {feedback.errorAnalysis.length > 0 && (
          <>
            <h4>Fehleranalyse</h4>
            <ul className="error-analysis-list">
              {feedback.errorAnalysis.map((item, index) => (
                <li key={index}>
                  <p><strong>Original:</strong> {item.original}</p>
                  <p><strong>Ihre Version:</strong> {item.interpretation}</p>
                  <p><strong>Vorschlag:</strong> {item.suggestion}</p>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    );
};

const DialogueResults = ({ originalText, userTranscript, feedback, getFeedback, isLoading, loadingMessage, error }: {
    originalText: string;
    userTranscript: string;
    feedback: Feedback | null;
    getFeedback: () => void;
    isLoading: boolean;
    loadingMessage: string;
    error: string | null;
}) => {
    const [activeTab, setActiveTab] = useState<PracticeAreaTab>('transcript');

    useEffect(() => {
        if(userTranscript && !feedback) {
            setActiveTab('transcript');
        }
    }, [userTranscript, feedback]);
    
    if (isLoading) {
        return (
            <section className="panel practice-area">
                <div className="loading-overlay">
                    <div className="spinner"></div>
                    <p>{loadingMessage}</p>
                </div>
            </section>
        );
    }

    return (
        <section className="panel practice-area">
            <div className="dialogue-results-wrapper">
                {error && <div className="error-banner">{error}</div>}
                <div className="tabs">
                    <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>Ihr Transkript</button>
                    <button className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} onClick={() => setActiveTab('feedback')}>Feedback</button>
                    <button className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`} onClick={() => setActiveTab('original')}>Originalskript</button>
                </div>
                <div className="tab-content">
                    {activeTab === 'original' && (
                        <div className="text-area"><p>{originalText}</p></div>
                    )}
                    {activeTab === 'transcript' && (
                        <div className="text-area"><p>{userTranscript || "Kein Transkript generiert."}</p></div>
                    )}
                    {activeTab === 'feedback' && (
                        <div className="text-area">
                           <FeedbackDisplay feedback={feedback} getFeedback={getFeedback} userTranscript={userTranscript} />
                        </div>
                    )}
                </div>
                 <footer className="practice-footer">
                     <p className="form-text-hint">Übung abgeschlossen. Starten Sie eine neue Übung über das Einstellungsmenü.</p>
                </footer>
            </div>
        </section>
    );
};


const PracticeArea = ({
  isLoading, loadingMessage, originalText, onRecordingFinished, getFeedback,
  userTranscript, feedback, error, settings, exerciseStarted,
  onPremiumVoiceAuthError, onDialogueFinished,
}: {
  isLoading: boolean; loadingMessage: string; originalText: string;
  onRecordingFinished: (transcript: string) => void; getFeedback: () => void;
  userTranscript: string; feedback: Feedback | null; error: string | null; settings: Settings;
  exerciseStarted: boolean; onPremiumVoiceAuthError: () => void;
  onDialogueFinished: () => void;
}) => {
  const [activeTab, setActiveTab] = useState<PracticeAreaTab>('original');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  // State for Dialogue Interpreting
  const [dialogueSegments, setDialogueSegments] = useState<DialogueSegment[]>([]);
  const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);
  const [dialogueState, setDialogueState] = useState<DialogueState>('idle');
  const [isSegmentTextVisible, setIsSegmentTextVisible] = useState(false);
  const dialogueTranscriptsRef = useRef<string[]>([]);

  const stopPlayback = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    window.speechSynthesis.cancel();
    setIsPlaying(false);
  }, []);

  const playText = useCallback(async (textToPlay: string, lang: Language, onEndedCallback?: () => void) => {
    stopPlayback();
    setIsPlaying(true);

    if (settings.voiceQuality === 'Premium') {
      try {
        const audioContent = await synthesizeSpeechGoogleCloud(textToPlay, lang, process.env.API_KEY || '');
        const audio = new Audio(`data:audio/mp3;base64,${audioContent}`);
        audioRef.current = audio;
        audio.play();
        audio.onended = () => {
            setIsPlaying(false);
            if (onEndedCallback) onEndedCallback();
        };
      } catch (e) {
        if (e instanceof TtsAuthError) {
          onPremiumVoiceAuthError();
        }
        console.error("Fehler bei der Premium-Sprachsynthese:", e);
        setIsPlaying(false);
        if (onEndedCallback) onEndedCallback();
      }
    } else { // Standard voice
      const chunks = splitTextIntoChunks(textToPlay);
      let currentChunk = 0;

      const speakChunk = () => {
        if (currentChunk < chunks.length && window.speechSynthesis) {
          const utterance = new SpeechSynthesisUtterance(chunks[currentChunk]);
          utterance.lang = LANGUAGE_CODES[lang];
          utterance.onend = () => {
            currentChunk++;
            if (currentChunk < chunks.length) {
              speakChunk();
            } else {
              setIsPlaying(false);
              if (onEndedCallback) onEndedCallback();
            }
          };
          window.speechSynthesis.speak(utterance);
        } else {
            setIsPlaying(false);
            if (onEndedCallback) onEndedCallback();
        }
      };
      
      const utterance = new SpeechSynthesisUtterance(chunks[0]);
      utterance.lang = LANGUAGE_CODES[lang];
      utterance.onend = () => {
        currentChunk++;
        if (currentChunk < chunks.length) {
            speakChunk();
        } else {
            setIsPlaying(false);
            if(onEndedCallback) onEndedCallback();
        }
      };
      window.speechSynthesis.speak(utterance);
    }
  }, [settings.voiceQuality, stopPlayback, onPremiumVoiceAuthError]);
  
  const startRecording = useCallback((lang: Language, onRecognitionEnd: (transcript: string) => void) => {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) {
        alert("Ihr Browser unterstützt die Web Speech API nicht. Bitte versuchen Sie es mit Chrome.");
        return;
    }
    recognitionRef.current = new SpeechRecognitionAPI();
    const recognition = recognitionRef.current;
    recognition.lang = LANGUAGE_CODES[lang];
    recognition.continuous = true;
    recognition.interimResults = false;
    
    let finalTranscript = '';
    recognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        }
      }
    };

    recognition.onstart = () => {
        setIsRecording(true);
        if (settings.mode === 'Gesprächsdolmetschen') setDialogueState('recording');
    }
    recognition.onend = () => {
      setIsRecording(false);
      onRecognitionEnd(finalTranscript.trim());
      recognitionRef.current = null;
    };
    recognition.onerror = (event) => {
      console.error('Speech recognition error', event);
      setIsRecording(false);
       if (settings.mode === 'Gesprächsdolmetschen') setDialogueState('waiting_for_record');
    };

    recognition.start();
  }, [settings.mode]);

  const stopRecording = useCallback(() => {
      recognitionRef.current?.stop();
  }, []);

  // --- Effect and Logic for Dialogue Interpreting ---
  useEffect(() => {
    if (settings.mode === 'Gesprächsdolmetschen' && originalText) {
        const lines = originalText.split('\n').filter(line => line.trim() !== '');
        const segments: DialogueSegment[] = lines.map(line => {
            const isQuestion = line.startsWith('Frage');
            return {
                type: isQuestion ? 'Frage' : 'Antwort',
                text: line.replace(/^(Frage|Antwort)\s*\d*:\s*/, ''),
                lang: isQuestion ? settings.sourceLang : settings.targetLang
            };
        });
        setDialogueSegments(segments);
        setCurrentSegmentIndex(0);
        dialogueTranscriptsRef.current = [];
        setDialogueState('ready'); 
    }
  }, [originalText, settings.mode, settings.sourceLang, settings.targetLang]);
  
  const advanceToNextSegment = useCallback((lastTranscript: string) => {
    setIsSegmentTextVisible(false);
    dialogueTranscriptsRef.current.push(lastTranscript);
    const nextIndex = currentSegmentIndex + 1;

    if (nextIndex >= dialogueSegments.length) {
      setDialogueState('finished');
      onRecordingFinished(dialogueTranscriptsRef.current.join(' '));
      onDialogueFinished();
      return;
    }

    setCurrentSegmentIndex(nextIndex);
    const nextSegment = dialogueSegments[nextIndex];
    setDialogueState('synthesizing');
    playText(nextSegment.text, nextSegment.lang, () => {
      setDialogueState('waiting_for_record');
    });
  }, [currentSegmentIndex, dialogueSegments, onDialogueFinished, onRecordingFinished, playText]);

  const startDialogue = useCallback(() => {
    if (dialogueSegments.length > 0) {
      setIsSegmentTextVisible(false);
      const firstSegment = dialogueSegments[0];
      setDialogueState('synthesizing');
      playText(firstSegment.text, firstSegment.lang, () => {
        setDialogueState('waiting_for_record');
      });
    }
  }, [dialogueSegments, playText]);


  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPlayback();
      stopRecording();
    };
  }, [stopPlayback, stopRecording]);

  if (isLoading) {
    return (
      <section className="panel practice-area">
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>{loadingMessage}</p>
        </div>
      </section>
    );
  }

  if (!exerciseStarted) {
    return (
      <section className="panel practice-area">
        <div className="placeholder">
          <h2>Willkommen beim Dolmetsch-Trainer Pro</h2>
          <p>Passen Sie die Einstellungen auf der linken Seite an und klicken Sie auf "Loslegen", um eine neue Übung zu starten.</p>
        </div>
      </section>
    );
  }

  if (settings.mode === 'Gesprächsdolmetschen') {
    const segment = dialogueSegments[currentSegmentIndex];
    const segmentNumber = Math.floor(currentSegmentIndex / 2) + 1;
    const canRecord = dialogueState === 'waiting_for_record';

    let statusText = '';
    let segmentInfo = '';
    
    if (segment) {
        segmentInfo = `${segment.type} ${segmentNumber} / ${dialogueSegments.length / 2}`;
    }

    switch(dialogueState) {
        case 'ready':
            statusText = 'Die Dialogübung ist bereit.';
            break;
        case 'synthesizing':
            statusText = `Spielt ${segmentInfo}...`;
            break;
        case 'waiting_for_record':
            statusText = `Sie sind dran. Bitte dolmetschen Sie ${segmentInfo}.`;
            break;
        case 'recording':
            statusText = `Aufnahme für ${segmentInfo} läuft...`;
            break;
        case 'finished':
            statusText = 'Übung abgeschlossen. Transkript wird verarbeitet.';
            break;
        default:
             statusText = 'Übung wird vorbereitet...';
    }
    
    const recordingLang = segment?.lang === settings.sourceLang ? settings.targetLang : settings.sourceLang;

    return (
         <section className="panel practice-area">
            <div className="dialogue-practice-container">
                <div className="dialogue-status">{statusText}</div>
                <div className="current-segment-display">
                    {dialogueState === 'ready' && (
                        <button className="btn btn-primary" onClick={startDialogue}>Dialog starten</button>
                    )}
                    
                    {['synthesizing', 'waiting_for_record', 'recording'].includes(dialogueState) && segment && (
                        <div className="dialogue-text-container">
                             {isSegmentTextVisible ? (
                                <p className="segment-text">{segment.text}</p>
                            ) : (
                                <p className="segment-text-hidden">
                                    {dialogueState === 'synthesizing' 
                                        ? 'Bitte hören Sie aufmerksam zu...' 
                                        : 'Sie sind an der Reihe zu dolmetschen...'}
                                </p>
                            )}
                            <button className="btn btn-secondary btn-show-text" onClick={() => setIsSegmentTextVisible(p => !p)}>
                                {isSegmentTextVisible ? 'Text ausblenden' : 'Text anzeigen'}
                            </button>
                        </div>
                    )}
                    
                     {dialogueState === 'finished' && (
                         <p className="segment-text-hidden">Übung abgeschlossen</p>
                    )}
                </div>
                <footer className="practice-footer">
                     <p className="recording-status-text">
                        {isRecording
                            ? 'Aufnahme läuft...'
                            : canRecord
                            ? 'Klicken Sie zum Starten der Aufnahme'
                            : ''
                        }
                    </p>
                    <button 
                        className={`btn-record ${isRecording ? 'recording' : ''}`} 
                        onClick={() => {
                            if (isRecording) {
                                stopRecording();
                            } else if (canRecord) {
                                startRecording(recordingLang, (transcript) => advanceToNextSegment(transcript));
                            }
                        }}
                        disabled={!canRecord && !isRecording}
                        aria-label={isRecording ? 'Aufnahme stoppen' : 'Aufnahme starten'}
                    >
                        <span className="mic-icon"></span>
                    </button>
                </footer>
            </div>
         </section>
    );
  }
  
  // Default view for other modes
  return (
    <section className="panel practice-area">
      {error && <div className="error-banner">{error}</div>}
      <div className="tabs">
        <button className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`} onClick={() => setActiveTab('original')}>Originaltext</button>
        <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>Ihr Transkript</button>
        <button className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} onClick={() => setActiveTab('feedback')}>Feedback</button>
      </div>

      <div className="tab-content">
        {activeTab === 'original' && (
          <>
            <div className="controls-bar">
              <button onClick={() => playText(originalText, settings.sourceLang)} disabled={isPlaying || isRecording}>
                {isPlaying ? 'Spielt...' : '▶ Abspielen'}
              </button>
              <button onClick={stopPlayback} disabled={!isPlaying}>■ Stopp</button>
            </div>
            <div className="text-area">
              <p>{originalText}</p>
            </div>
          </>
        )}
        {activeTab === 'transcript' && (
          <>
            <div className="controls-bar">
              <button onClick={getFeedback} disabled={!userTranscript.trim() || isRecording}>Feedback erhalten</button>
            </div>
             <div className="text-area">
              <p>{userTranscript || "Noch kein Transkript vorhanden. Machen Sie eine Aufnahme."}</p>
            </div>
          </>
        )}
        {activeTab === 'feedback' && (
           <div className="text-area">
               <FeedbackDisplay feedback={feedback} getFeedback={getFeedback} userTranscript={userTranscript} />
           </div>
        )}
      </div>

      <footer className="practice-footer">
        <p className="recording-status-text">
            {isRecording
                ? 'Aufnahme läuft...'
                : isPlaying
                ? 'Wiedergabe läuft...'
                : 'Klicken Sie zum Starten der Aufnahme'
            }
        </p>
        <button 
          className={`btn-record ${isRecording ? 'recording' : ''}`} 
          onClick={() => {
              if (isRecording) {
                stopRecording();
              } else {
                startRecording(settings.targetLang, onRecordingFinished);
              }
          }}
          disabled={isPlaying}
          aria-label={isRecording ? 'Aufnahme stoppen' : 'Aufnahme starten'}
        >
          <span className="mic-icon"></span>
        </button>
      </footer>
    </section>
  );
};

const container = document.getElementById('root');
if (container) {
    const root = createRoot(container);
    root.render(
      <React.StrictMode>
        <App />
      </React.StrictMode>
    );
}