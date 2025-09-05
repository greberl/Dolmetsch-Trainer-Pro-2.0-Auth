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
  const [structuredDialogueResults, setStructuredDialogueResults] = useState<StructuredDialogueResult[] | null>(null);
  
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
      setStructuredDialogueResults(null);
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
  
  const handleTranscriptChange = (newTranscript: string) => {
    setUserTranscript(newTranscript);
  };

  const getFeedback = useCallback(async (textToCompare?: string) => {
      if (!userTranscript) {
          setError("Kein Transkript vorhanden, um Feedback zu erhalten.");
          return;
      }
      setIsLoading(true);
      setLoadingMessage('Analysiere und generiere Feedback...');
      setError(null);
      setFeedback(null);

      const textForFeedback = textToCompare ?? originalText;

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
Originaltext (${settings.sourceLang}): "${textForFeedback}"
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

  const handleDialogueFinished = (results: StructuredDialogueResult[]) => {
      setDialogueFinished(true);
      setStructuredDialogueResults(results);
      const combinedTranscript = results.map(r => r.userInterpretation).join('\n\n');
      setUserTranscript(combinedTranscript);
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
         {settings.mode === 'Gesprächsdolmetschen' && dialogueFinished ? (
              <DialogueResults
                feedback={feedback}
                getFeedback={getFeedback}
                isLoading={isLoading}
                loadingMessage={loadingMessage}
                error={error}
                structuredResults={structuredDialogueResults}
                originalText={originalText}
                userTranscript={userTranscript}
              />
            ) : (
              <PracticeArea
                  key={originalText} // Remount component when text changes
                  isLoading={isLoading} 
                  loadingMessage={loadingMessage}
                  originalText={originalText}
                  onRecordingFinished={handleRecordingFinished}
                  onTranscriptChange={handleTranscriptChange}
                  getFeedback={getFeedback}
                  userTranscript={userTranscript}
                  feedback={feedback}
                  error={error}
                  settings={settings}
                  exerciseStarted={exerciseStarted}
                  onPremiumVoiceAuthError={handlePremiumVoiceAuthError}
                  onDialogueFinished={handleDialogueFinished}
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
                    <label>Quelle des Originaltextes</label>
                    <select id="sourceType" name="sourceType" className="form-control" value={currentSettings.sourceType} onChange={handleChange} disabled={disabled}>
                        <option value="ai">KI-generiert</option>
                        <option value="upload">Textdatei (.txt)</option>
                    </select>
                </div>
                
                {currentSettings.sourceType === 'ai' ? (
                    <div className="form-group">
                        <label htmlFor="topic">Thema</label>
                        <input type="text" id="topic" name="topic" className="form-control" value={currentSettings.topic} onChange={handleChange} disabled={disabled} placeholder="z.B. Erneuerbare Energien" />
                    </div>
                ) : (
                    <div className="form-group">
                        <label>Datei hochladen</label>
                         <div className="upload-group">
                            <button className="btn btn-secondary" onClick={handleUploadClick} disabled={disabled}>Datei wählen</button>
                            {uploadedFile && <span className="file-name" title={uploadedFile.name}>{uploadedFile.name}</span>}
                        </div>
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".txt" style={{ display: 'none' }} />
                    </div>
                )}

                {currentSettings.mode === 'Gesprächsdolmetschen' && (
                     <div className="form-group">
                        <label htmlFor="qaLength">Länge der Fragen/Antworten</label>
                        <select id="qaLength" name="qaLength" className="form-control" value={currentSettings.qaLength} onChange={handleChange} disabled={disabled}>
                            {QA_LENGTHS.map(l => <option key={l} value={l}>{l}</option>)}
                        </select>
                    </div>
                )}

                {speechModes.includes(currentSettings.mode) && (
                     <div className="form-group">
                        <label htmlFor="speechLength">Länge der Rede</label>
                        <select id="speechLength" name="speechLength" className="form-control" value={currentSettings.speechLength} onChange={handleChange} disabled={disabled}>
                           {SPEECH_LENGTHS.map(l => <option key={l} value={l}>{l}</option>)}
                        </select>
                    </div>
                )}
                 {currentSettings.mode === 'Stegreifübersetzen' && (
                     <p className="form-text-hint">Textlänge: ca. 1300-1450 Zeichen</p>
                 )}

            </div>
            <div className="settings-footer">
                <button className="btn btn-primary" onClick={handleSubmit} disabled={disabled}>
                    Übung starten
                </button>
            </div>
        </aside>
    );
};

const PracticeArea = ({
    isLoading,
    loadingMessage,
    originalText,
    onRecordingFinished,
    onTranscriptChange,
    getFeedback,
    userTranscript,
    feedback,
    error,
    settings,
    exerciseStarted,
    onPremiumVoiceAuthError,
    onDialogueFinished,
} : {
    isLoading: boolean,
    loadingMessage: string,
    originalText: string,
    onRecordingFinished: (transcript: string) => void,
    onTranscriptChange: (newTranscript: string) => void,
    getFeedback: (textToCompare?: string) => void,
    userTranscript: string,
    feedback: Feedback | null,
    error: string | null,
    settings: Settings,
    exerciseStarted: boolean,
    onPremiumVoiceAuthError: () => void,
    onDialogueFinished: (results: StructuredDialogueResult[]) => void;
}) => {
    const [activeTab, setActiveTab] = useState<PracticeAreaTab>('original');
    
    const [isRecording, setIsRecording] = useState(false);
    const [recordingStatus, setRecordingStatus] = useState('Klicken zum Starten der Aufnahme');
    const recognitionRef = useRef<SpeechRecognition | null>(null);
    const finalTranscriptRef = useRef('');

    const handleGetFeedback = () => {
      let textToCompare = originalText;
      if (settings.mode === 'Shadowing') {
          textToCompare = originalText; 
      }
      getFeedback(textToCompare);
    };

    const startRecording = useCallback(() => {
        if (isRecording) return;
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            setRecordingStatus("Spracherkennung wird von Ihrem Browser nicht unterstützt.");
            return;
        }

        const recognition = new SpeechRecognition();
        recognitionRef.current = recognition;
        
        const langCode = settings.mode === 'Shadowing' ? LANGUAGE_CODES[settings.sourceLang] : LANGUAGE_CODES[settings.targetLang];
        recognition.lang = langCode;
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onstart = () => {
            console.log("Speech recognition started.");
            setIsRecording(true);
            setRecordingStatus('Aufnahme läuft... Klicken zum Stoppen.');
        };
        
        recognition.onresult = (event: SpeechRecognitionEvent) => {
            let interimTranscript = '';
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            finalTranscriptRef.current += finalTranscript;
            setRecordingStatus(`Aufnahme läuft... (Zwischenergebnis: ${interimTranscript})`);
        };

        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
            console.error("Speech recognition error", event.error, event.message);
            setRecordingStatus(`Fehler: ${event.error}. Klicken zum erneuten Start.`);
            setIsRecording(false);
        };

        recognition.onend = () => {
            console.log("Speech recognition service disconnected.");
            setIsRecording(false);
            setRecordingStatus('Verarbeitung...');
            onRecordingFinished(finalTranscriptRef.current);
        };
        
        finalTranscriptRef.current = '';
        recognition.start();

    }, [isRecording, settings.targetLang, settings.sourceLang, settings.mode, onRecordingFinished]);

    const stopRecording = useCallback(() => {
        if (recognitionRef.current) {
            recognitionRef.current.stop();
        }
    }, []);

    const handleRecordButtonClick = () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    };

    const renderContent = () => {
        if (settings.mode === 'Gesprächsdolmetschen') {
            return (
                <DialoguePractice 
                    script={originalText}
                    settings={settings}
                    onDialogueFinished={onDialogueFinished}
                    onPremiumVoiceAuthError={onPremiumVoiceAuthError}
                />
            );
        }
    
        if (settings.mode === 'Stegreifübersetzen') {
            return <SightTranslationPractice text={originalText} />;
        }
    
        // Default for Speech-based modes
        return (
            <>
                <AudioPlayer 
                    key={originalText} 
                    text={originalText}
                    lang={settings.sourceLang} 
                    voiceQuality={settings.voiceQuality}
                    onAuthError={onPremiumVoiceAuthError}
                />
                <div className="practice-footer">
                    <p className="recording-status-text">{recordingStatus}</p>
                    <button 
                        className={`btn-record ${isRecording ? 'recording' : ''}`}
                        onClick={handleRecordButtonClick}
                        disabled={isLoading}
                        aria-label={isRecording ? 'Aufnahme stoppen' : 'Aufnahme starten'}
                    >
                      <div className="mic-icon"></div>
                    </button>
                </div>
            </>
        );
    };

    if (!exerciseStarted && !isLoading) {
        return (
            <aside className="panel practice-area placeholder">
                <h2>Willkommen beim Dolmetsch-Trainer Pro</h2>
                <p>Bitte nehmen Sie Ihre Einstellungen vor und starten Sie eine neue Übung.</p>
            </aside>
        );
    }
    
    return (
        <aside className="panel practice-area">
            {isLoading && !exerciseStarted && (
              <div className="loading-overlay">
                <div className="spinner"></div>
                <p>{loadingMessage}</p>
              </div>
            )}
            <div className="tabs">
                <button className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`} onClick={() => setActiveTab('original')}>
                    Übung
                </button>
                <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')} disabled={!userTranscript && !feedback}>
                    Ihr Transkript
                </button>
                <button className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} onClick={() => setActiveTab('feedback')} disabled={!feedback}>
                    Feedback
                </button>
            </div>
            <div className="tab-content">
                {error && <div className="error-banner">{error}</div>}
                
                 {isLoading && exerciseStarted && (
                    <div className="loading-overlay">
                        <div className="spinner"></div>
                        <p>{loadingMessage}</p>
                    </div>
                )}

                {activeTab === 'original' && (
                    <div className="text-area">
                       {renderContent()}
                    </div>
                )}
                {activeTab === 'transcript' && (
                     <TranscriptEditor 
                        transcript={userTranscript}
                        onTranscriptChange={onTranscriptChange}
                        onGetFeedback={handleGetFeedback}
                        disabled={isLoading}
                    />
                )}
                {activeTab === 'feedback' && feedback && (
                    <FeedbackDisplay feedback={feedback} />
                )}
            </div>
        </aside>
    );
};


const AudioPlayer = ({ text, lang, voiceQuality, onAuthError }: { text: string, lang: Language, voiceQuality: VoiceQuality, onAuthError: () => void }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [playbackRate, setPlaybackRate] = useState(1.0);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const synth = window.speechSynthesis;
    const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

    const playPause = useCallback(() => {
        if (voiceQuality === 'Premium') {
            if (audioRef.current) {
                if (isPlaying) {
                    audioRef.current.pause();
                } else {
                    audioRef.current.play().catch(e => console.error("Audio play failed:", e));
                }
                setIsPlaying(!isPlaying);
            }
        } else { // Standard voice
            if (isPlaying) {
                synth.pause();
                setIsPlaying(false);
            } else {
                if (synth.paused && utteranceRef.current) {
                    synth.resume();
                } else if (utteranceRef.current){
                    synth.speak(utteranceRef.current);
                }
                setIsPlaying(true);
            }
        }
    }, [isPlaying, synth, voiceQuality]);

    const stop = useCallback(() => {
       if (voiceQuality === 'Premium') {
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current.currentTime = 0;
                setIsPlaying(false);
            }
        } else {
            synth.cancel();
            setIsPlaying(false);
        }
    }, [synth, voiceQuality]);
    
    const changePlaybackRate = (rate: number) => {
        const newRate = Math.max(0.5, Math.min(2.0, rate));
        setPlaybackRate(newRate);
         if (voiceQuality === 'Premium' && audioRef.current) {
            audioRef.current.playbackRate = newRate;
        } else if (utteranceRef.current) {
            // Note: SpeechSynthesisUtterance rate is tricky and might not update mid-speech
            // Best to stop and start again, but for simplicity, we set it here.
            utteranceRef.current.rate = newRate;
        }
    };
    
    useEffect(() => {
        let isMounted = true;
        
        const initializeAudio = async () => {
             if (!text) return;
             
             // Cleanup previous instances
             if (audioRef.current) {
                 audioRef.current.src = '';
                 audioRef.current = null;
             }
             if (utteranceRef.current) {
                 utteranceRef.current = null;
             }
             synth.cancel();
             
            if (voiceQuality === 'Premium') {
                try {
                    if (!process.env.API_KEY) throw new Error("API Key for TTS not found");
                    const audioContent = await synthesizeSpeechGoogleCloud(text, lang, process.env.API_KEY);
                    if (!isMounted) return;
                    
                    const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = new Audio(audioUrl);
                    audioRef.current = audio;
                    
                    audio.addEventListener('loadedmetadata', () => setDuration(audio.duration));
                    audio.addEventListener('timeupdate', () => setCurrentTime(audio.currentTime));
                    audio.addEventListener('ended', () => setIsPlaying(false));
                    audio.playbackRate = playbackRate;
                } catch (err) {
                    console.error("Premium voice synthesis failed:", err);
                    if (err instanceof TtsAuthError) {
                        onAuthError();
                    }
                    // Fallback or show error would be handled by the parent
                }
            } else { // Standard voice
                const utterance = new SpeechSynthesisUtterance(text);
                const voices = synth.getVoices();
                const langCode = LANGUAGE_CODES[lang];
                utterance.voice = voices.find(voice => voice.lang === langCode) || voices.find(voice => voice.lang.startsWith(langCode.split('-')[0])) || null;
                utterance.lang = langCode;
                utterance.rate = playbackRate;
                utterance.onend = () => isMounted && setIsPlaying(false);
                utteranceRef.current = utterance;
                // Duration/currentTime is not reliably available for SpeechSynthesis
                setDuration(0);
                setCurrentTime(0);
            }
        };

        // SpeechSynthesis voices can load asynchronously
        if (synth.getVoices().length === 0) {
            synth.onvoiceschanged = initializeAudio;
        } else {
            initializeAudio();
        }

        return () => {
            isMounted = false;
            synth.cancel();
            if (audioRef.current) {
                audioRef.current.pause();
                URL.revokeObjectURL(audioRef.current.src);
            }
        };
    }, [text, lang, synth, voiceQuality, onAuthError, playbackRate]);


    return (
        <div className="audio-player">
            <div className="controls-bar">
                <button onClick={playPause} disabled={!text}>{isPlaying ? 'Pause' : 'Play'}</button>
                <button onClick={stop} disabled={!text}>Stop</button>
                 <p>Geschw: {playbackRate.toFixed(1)}x</p>
                <button onClick={() => changePlaybackRate(playbackRate - 0.1)} disabled={!text}>-</button>
                <button onClick={() => changePlaybackRate(playbackRate + 0.1)} disabled={!text}>+</button>
            </div>
            {voiceQuality === 'Premium' &&
                <input
                    type="range"
                    min="0"
                    max={duration || 0}
                    value={currentTime}
                    onChange={(e) => {
                        if (audioRef.current) audioRef.current.currentTime = Number(e.target.value);
                    }}
                    style={{ width: '100%' }}
                />
            }
             <div className="text-area">
                <p>{text || "Kein Text zum Abspielen verfügbar."}</p>
            </div>
        </div>
    );
};

const SightTranslationPractice = ({ text }: { text: string }) => {
    const [showText, setShowText] = useState(false);

    if (!showText) {
        return (
            <div className="current-segment-display">
                <div className="dialogue-text-container">
                    <p className="segment-text-hidden">Der Text ist verborgen, um ein realistisches Prüfungsszenario zu simulieren.</p>
                    <button className="btn btn-secondary btn-show-text" onClick={() => setShowText(true)}>Text anzeigen</button>
                </div>
            </div>
        );
    }

    return (
        <div className="text-area">
            <p>{text}</p>
        </div>
    );
};

const DialoguePractice = ({ script, settings, onDialogueFinished, onPremiumVoiceAuthError }: {
    script: string,
    settings: Settings,
    onDialogueFinished: (results: StructuredDialogueResult[]) => void,
    onPremiumVoiceAuthError: () => void,
}) => {
    const [dialogueState, setDialogueState] = useState<DialogueState>('idle');
    const [segments, setSegments] = useState<DialogueSegment[]>([]);
    const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);
    const [userInterpretations, setUserInterpretations] = useState<string[]>([]);
    const [showText, setShowText] = useState(false);

    const recognitionRef = useRef<SpeechRecognition | null>(null);
    const finalTranscriptRef = useRef('');
    
    // Parse script into segments
    useEffect(() => {
        if (script) {
            const lines = script.trim().split('\n');
            // Fix: Add explicit return type to the map callback to ensure correct type inference for the 'type' property.
            const parsedSegments: DialogueSegment[] = lines.map((line): DialogueSegment => {
                const isQuestion = line.toLowerCase().startsWith('frage');
                const text = line.substring(line.indexOf(':') + 1).trim();
                return {
                    type: isQuestion ? 'Frage' : 'Antwort',
                    text: text,
                    lang: isQuestion ? settings.sourceLang : settings.targetLang
                };
            }).filter(s => s.text); // Filter out empty lines
            setSegments(parsedSegments);
            setDialogueState('ready');
        }
    }, [script, settings.sourceLang, settings.targetLang]);

    const handleAudioEnded = useCallback(() => {
        setDialogueState('waiting_for_record');
    }, []);

    const startInterpretationRecording = useCallback(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert("Spracherkennung wird von Ihrem Browser nicht unterstützt.");
            return;
        }

        const recognition = new SpeechRecognition();
        recognitionRef.current = recognition;
        
        // User always interprets into the OTHER language
        const currentSegment = segments[currentSegmentIndex];
        const targetLangCode = currentSegment.lang === settings.sourceLang 
                               ? LANGUAGE_CODES[settings.targetLang] 
                               : LANGUAGE_CODES[settings.sourceLang];
        
        recognition.lang = targetLangCode;
        recognition.continuous = true;
        recognition.interimResults = false; // Simpler for dialogue

        recognition.onstart = () => {
            setDialogueState('recording');
        };

        recognition.onresult = (event: SpeechRecognitionEvent) => {
            let transcript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    transcript += event.results[i][0].transcript;
                }
            }
            finalTranscriptRef.current += transcript;
        };
        
        recognition.onend = () => {
            setUserInterpretations(prev => [...prev, finalTranscriptRef.current]);
            if (currentSegmentIndex < segments.length - 1) {
                setCurrentSegmentIndex(prev => prev + 1);
                setDialogueState('ready');
                setShowText(false);
            } else {
                setDialogueState('finished');
            }
        };

        finalTranscriptRef.current = '';
        recognition.start();

    }, [currentSegmentIndex, segments, settings.sourceLang, settings.targetLang]);

    const stopInterpretationRecording = useCallback(() => {
        if (recognitionRef.current) {
            recognitionRef.current.stop();
        }
    }, []);

    // Effect to handle state transitions to finished
    useEffect(() => {
        if (dialogueState === 'finished') {
            const results: StructuredDialogueResult[] = segments.map((segment, index) => {
                 const interpretationLang = segment.lang === settings.sourceLang ? settings.targetLang : settings.sourceLang;
                 return {
                    originalSegment: segment,
                    userInterpretation: userInterpretations[index] || '(Keine Aufnahme)',
                    interpretationLang: interpretationLang
                 }
            });
            onDialogueFinished(results);
        }
    }, [dialogueState, segments, userInterpretations, onDialogueFinished, settings.sourceLang, settings.targetLang]);


    const renderCurrentStep = () => {
        if (segments.length === 0) {
            return <p>Lade Dialog...</p>;
        }

        const currentSegment = segments[currentSegmentIndex];
        const interpretationLang = currentSegment.lang === settings.sourceLang ? settings.targetLang : settings.sourceLang;

        switch (dialogueState) {
            case 'ready':
                return (
                    <div className="current-segment-display">
                        {!showText ? (
                            <div className="dialogue-text-container">
                                <p className="segment-text-hidden">{currentSegment.type} {currentSegmentIndex + 1} ist bereit.</p>
                                <button className="btn btn-secondary btn-show-text" onClick={() => setShowText(true)}>Text anzeigen & vorlesen</button>
                            </div>
                        ) : (
                            <DialogueAudioPlayer
                                text={currentSegment.text}
                                lang={currentSegment.lang}
                                onEnded={handleAudioEnded}
                                onAuthError={onPremiumVoiceAuthError}
                            />
                        )}
                    </div>
                );
            case 'waiting_for_record':
                return (
                     <div className="dialogue-text-container">
                        <p className="segment-text">Bitte dolmetschen Sie nun nach {interpretationLang}.</p>
                        <button className="btn btn-primary" onClick={startInterpretationRecording}>Aufnahme starten</button>
                    </div>
                );
            case 'recording':
                return (
                    <div className="dialogue-text-container">
                        <p className="segment-text">Aufnahme läuft...</p>
                        <button className="btn btn-record recording" onClick={stopInterpretationRecording}>
                            <div className="mic-icon"></div>
                        </button>
                    </div>
                );
            case 'finished':
                 return <p>Übung abgeschlossen. Ergebnisse werden angezeigt.</p>;
            default:
                return <p>Initialisiere Übung...</p>;
        }
    };
    
    return (
        <div className="dialogue-practice-container">
            <div className="dialogue-status">
                {dialogueState !== 'finished' ? 
                `Segment ${currentSegmentIndex + 1} / ${segments.length} - ${segments[currentSegmentIndex]?.type} (${segments[currentSegmentIndex]?.lang})`
                : 'Dialog beendet'}
            </div>
            {renderCurrentStep()}
        </div>
    );
};

const DialogueAudioPlayer = ({ text, lang, onEnded, onAuthError }: { text: string, lang: Language, onEnded: () => void, onAuthError: () => void }) => {
    const [error, setError] = useState('');
    
    useEffect(() => {
        let isMounted = true;
        let audio: HTMLAudioElement | null = null;
        const synth = window.speechSynthesis;

        const playAudio = async () => {
            try {
                if (!process.env.API_KEY) throw new Error("API Key for TTS not found");
                const audioContent = await synthesizeSpeechGoogleCloud(text, lang, process.env.API_KEY);
                if (!isMounted) return;

                const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
                const audioUrl = URL.createObjectURL(audioBlob);
                audio = new Audio(audioUrl);
                audio.addEventListener('ended', () => {
                    if(isMounted) onEnded();
                });
                audio.play().catch(e => {
                    console.error("Audio play failed:", e);
                    setError('Fehler beim Abspielen des Audios.');
                });

            } catch (err) {
                 console.error("Premium voice synthesis failed:", err);
                 if (err instanceof TtsAuthError) {
                    onAuthError(); // Signal parent about auth issue
                    // Fallback to standard voice
                    playStandardVoice();
                 } else {
                     setError('Fehler bei der Sprachsynthese.');
                 }
            }
        };

        const playStandardVoice = () => {
            const utterance = new SpeechSynthesisUtterance(text);
            const voices = synth.getVoices();
            const langCode = LANGUAGE_CODES[lang];
            utterance.voice = voices.find(voice => voice.lang === langCode) || voices.find(voice => voice.lang.startsWith(langCode.split('-')[0])) || null;
            utterance.lang = langCode;
            utterance.onend = () => {
                 if (isMounted) onEnded();
            };
            synth.speak(utterance);
        };
        
        // Start playing automatically
        playAudio();

        return () => {
            isMounted = false;
            synth.cancel();
            if (audio) {
                audio.pause();
                URL.revokeObjectURL(audio.src);
            }
        };
    }, [text, lang, onEnded, onAuthError]);

    return (
        <div className="dialogue-text-container">
            {error ? <p className="error-banner">{error}</p> : <p className="segment-text"><i>Spielt Audio ab...</i></p>}
            <p className="segment-text">{text}</p>
        </div>
    );
};


const TranscriptEditor = ({ transcript, onTranscriptChange, onGetFeedback, disabled }: {
    transcript: string,
    onTranscriptChange: (newTranscript: string) => void,
    onGetFeedback: () => void,
    disabled: boolean
}) => {
    return (
        <>
            <div className="text-area">
                <textarea 
                    className="form-control text-area-editor"
                    value={transcript}
                    onChange={(e) => onTranscriptChange(e.target.value)}
                    placeholder="Hier erscheint Ihr Transkript nach der Aufnahme..."
                />
            </div>
            <div className="practice-footer">
                <p className="recording-status-text">Sie können das Transkript vor der Analyse korrigieren.</p>
                <button className="btn btn-primary" onClick={onGetFeedback} disabled={disabled || !transcript}>
                    Feedback erhalten
                </button>
            </div>
        </>
    );
};

const FeedbackDisplay = ({ feedback }: { feedback: Feedback }) => {
    const renderStars = (rating: number) => {
        return Array.from({ length: 10 }, (_, i) => (
            <span key={i} className={`star ${i < rating ? 'filled' : ''}`}>★</span>
        ));
    };

    return (
        <div className="feedback-content text-area">
            <h3>Zusammenfassung</h3>
            <p>{feedback.summary}</p>

            <h3>Bewertungen</h3>
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
            
            {feedback.errorAnalysis && feedback.errorAnalysis.length > 0 && (
                <>
                    <h3>Fehleranalyse</h3>
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

const DialogueResults = ({
    structuredResults,
    feedback,
    getFeedback,
    isLoading,
    loadingMessage,
    error,
    originalText,
    userTranscript,
}: {
    structuredResults: StructuredDialogueResult[] | null,
    feedback: Feedback | null,
    getFeedback: (textToCompare?: string) => void,
    isLoading: boolean,
    loadingMessage: string,
    error: string | null,
    originalText: string,
    userTranscript: string
}) => {
    const [activeTab, setActiveTab] = useState<'transcript' | 'feedback'>('transcript');

    const handleGetFeedback = () => {
        // In dialogue mode, we compare the entire conversation
        const combinedOriginal = structuredResults?.map(r => r.originalSegment.text).join('\n\n') ?? originalText;
        getFeedback(combinedOriginal);
    }

    return (
        <aside className="panel practice-area">
            {isLoading && (
              <div className="loading-overlay">
                <div className="spinner"></div>
                <p>{loadingMessage}</p>
              </div>
            )}
            <div className="tabs">
                <button 
                    className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} 
                    onClick={() => setActiveTab('transcript')}
                >
                    Dialog-Transkript
                </button>
                <button 
                    className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`} 
                    onClick={() => setActiveTab('feedback')} 
                    disabled={!feedback}
                >
                    Feedback
                </button>
            </div>
            <div className="tab-content">
                {error && <div className="error-banner">{error}</div>}

                {activeTab === 'transcript' && (
                    <div className="dialogue-results-wrapper">
                        <div className="text-area structured-transcript">
                            {structuredResults?.map((result, index) => (
                                <div key={index} className="transcript-segment">
                                    <div className="transcript-segment-header">
                                        <h4>{result.originalSegment.type} {index + 1} ({result.originalSegment.lang})</h4>
                                    </div>
                                    <p className="transcript-segment-original">{result.originalSegment.text}</p>
                                    <p className="transcript-segment-user">
                                        <strong>Ihre Verdolmetschung ({result.interpretationLang}):</strong><br/>
                                        {result.userInterpretation.trim() ? result.userInterpretation : <em>(Keine Aufnahme)</em>}
                                    </p>
                                </div>
                            ))}
                        </div>
                         <div className="practice-footer">
                            <button className="btn btn-primary" onClick={handleGetFeedback} disabled={isLoading || !userTranscript}>
                                Feedback für den gesamten Dialog erhalten
                            </button>
                        </div>
                    </div>
                )}

                {activeTab === 'feedback' && feedback && (
                     <FeedbackDisplay feedback={feedback} />
                )}
            </div>
        </aside>
    );
};


const root = createRoot(document.getElementById('root')!);
root.render(<App />);