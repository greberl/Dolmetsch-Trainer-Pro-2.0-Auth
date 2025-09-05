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
            // FIX: Access the .text property on the response, not .text() method.
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
            // FIX: Access the .text property on the response, not .text() method.
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
                // FIX: Access the .text property on the response, not .text() method.
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
      setExerciseId(Date.now());
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
        // FIX: Access the .text property on the response, not .text() method.
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

  const handleOriginalTextChange = (newText: string) => {
    setOriginalText(newText);
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
        
        // FIX: Access the .text property on the response, not .text() method.
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
                  key={exerciseId} // Remount component when exerciseId changes
                  isLoading={isLoading} 
                  loadingMessage={loadingMessage}
                  originalText={originalText}
                  onRecordingFinished={handleRecordingFinished}
                  onTranscriptChange={handleTranscriptChange}
                  onOriginalTextChange={handleOriginalTextChange}
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
    onOriginalTextChange,
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
    onOriginalTextChange: (newText: string) => void,
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
    const recognitionRef = useRef<SpeechRecognition | null>(null);
    const transcriptRef = useRef<string>('');
    const [isSpeechSupported, setIsSpeechSupported] = useState(true);

    const prevUserTranscript = usePrevious(userTranscript);
    const prevFeedback = usePrevious(feedback);

    // Automatically switch to the transcript tab when it's generated
    useEffect(() => {
        if (userTranscript && !prevUserTranscript) {
            setActiveTab('transcript');
        }
    }, [userTranscript, prevUserTranscript]);

    // Automatically switch to the feedback tab when it's generated
    useEffect(() => {
        if (feedback && !prevFeedback) {
            setActiveTab('feedback');
        }
    }, [feedback, prevFeedback]);


    const isSpeechMode = ["Vortragsdolmetschen", "Simultandolmetschen", "Shadowing"].includes(settings.mode);

    useEffect(() => {
        const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognitionAPI) {
            setIsSpeechSupported(false);
            console.error("Speech Recognition API is not supported in this browser.");
            return;
        }

        const recognition = new SpeechRecognitionAPI();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = LANGUAGE_CODES[settings.targetLang];

        recognition.onresult = (event: SpeechRecognitionEvent) => {
            let interimTranscript = '';
            let finalTranscript = transcriptRef.current;
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript + ' ';
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            transcriptRef.current = finalTranscript;
            // Optionally, you can update the UI with the interim transcript here
        };

        recognition.onend = () => {
            setIsRecording(false);
            onRecordingFinished(transcriptRef.current);
        };
        
        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
            console.error("Speech recognition error:", event.error, event.message);
            setIsRecording(false);
            onRecordingFinished(transcriptRef.current); // Process what was recorded
        };

        recognitionRef.current = recognition;

        return () => {
             if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
        };
    }, [settings.targetLang, onRecordingFinished]);

    const handleToggleRecording = () => {
        if (!isSpeechSupported) {
            alert("Die Spracherkennung wird von Ihrem Browser nicht unterstützt.");
            return;
        }
        if (isRecording) {
            recognitionRef.current?.stop();
        } else {
            transcriptRef.current = '';
            recognitionRef.current?.start();
            setIsRecording(true);
        }
    };
    
    const handleGetFeedback = () => {
        const textToCompare = settings.mode === 'Shadowing' ? originalText.split(' ').slice(0, userTranscript.split(' ').length).join(' ') : originalText;
        getFeedback(textToCompare);
    }
    
    if (!exerciseStarted && !isLoading) {
        return (
            <section className="panel practice-area">
                <div className="placeholder">
                    <h2>Willkommen beim Dolmetsch-Trainer Pro</h2>
                    <p>Passen Sie die Einstellungen links an und klicken Sie auf "Übung starten", um zu beginnen.</p>
                </div>
            </section>
        );
    }

    if (settings.mode === 'Gesprächsdolmetschen') {
        return (
            <DialoguePractice
              originalText={originalText}
              settings={settings}
              isLoading={isLoading}
              loadingMessage={loadingMessage}
              onDialogueFinished={onDialogueFinished}
              onPremiumVoiceAuthError={onPremiumVoiceAuthError}
            />
        );
    }

    return (
        <section className="panel practice-area">
            {isLoading && (
                <div className="loading-overlay">
                    <div className="spinner"></div>
                    <p>{loadingMessage}</p>
                </div>
            )}
            <div className="tabs">
                <button 
                    className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`} 
                    onClick={() => setActiveTab('original')}
                    disabled={!originalText}
                >
                    Originaltext
                </button>
                 <button 
                    className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} 
                    onClick={() => setActiveTab('transcript')}
                    disabled={!userTranscript && !isRecording}
                >
                    Ihr Transkript
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
                {activeTab === 'original' && (
                    <TextDisplay 
                        text={originalText} 
                        settings={settings}
                        isSpeechMode={isSpeechMode}
                        onPremiumVoiceAuthError={onPremiumVoiceAuthError}
                        onTextChange={onOriginalTextChange}
                    />
                )}
                 {activeTab === 'transcript' && (
                     <TranscriptDisplay
                        transcript={userTranscript}
                        onTranscriptChange={onTranscriptChange}
                        onGetFeedback={handleGetFeedback}
                     />
                 )}
                 {activeTab === 'feedback' && feedback && (
                     <FeedbackDisplay feedback={feedback} />
                 )}
            </div>

            <footer className="practice-footer">
                <p className="recording-status-text">
                    {isRecording ? "Aufnahme läuft..." : (userTranscript ? "Aufnahme beendet." : "Bereit zur Aufnahme.")}
                </p>
                <button 
                    className={`btn-record ${isRecording ? 'recording' : ''}`}
                    onClick={handleToggleRecording}
                    disabled={isLoading || !originalText || (isSpeechMode && !window.speechSynthesis.speaking && activeTab !== 'original')}
                    aria-label={isRecording ? 'Aufnahme stoppen' : 'Aufnahme starten'}
                >
                    <span className="mic-icon"></span>
                </button>
            </footer>
        </section>
    );
};


const TextDisplay = ({ text, settings, isSpeechMode, onPremiumVoiceAuthError, onTextChange }: { text: string, settings: Settings, isSpeechMode: boolean, onPremiumVoiceAuthError: () => void, onTextChange: (newText: string) => void }) => {
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [playbackRate, setPlaybackRate] = useState(1);
    const [charCount, setCharCount] = useState(0);
    const [isEditing, setIsEditing] = useState(false);

    const handlePlayPause = async () => {
        if (isPlaying) {
            window.speechSynthesis.cancel();
            if (audioRef.current) {
                audioRef.current.pause();
            }
            setIsPlaying(false);
        } else {
            if (settings.voiceQuality === 'Premium') {
                try {
                    const apiKey = process.env.API_KEY;
                    if (!apiKey) throw new TtsAuthError('API key for TTS is not configured.');
                    const audioContent = await synthesizeSpeechGoogleCloud(text, settings.sourceLang, apiKey);
                    const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    if (!audioRef.current) {
                        audioRef.current = new Audio();
                        audioRef.current.onended = () => setIsPlaying(false);
                    }
                    audioRef.current.src = audioUrl;
                    audioRef.current.playbackRate = playbackRate;
                    await audioRef.current.play();
                    setIsPlaying(true);
                } catch (error) {
                    console.error("Premium voice synthesis failed:", error);
                    if (error instanceof TtsAuthError) {
                        alert("Premium-Stimme fehlgeschlagen: " + error.message + " Standardstimme wird verwendet.");
                        onPremiumVoiceAuthError();
                    } else {
                         alert("Ein unerwarteter Fehler ist bei der Premium-Stimme aufgetreten. Standardstimme wird verwendet.");
                    }
                    // Fallback to standard voice
                    playWithStandardVoice();
                }
            } else {
                playWithStandardVoice();
            }
        }
    };
    
    const playWithStandardVoice = () => {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = LANGUAGE_CODES[settings.sourceLang];
        utterance.rate = playbackRate;
        utterance.onend = () => setIsPlaying(false);
        window.speechSynthesis.speak(utterance);
        setIsPlaying(true);
    }
    
    const handleRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newRate = parseFloat(e.target.value);
        setPlaybackRate(newRate);
        if (isPlaying) {
             if (settings.voiceQuality === 'Premium' && audioRef.current) {
                audioRef.current.playbackRate = newRate;
            } else if (settings.voiceQuality === 'Standard') {
                window.speechSynthesis.cancel();
                playWithStandardVoice(); // Restart with new rate
            }
        }
    };
    
    useEffect(() => {
        setCharCount(text.length);
        return () => { // Cleanup on unmount
            window.speechSynthesis.cancel();
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current = null;
            }
        };
    }, [text]);


    return (
        <>
            <div className="text-area">
                 {(isSpeechMode || settings.mode === 'Stegreifübersetzen') && (
                    <div className="controls-bar">
                        {isSpeechMode && (
                            <>
                               <button onClick={handlePlayPause} disabled={!text || isEditing} className="btn-play-pause" aria-label={isPlaying ? "Pausieren" : "Abspielen"}>
                                    {isPlaying ? (
                                        <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" fill="currentColor"><path d="M0 0h24v24H0V0z" fill="none" /><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg>
                                    ) : (
                                        <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" fill="currentColor"><path d="M0 0h24v24H0V0z" fill="none" /><path d="M8 5v14l11-7L8 5z" /></svg>
                                    )}
                                </button>
                                <label htmlFor="playbackRate">Geschwindigkeit: {playbackRate.toFixed(1)}x</label>
                                <input
                                    type="range"
                                    id="playbackRate"
                                    name="playbackRate"
                                    min="0.5"
                                    max="1.5"
                                    step="0.1"
                                    value={playbackRate}
                                    onChange={handleRateChange}
                                    disabled={isEditing}
                                />
                            </>
                        )}
                         <p>Zeichen (inkl. Leerzeichen): {charCount}</p>
                    </div>
                )}
                {isEditing ? (
                     <textarea 
                        className="form-control text-area-editor"
                        value={text}
                        onChange={(e) => onTextChange(e.target.value)}
                        placeholder="Originaltext hier bearbeiten..."
                        autoFocus
                    />
                ) : (
                    <p>{text || "Kein Text zum Anzeigen."}</p>
                )}
            </div>
            <div className="controls-bar" style={{marginTop: '1rem', justifyContent: 'flex-end'}}>
                 {isEditing ? (
                     <button className="btn btn-secondary" onClick={() => setIsEditing(false)}>
                        Fertig
                     </button>
                ) : (
                     <button className="btn btn-secondary" onClick={() => setIsEditing(true)} disabled={!text}>
                        Bearbeiten
                     </button>
                )}
            </div>
        </>
    );
};

const TranscriptDisplay = ({ transcript, onTranscriptChange, onGetFeedback }: { transcript: string, onTranscriptChange: (newVal: string) => void, onGetFeedback: () => void }) => {
    const [isEditing, setIsEditing] = useState(false);
    
    return (
        <>
            <div className="text-area">
                {isEditing ? (
                    <textarea 
                        className="form-control text-area-editor"
                        value={transcript}
                        onChange={(e) => onTranscriptChange(e.target.value)}
                        placeholder="Ihr Transkript wird hier angezeigt..."
                        autoFocus
                    />
                ) : (
                    <p>{transcript || "Ihr Transkript wird hier angezeigt..."}</p>
                )}
            </div>
            <div className="controls-bar" style={{marginTop: '1rem', justifyContent: 'flex-end'}}>
                {isEditing ? (
                     <button className="btn btn-secondary" onClick={() => setIsEditing(false)}>
                        Fertig
                     </button>
                ) : (
                     <button className="btn btn-secondary" onClick={() => setIsEditing(true)} disabled={!transcript}>
                        Bearbeiten
                     </button>
                )}
                <button className="btn btn-secondary" onClick={onGetFeedback} disabled={!transcript || isEditing}>
                    Feedback anfordern
                </button>
            </div>
        </>
    );
};

const FeedbackDisplay = ({ feedback }: { feedback: Feedback }) => {
    const { summary, ratings, errorAnalysis } = feedback;
    
    const renderStars = (rating: number) => {
        return Array.from({ length: 10 }, (_, i) => (
            <span key={i} className={`star ${i < rating ? 'filled' : ''}`}>★</span>
        ));
    };

    return (
         <div className="text-area feedback-content">
            <h3>Zusammenfassung</h3>
            <p>{summary}</p>
            
            <h3>Bewertung</h3>
             <table className="ratings-table">
                <tbody>
                    <tr>
                        <td>Inhalt</td>
                        <td>{renderStars(ratings.content)}</td>
                        <td>{ratings.content}/10</td>
                    </tr>
                     <tr>
                        <td>Ausdruck</td>
                        <td>{renderStars(ratings.expression)}</td>
                        <td>{ratings.expression}/10</td>
                    </tr>
                     <tr>
                        <td>Terminologie</td>
                        <td>{renderStars(ratings.terminology)}</td>
                        <td>{ratings.terminology}/10</td>
                    </tr>
                </tbody>
            </table>
            
            <h3>Fehleranalyse</h3>
            {errorAnalysis && errorAnalysis.length > 0 ? (
                <ul className="error-analysis-list">
                    {errorAnalysis.map((item, index) => (
                        <li key={index}>
                            <p><strong>Original:</strong> {item.original}</p>
                            <p><strong>Ihre Version:</strong> {item.interpretation}</p>
                            <p><strong>Vorschlag:</strong> {item.suggestion}</p>
                        </li>
                    ))}
                </ul>
            ) : <p>Sehr gut! Es wurden keine signifikanten Fehler gefunden.</p>}
        </div>
    );
};

const DialoguePractice = ({
    originalText,
    settings,
    isLoading,
    loadingMessage,
    onDialogueFinished,
    onPremiumVoiceAuthError,
} : {
    originalText: string;
    settings: Settings;
    isLoading: boolean;
    loadingMessage: string;
    onDialogueFinished: (results: StructuredDialogueResult[]) => void;
    onPremiumVoiceAuthError: () => void;
}) => {
    const [dialogue, setDialogue] = useState<DialogueSegment[]>([]);
    const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);
    const [dialogueState, setDialogueState] = useState<DialogueState>('idle');
    const [userInterpretation, setUserInterpretation] = useState('');
    const [results, setResults] = useState<StructuredDialogueResult[]>([]);
    const [showText, setShowText] = useState(false);

    const recognitionRef = useRef<SpeechRecognition | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    // --- Parse and Setup Dialogue ---
    useEffect(() => {
        if (!originalText) return;
        
        const segments: DialogueSegment[] = originalText
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.startsWith('Frage') || line.startsWith('Antwort'))
            .map(line => {
                const isQuestion = line.startsWith('Frage');
                return {
                    type: isQuestion ? 'Frage' : 'Antwort',
                    text: line.replace(/^(Frage|Antwort)\s*\d*:\s*/, '').trim(),
                    lang: isQuestion ? settings.sourceLang : settings.targetLang,
                };
            });
        
        setDialogue(segments);
        setDialogueState('ready');
    }, [originalText, settings]);

    // --- Setup Speech Recognition ---
    useEffect(() => {
        const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognitionAPI) {
            console.error("Speech Recognition API is not supported in this browser.");
            return;
        }

        const recognition = new SpeechRecognitionAPI();
        recognition.continuous = true;
        recognition.interimResults = false;
        
        recognition.onresult = (event: SpeechRecognitionEvent) => {
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript + ' ';
                }
            }
            setUserInterpretation(prev => prev + finalTranscript);
        };
        
        recognition.onend = () => {
             if (dialogueState === 'recording') {
                setDialogueState('synthesizing'); // Move to next state automatically
             }
        };

        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
            console.error("Speech recognition error:", event.error, event.message);
             if (dialogueState === 'recording') {
                setDialogueState('synthesizing'); // Try to continue
             }
        };

        recognitionRef.current = recognition;
        
        return () => recognitionRef.current?.stop();

    }, [dialogueState]);


    const startOrContinueDialogue = async () => {
        if (currentSegmentIndex >= dialogue.length) {
            setDialogueState('finished');
            onDialogueFinished(results);
            return;
        }

        setShowText(false);
        const currentSegment = dialogue[currentSegmentIndex];
        
        // Synthesize and play the audio
        setDialogueState('synthesizing');
        try {
            let audioContent;
             if (settings.voiceQuality === 'Premium') {
                try {
                    const apiKey = process.env.API_KEY;
                    if (!apiKey) throw new TtsAuthError('API key for TTS is not configured.');
                    audioContent = await synthesizeSpeechGoogleCloud(currentSegment.text, currentSegment.lang, apiKey);
                } catch (error) {
                     console.error("Premium voice synthesis failed, falling back to standard.", error);
                     if (error instanceof TtsAuthError) onPremiumVoiceAuthError();
                     // Fallback will happen below
                }
            }
            
            if (audioContent) {
                const audioBlob = new Blob([Uint8Array.from(atob(audioContent), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
                const audioUrl = URL.createObjectURL(audioBlob);
                if (!audioRef.current) audioRef.current = new Audio();
                audioRef.current.src = audioUrl;
                audioRef.current.onended = () => setDialogueState('waiting_for_record');
                await audioRef.current.play();
                setDialogueState('playing');
            } else { // Standard voice fallback
                const utterance = new SpeechSynthesisUtterance(currentSegment.text);
                utterance.lang = LANGUAGE_CODES[currentSegment.lang];
                utterance.onend = () => setDialogueState('waiting_for_record');
                window.speechSynthesis.speak(utterance);
                setDialogueState('playing');
            }

        } catch (error) {
             console.error("Failed to play audio for dialogue segment:", error);
             setDialogueState('waiting_for_record'); // Skip playing if it fails
        }
    };
    
    // --- State Machine ---
    useEffect(() => {
        if (dialogueState === 'starting') {
            startOrContinueDialogue();
        } else if (dialogueState === 'waiting_for_record') {
             // Determine language for recording
            const isQuestion = dialogue[currentSegmentIndex].type === 'Frage';
            const recordingLang = isQuestion ? settings.targetLang : settings.sourceLang;
            
            if (recognitionRef.current) {
                recognitionRef.current.lang = LANGUAGE_CODES[recordingLang];
                recognitionRef.current.start();
                setDialogueState('recording');
            }
        } else if (dialogueState === 'synthesizing' && currentSegmentIndex > 0) {
            // This state is entered after recording stops
            const lastSegment = dialogue[currentSegmentIndex - 1];
            const newResult: StructuredDialogueResult = {
                originalSegment: lastSegment,
                userInterpretation: userInterpretation.trim(),
                interpretationLang: lastSegment.type === 'Frage' ? settings.targetLang : settings.sourceLang
            };
            setResults(prev => [...prev, newResult]);
            setUserInterpretation('');
            startOrContinueDialogue(); // Play next segment
        } else if (dialogueState === 'finished') {
             const finalResult: StructuredDialogueResult = {
                originalSegment: dialogue[currentSegmentIndex - 1],
                userInterpretation: userInterpretation.trim(),
                interpretationLang: dialogue[currentSegmentIndex -1].type === 'Frage' ? settings.targetLang : settings.sourceLang
            };
            const allResults = [...results, finalResult];
            onDialogueFinished(allResults);
        }

    }, [dialogueState]);


    // After first recording, advance the index
    useEffect(() => {
        if (dialogueState === 'recording') {
            setCurrentSegmentIndex(prev => prev + 1);
        }
    }, [dialogueState]);


    const getStatusText = () => {
        switch (dialogueState) {
            case 'idle': return 'Bereit zum Start.';
            case 'ready': return 'Drücken Sie "Start", um den Dialog zu beginnen.';
            case 'starting':
            case 'synthesizing': return `Segment ${currentSegmentIndex + 1}/${dialogue.length} wird vorbereitet...`;
            case 'playing': return `Segment ${currentSegmentIndex + 1}/${dialogue.length} wird abgespielt...`;
            case 'waiting_for_record': return 'Bereit zur Aufnahme Ihrer Antwort...';
            case 'recording': return `Aufnahme läuft... (Segment ${currentSegmentIndex}/${dialogue.length})`;
            case 'finished': return 'Dialog beendet. Ergebnisse werden angezeigt.';
            default: return '';
        }
    };
    
    const currentSegment = dialogue[currentSegmentIndex - 1];

    return (
        <section className="panel practice-area">
            {isLoading && dialogueState === 'idle' && (
                <div className="loading-overlay">
                    <div className="spinner"></div>
                    <p>{loadingMessage}</p>
                </div>
            )}
            <div className="dialogue-practice-container">
                <div className="dialogue-status">{getStatusText()}</div>
                
                 <div className="current-segment-display">
                     {dialogueState === 'idle' || dialogueState === 'ready' || dialogueState === 'finished' ? (
                        <p>Warten auf Dialogstart...</p>
                     ) : (
                        <div className="dialogue-text-container">
                          {currentSegment && (
                            <>
                                <p className="segment-text-hidden">
                                  {currentSegment.type} ({currentSegment.lang})
                                </p>
                                {showText ? (
                                    <p className="segment-text">{currentSegment.text}</p>
                                ) : (
                                    <button className="btn btn-secondary btn-show-text" onClick={() => setShowText(true)}>Text anzeigen</button>
                                )}
                            </>
                          )}
                        </div>
                     )}
                </div>

                 <footer className="practice-footer">
                    {dialogueState === 'ready' && (
                        <button className="btn btn-primary" onClick={() => setDialogueState('starting')}>
                            Dialog starten
                        </button>
                    )}
                    {(dialogueState === 'playing' || dialogueState === 'recording') && (
                        <button 
                            className="btn-record recording"
                            aria-label="Aufnahme läuft"
                            disabled
                        >
                           <span className="mic-icon"></span>
                        </button>
                    )}
                </footer>
            </div>
        </section>
    );
};

const DialogueResults = ({
    feedback,
    getFeedback,
    isLoading,
    loadingMessage,
    error,
    structuredResults,
    originalText,
    userTranscript,
}: {
    feedback: Feedback | null;
    getFeedback: (textToCompare: string) => void;
    isLoading: boolean;
    loadingMessage: string;
    error: string | null;
    structuredResults: StructuredDialogueResult[] | null;
    originalText: string;
    userTranscript: string;
}) => {
    const [activeTab, setActiveTab] = useState<'transcript' | 'feedback'>('transcript');

    const handleGetFeedback = () => {
        // We need to pass the "correct" interpretation for the AI to compare against.
        // The original text contains both questions and answers. The user transcript contains only the interpretations.
        // We will pass the full original dialogue script. The AI is instructed to compare the user's interpretation to the relevant parts.
        getFeedback(originalText);
    };

    return (
         <section className="panel practice-area">
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
                     <div className="text-area structured-transcript">
                         {structuredResults?.map((result, index) => (
                             <div key={index} className="transcript-segment">
                                <div className="transcript-segment-header">
                                    <h4>
                                       {result.originalSegment.type} {Math.floor(index / 2) + 1}
                                    </h4>
                                </div>
                                <p className="transcript-segment-original">
                                    <strong>Original ({result.originalSegment.lang}):</strong> {result.originalSegment.text}
                                </p>
                                <p className="transcript-segment-user">
                                    <strong>Ihre Verdolmetschung ({result.interpretationLang}):</strong>{' '}
                                    {result.userInterpretation || <em>(keine Aufnahme)</em>}
                                </p>
                             </div>
                         ))}
                     </div>
                 )}
                {activeTab === 'feedback' && feedback && (
                     <FeedbackDisplay feedback={feedback} />
                 )}
             </div>

             <footer className="practice-footer">
                <button className="btn btn-primary" onClick={handleGetFeedback} disabled={!userTranscript || isLoading}>
                    Gesamt-Feedback anfordern
                </button>
             </footer>
         </section>
    );
};


const container = document.getElementById('root');
if (container) {
    const root = createRoot(container);
    root.render(<App />);
}