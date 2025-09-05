

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
      // Fix: Combine and set the full transcript in the parent state when the dialogue finishes.
      const fullTranscript = results.map(r => r.userInterpretation).join('\n\n');
      setUserTranscript(fullTranscript);
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
              // Fix: Removed props `originalText`, `userTranscript`, and `loadingMessage`
              // as they are not defined on the `DialogueResults` component. The necessary data
              // is now derived from other state or props within the component.
              <DialogueResults
                feedback={feedback}
                getFeedback={getFeedback}
                isLoading={isLoading}
                error={error}
                structuredResults={structuredDialogueResults}
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
                        <label htmlFor="topic">Thema</label>
                        <input
                            type="text"
                            id="topic"
                            name="topic"
                            className="form-control"
                            value={currentSettings.topic}
                            onChange={handleChange}
                            disabled={disabled}
                            placeholder="z.B. Erneuerbare Energien"
                        />
                    </div>
                )}
                {currentSettings.mode === 'Gesprächsdolmetschen' && currentSettings.sourceType === 'ai' && (
                    <div className="form-group">
                        <label htmlFor="qaLength">Länge der Fragen/Antworten</label>
                        <select id="qaLength" name="qaLength" className="form-control" value={currentSettings.qaLength} onChange={handleChange} disabled={disabled}>
                            {QA_LENGTHS.map(l => <option key={l} value={l}>{l}</option>)}
                        </select>
                    </div>
                )}
                {speechModes.includes(currentSettings.mode) && currentSettings.sourceType === 'ai' && (
                     <div className="form-group">
                        <label htmlFor="speechLength">Länge der Rede</label>
                        <select id="speechLength" name="speechLength" className="form-control" value={currentSettings.speechLength} onChange={handleChange} disabled={disabled}>
                            {SPEECH_LENGTHS.map(l => <option key={l} value={l}>{l}</option>)}
                        </select>
                    </div>
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


interface PracticeAreaProps {
  isLoading: boolean;
  loadingMessage: string;
  originalText: string;
  onRecordingFinished: (transcript: string) => void;
  onTranscriptChange: (newTranscript: string) => void;
  getFeedback: (textToCompare?: string) => void;
  userTranscript: string;
  feedback: Feedback | null;
  error: string | null;
  settings: Settings;
  exerciseStarted: boolean;
  onPremiumVoiceAuthError: () => void;
  onDialogueFinished: (results: StructuredDialogueResult[]) => void;
}

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
}: PracticeAreaProps) => {
    const [activeTab, setActiveTab] = useState<PracticeAreaTab>('original');
    
    const [transcript, setTranscript] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [recordingStatus, setRecordingStatus] = useState('Bereit zum Aufnehmen. Klicken Sie auf den Button.');
    const recognitionRef = useRef<SpeechRecognition | null>(null);
    const manualStopRef = useRef(false); // Flag to distinguish manual stop from timeout

    // --- Audio Playback State ---
    const [isPlaying, setIsPlaying] = useState(false);
    const [playbackProgress, setPlaybackProgress] = useState(0);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    
    // --- Dialogue Mode State ---
    const [dialogueSegments, setDialogueSegments] = useState<DialogueSegment[]>([]);
    const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);
    const [dialogueState, setDialogueState] = useState<DialogueState>('idle');
    const [showDialogueText, setShowDialogueText] = useState(false);
    const dialogueResultsRef = useRef<StructuredDialogueResult[]>([]);

    useEffect(() => {
        if (settings.mode === 'Gesprächsdolmetschen' && originalText) {
            const segments: DialogueSegment[] = originalText.split('\n').filter(line => line.trim() !== '').map(line => {
                const isQuestion = line.startsWith('Frage');
                const text = line.replace(/^(Frage \d:|Antwort \d:)\s*/, '');
                const lang = isQuestion ? settings.sourceLang : settings.targetLang;
                return { type: isQuestion ? 'Frage' : 'Antwort', text, lang };
            });
            setDialogueSegments(segments);
            setDialogueState('ready');
        }
    }, [originalText, settings.mode, settings.sourceLang, settings.targetLang]);


    const handlePlayPause = useCallback(async () => {
        if (isPlaying) {
            audioRef.current?.pause();
            setIsPlaying(false);
            return;
        }

        setIsPlaying(true);
        if (audioRef.current && audioRef.current.src && audioRef.current.currentTime > 0) {
            audioRef.current.play();
            return;
        }

        try {
            let audioBase64: string;
            if (settings.voiceQuality === 'Premium') {
                 if (!process.env.API_KEY) {
                    throw new Error("Google Cloud API Key not provided for Premium voice.");
                 }
                try {
                    audioBase64 = await synthesizeSpeechGoogleCloud(originalText, settings.sourceLang, process.env.API_KEY);
                } catch (e) {
                    if (e instanceof TtsAuthError) {
                        console.warn("Premium voice authentication failed, falling back to standard voice.", e.message);
                        onPremiumVoiceAuthError();
                        // Fallback logic is handled by re-rendering with disabled premium option,
                        // so we just synthesize with standard here for the current attempt.
                        const utterance = new SpeechSynthesisUtterance(originalText);
                        utterance.lang = LANGUAGE_CODES[settings.sourceLang];
                        speechSynthesis.speak(utterance);
                        // Since there's no easy way to get audio data from browser API, we can't show progress.
                        // We'll manage state based on events.
                        utterance.onend = () => setIsPlaying(false);
                        utterance.onerror = () => setIsPlaying(false);
                        return; // Exit here
                    }
                    throw e; // Re-throw other errors
                }
            } else {
                // Standard browser synthesis - no audio data to return, handled by events
                const utterance = new SpeechSynthesisUtterance(originalText);
                utterance.lang = LANGUAGE_CODES[settings.sourceLang];
                speechSynthesis.speak(utterance);
                utterance.onend = () => {
                    setIsPlaying(false);
                    setPlaybackProgress(0);
                };
                 utterance.onerror = (e) => {
                    console.error("SpeechSynthesis Error:", e);
                    setIsPlaying(false);
                };
                return;
            }
            
            const audioSrc = `data:audio/mp3;base64,${audioBase64}`;
            if (audioRef.current) {
                audioRef.current.src = audioSrc;
                audioRef.current.play();
            }

        } catch (error) {
            console.error("Error synthesizing speech:", error);
            setIsPlaying(false);
        }
    }, [isPlaying, originalText, settings.sourceLang, settings.voiceQuality, onPremiumVoiceAuthError]);
    
     useEffect(() => {
        const audio = new Audio();
        audioRef.current = audio;

        const handleTimeUpdate = () => {
            if (audio.duration) {
                setPlaybackProgress((audio.currentTime / audio.duration) * 100);
            }
        };

        const handleEnded = () => {
            setIsPlaying(false);
            setPlaybackProgress(0);
        };

        audio.addEventListener('timeupdate', handleTimeUpdate);
        audio.addEventListener('ended', handleEnded);

        return () => {
            audio.removeEventListener('timeupdate', handleTimeUpdate);
            audio.removeEventListener('ended', handleEnded);
            audio.pause();
            audioRef.current = null;
        };
    }, []);

    // Setup Speech Recognition
    useEffect(() => {
        const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognitionAPI) {
            setRecordingStatus("Speech Recognition wird von diesem Browser nicht unterstützt.");
            return;
        }

        const recognition = new SpeechRecognitionAPI();
        recognitionRef.current = recognition;
        recognition.lang = LANGUAGE_CODES[settings.targetLang];
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onresult = (event: SpeechRecognitionEvent) => {
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                }
            }
            if(finalTranscript){
                setTranscript(prev => prev + finalTranscript);
            }
        };

        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
            console.error('Speech recognition error:', event.error, event.message);
            setRecordingStatus(`Fehler: ${event.error}`);
            setIsRecording(false);
        };
        
        recognition.onend = () => {
            if (manualStopRef.current) {
                // Manual stop by user, proceed to finish the recording.
                setIsRecording(false);
                onRecordingFinished(transcript);
            } else {
                // Automatic stop (timeout), restart the recognition to continue listening.
                try {
                    recognitionRef.current?.start();
                } catch (e) {
                    console.error("Failed to restart speech recognition:", e);
                    // If restart fails, then we must stop.
                    setIsRecording(false);
                }
            }
        };

        return () => {
            recognition.stop();
        };
    }, [settings.targetLang, onRecordingFinished, transcript]);
    
    const handleRecordClick = () => {
        if (!recognitionRef.current) return;

        if (!isRecording) {
            manualStopRef.current = false; // Reset flag on start
            setTranscript('');
            recognitionRef.current.start();
            setRecordingStatus('Aufnahme läuft... Klicken Sie zum Stoppen.');
            setIsRecording(true);
        } else {
            manualStopRef.current = true; // Set flag on manual stop
            recognitionRef.current.stop();
        }
    };

    const handleDialogueAction = async () => {
        const segment = dialogueSegments[currentSegmentIndex];
        if (!segment) return;
        
        switch(dialogueState) {
            case 'ready':
            case 'finished':
                setDialogueState('synthesizing');
                setShowDialogueText(true);
                try {
                    let audioBase64: string;
                     if (settings.voiceQuality === 'Premium') {
                        if (!process.env.API_KEY) throw new Error("API key not available for Premium voice.");
                        audioBase64 = await synthesizeSpeechGoogleCloud(segment.text, segment.lang, process.env.API_KEY);
                        const audioSrc = `data:audio/mp3;base64,${audioBase64}`;
                        if (audioRef.current) {
                            audioRef.current.src = audioSrc;
                            audioRef.current.onended = () => setDialogueState('waiting_for_record');
                            audioRef.current.play();
                            setDialogueState('playing');
                        }
                    } else {
                        const utterance = new SpeechSynthesisUtterance(segment.text);
                        utterance.lang = LANGUAGE_CODES[segment.lang];
                        utterance.onstart = () => setDialogueState('playing');
                        utterance.onend = () => setDialogueState('waiting_for_record');
                        speechSynthesis.speak(utterance);
                    }
                } catch (e) {
                     console.error("Dialogue speech synthesis failed:", e);
                     if (e instanceof TtsAuthError) onPremiumVoiceAuthError();
                     setDialogueState('waiting_for_record'); // Fallback to allow user to proceed
                }
                break;
            
            case 'waiting_for_record':
                 if (!recognitionRef.current) return;
                 setTranscript(''); // Clear previous transcript
                 recognitionRef.current.lang = LANGUAGE_CODES[segment.type === 'Frage' ? settings.targetLang : settings.sourceLang];
                 recognitionRef.current.start();
                 setDialogueState('recording');
                 break;

            case 'recording':
                if (!recognitionRef.current) return;
                recognitionRef.current.stop();

                dialogueResultsRef.current.push({
                    originalSegment: segment,
                    userInterpretation: transcript,
                    interpretationLang: segment.type === 'Frage' ? settings.targetLang : settings.sourceLang
                });
                
                if (currentSegmentIndex < dialogueSegments.length - 1) {
                    setCurrentSegmentIndex(prev => prev + 1);
                    setDialogueState('ready');
                    setShowDialogueText(false);
                } else {
                    onDialogueFinished(dialogueResultsRef.current);
                    setDialogueState('idle');
                }
                break;
        }
    };
    
    const handleGetTotalFeedback = () => {
        if (!dialogueResultsRef.current) return;
        const fullOriginal = dialogueResultsRef.current.map(r => r.originalSegment.text).join('\n');
        const fullTranscript = dialogueResultsRef.current.map(r => r.userInterpretation).join('\n');
        onTranscriptChange(fullTranscript); // Set the full transcript in parent for feedback
        getFeedback(fullOriginal); // Get feedback on the combined text
    };


    if (!exerciseStarted) {
        return (
            <aside className="panel practice-area">
                <div className="placeholder">
                    <h2>Willkommen beim Dolmetsch-Trainer Pro</h2>
                    <p>Passen Sie die Einstellungen an und klicken Sie auf "Übung starten", um zu beginnen.</p>
                </div>
            </aside>
        );
    }
    
    if (settings.mode === 'Gesprächsdolmetschen') {
        const segment = dialogueSegments[currentSegmentIndex];
        const getButtonText = () => {
            switch(dialogueState) {
                case 'ready': return `Segment ${currentSegmentIndex + 1}/${dialogueSegments.length} starten`;
                case 'synthesizing': return 'Audio wird generiert...';
                case 'playing': return 'Audio wird abgespielt...';
                case 'waiting_for_record': return 'Aufnahme starten';
                case 'recording': return 'Aufnahme beenden';
                case 'finished': return `Nächstes Segment (${currentSegmentIndex + 2}/${dialogueSegments.length}) starten`;
                default: return 'Aktion';
            }
        };

        return (
             <div className="panel practice-area">
                {isLoading && (
                    <div className="loading-overlay">
                        <div className="spinner"></div>
                        <p>{loadingMessage}</p>
                    </div>
                )}
                <h2>Gesprächsdolmetschen</h2>
                <div className="dialogue-practice-container">
                    <div className="dialogue-status">
                        {segment ? `${segment.type} auf ${segment.lang}` : "Übung beendet"}
                    </div>
                    <div className="current-segment-display">
                      {dialogueState === 'idle' ? (
                          <div className="dialogue-text-container">
                              <p className="segment-text-hidden">Die Übung ist abgeschlossen.</p>
                              <button className="btn btn-secondary" onClick={handleGetTotalFeedback}>
                                  Gesamtfeedback anfordern
                              </button>
                          </div>
                      ) : (
                        <div className="dialogue-text-container">
                            {showDialogueText ? (
                                <p className="segment-text">{segment?.text}</p>
                            ) : (
                                <>
                                 <p className="segment-text-hidden">Der Text ist verborgen, um das Zuhören zu fördern.</p>
                                 <button className="btn btn-secondary btn-show-text" onClick={() => setShowDialogueText(true)}>Text anzeigen</button>
                                </>
                            )}
                        </div>
                       )}
                    </div>
                </div>
                <div className="practice-footer">
                    <p className="recording-status-text">
                        {dialogueState === 'recording' ? 'Aufnahme läuft...' : ''}
                        {dialogueState === 'waiting_for_record' ? `Bereit zur Aufnahme auf ${segment.type === 'Frage' ? settings.targetLang : settings.sourceLang}` : ''}
                    </p>
                    <button 
                        className={`btn-record ${dialogueState === 'recording' ? 'recording' : ''}`} 
                        onClick={handleDialogueAction} 
                        disabled={dialogueState === 'synthesizing' || dialogueState === 'playing' || dialogueState === 'idle'}
                        aria-label={getButtonText()}
                    >
                        <div className="mic-icon"></div>
                    </button>
                    <span>{getButtonText()}</span>
                </div>
            </div>
        )
    }

    return (
        <div className="panel practice-area">
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
                >
                    Originaltext
                </button>
                 <button 
                    className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`}
                    onClick={() => setActiveTab('transcript')}
                    disabled={!userTranscript && !isRecording}
                >
                    Ihre Verdolmetschung
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
                    <>
                        {["Vortragsdolmetschen", "Simultandolmetschen", "Shadowing"].includes(settings.mode) && (
                           <div className="controls-bar">
                             <button onClick={handlePlayPause} disabled={isLoading}>
                               {isPlaying ? 'Pause' : 'Abspielen'}
                             </button>
                             <div style={{width: '100%', backgroundColor: '#e9ecef', borderRadius: '5px'}}>
                               <div style={{width: `${playbackProgress}%`, height: '8px', backgroundColor: 'var(--primary-color)', borderRadius: '5px'}}></div>
                             </div>
                           </div>
                        )}
                        <div className="text-area">
                           <p>{originalText}</p>
                        </div>
                    </>
                )}

                {activeTab === 'transcript' && (
                   <>
                     <div className="text-area">
                       <textarea 
                         className="text-area-editor"
                         value={userTranscript || transcript}
                         onChange={(e) => onTranscriptChange(e.target.value)}
                         placeholder={isRecording ? "Transkript wird hier angezeigt..." : "Hier erscheint Ihr Transkript nach der Aufnahme."}
                         readOnly={isRecording}
                       />
                     </div>
                     <div style={{marginTop: 'auto', paddingTop: '1rem'}}>
                       <button className="btn btn-primary" onClick={() => getFeedback()} disabled={isLoading || !userTranscript}>
                         Feedback anfordern
                       </button>
                     </div>
                   </>
                )}

                {activeTab === 'feedback' && feedback && (
                    <div className="feedback-content text-area">
                        <h3>Zusammenfassung</h3>
                        <p>{feedback.summary}</p>
                        <h3>Bewertung</h3>
                        <RatingsTable ratings={feedback.ratings} />
                        <h3>Fehleranalyse</h3>
                        <ErrorAnalysisList items={feedback.errorAnalysis} />
                    </div>
                )}
            </div>

            {settings.mode !== 'Stegreifübersetzen' && (
                 <div className="practice-footer">
                     <p className="recording-status-text">{recordingStatus}</p>
                     <button
                        className={`btn-record ${isRecording ? 'recording' : ''}`}
                        onClick={handleRecordClick}
                        disabled={isLoading || (settings.mode.includes("dolmetschen") && isPlaying)}
                        aria-label={isRecording ? 'Aufnahme stoppen' : 'Aufnahme starten'}
                     >
                         <div className="mic-icon"></div>
                     </button>
                 </div>
            )}
        </div>
    );
};

const RatingsTable = ({ ratings }: { ratings: Feedback['ratings'] }) => {
  const renderStars = (rating: number) => {
    return Array.from({ length: 10 }, (_, i) => (
      <span key={i} className={`star ${i < rating ? 'filled' : ''}`}>★</span>
    ));
  };

  return (
    <table className="ratings-table">
      <tbody>
        <tr>
          <td>Inhaltliche Korrektheit</td>
          <td>{renderStars(ratings.content)}</td>
        </tr>
        <tr>
          <td>Ausdruck & Stil</td>
          <td>{renderStars(ratings.expression)}</td>
        </tr>
        <tr>
          <td>Terminologie</td>
          <td>{renderStars(ratings.terminology)}</td>
        </tr>
      </tbody>
    </table>
  );
};

const ErrorAnalysisList = ({ items }: { items: ErrorAnalysisItem[] }) => {
  if (items.length === 0) {
    return <p>Sehr gut! Es wurden keine signifikanten Fehler gefunden.</p>;
  }
  return (
    <ul className="error-analysis-list">
      {items.map((item, index) => (
        <li key={index}>
          <p><strong>Original:</strong> {item.original}</p>
          <p><strong>Ihre Version:</strong> {item.interpretation}</p>
          <p><strong>Vorschlag:</strong> {item.suggestion}</p>
        </li>
      ))}
    </ul>
  );
};

const DialogueResults = ({
  structuredResults,
  getFeedback,
  feedback,
  isLoading,
  error,
}: {
  structuredResults: StructuredDialogueResult[] | null;
  getFeedback: (textToCompare?: string) => void;
  feedback: Feedback | null;
  isLoading: boolean;
  error: string | null;
}) => {
    
    const handleGetTotalFeedback = () => {
        if (!structuredResults) return;
        const fullOriginal = structuredResults.map(r => r.originalSegment.text).join('\n\n');
        // The user transcript is already combined in the parent state, so we just trigger the feedback
        getFeedback(fullOriginal);
    };

    return (
        <div className="panel practice-area">
             {isLoading && (
                <div className="loading-overlay">
                    <div className="spinner"></div>
                </div>
            )}
            <div className="tabs">
                 <button className="tab-btn active">Ergebnisse</button>
                 <button className={`tab-btn ${feedback ? 'active' : ''}`} disabled={!feedback}>Feedback</button>
            </div>
             <div className="tab-content">
              <div className="dialogue-results-wrapper">
                    <div className="text-area">
                        {error && <div className="error-banner">{error}</div>}
                        {feedback ? (
                             <div className="feedback-content">
                                <h3>Zusammenfassung</h3>
                                <p>{feedback.summary}</p>
                                <h3>Bewertung</h3>
                                <RatingsTable ratings={feedback.ratings} />
                                <h3>Fehleranalyse</h3>
                                <ErrorAnalysisList items={feedback.errorAnalysis} />
                            </div>
                        ) : (
                            <div className="structured-transcript">
                            {structuredResults?.map((result, index) => (
                                <div key={index} className="transcript-segment">
                                <div className="transcript-segment-header">
                                    <h4>{result.originalSegment.type} {Math.floor(index / 2) + 1}</h4>
                                </div>
                                <p className="transcript-segment-original">
                                    <strong>Original ({result.originalSegment.lang}):</strong> {result.originalSegment.text}
                                </p>
                                <p className="transcript-segment-user">
                                    <strong>Ihre Verdolmetschung ({result.interpretationLang}):</strong> {result.userInterpretation.trim() ? result.userInterpretation : <em>Keine Aufnahme für dieses Segment.</em>}
                                </p>
                                </div>
                            ))}
                            </div>
                        )}
                    </div>
                    {!feedback && (
                        <div className="settings-footer">
                             <button className="btn btn-primary" onClick={handleGetTotalFeedback} disabled={isLoading}>
                                Gesamtfeedback anfordern
                            </button>
                        </div>
                    )}
                </div>
             </div>
        </div>
    );
};



const container = document.getElementById('root');
if (container) {
    const root = createRoot(container);
    root.render(<App />);
}