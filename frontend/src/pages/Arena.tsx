import { useState, useRef, useCallback, useEffect } from "react";
import { Sidebar, SidebarBody, SidebarLink } from "@/components/ui/sidebar";
import { LayoutDashboard, MessageSquare, Settings, LogOut, Bot, Languages, Volume2, VolumeX } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { AIInput } from "@/components/ui/ai-input";
import { useToast } from "@/hooks/use-toast";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  translatedContent?: {
    en?: string;
    hi?: string;
    mr?: string;
  };
  currentLanguage?: 'en' | 'hi' | 'mr';
}

interface ArenaMessage {
  id: string;
  gemini: Message;
  gpt: Message;
  claude: Message;
  // Add a field to track if this is a variable collection question
  isVariableQuestion?: boolean;
}

// New component for typing simulation
const TypingSimulation = ({ content }: { content: string }) => {
  const [displayedContent, setDisplayedContent] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (currentIndex < content.length) {
      intervalRef.current = setInterval(() => {
        setDisplayedContent(prev => prev + content[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, 30); // Adjust typing speed here (milliseconds per character)
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [currentIndex, content]);

  return <span>{displayedContent}</span>;
};

export default function Arena() {
  const [messages, setMessages] = useState<ArenaMessage[]>([]);
  const [open, setOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  // State to track which message is currently being spoken
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);
  // Ref to keep track of the audio element
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Toast hook for error notifications
  const { toast } = useToast();
  // Cache for pre-fetched audio
  const audioCache = useRef(new Map<string, Blob>());
  // Ref to store available voices
  const voicesRef = useRef<SpeechSynthesisVoice[]>([]);

  // Function to detect language based on text content
  const detectLanguage = (text: string): 'en' | 'hi' | 'mr' => {
    // Check for Hindi characters (Devanagari script)
    const hindiRegex = /[\u0900-\u097F]/;
    // Check for Marathi characters (Devanagari script + specific Marathi characters)
    const marathiRegex = /[\u0900-\u097F\uA8E0-\uA8FF]/;
    
    // Count Hindi and Marathi characters
    const hindiChars = (text.match(hindiRegex) || []).length;
    const marathiChars = (text.match(marathiRegex) || []).length;
    
    // If we have significant Hindi characters
    if (hindiChars > 5) {
      // If we have more Marathi specific characters, it's likely Marathi
      if (marathiChars > hindiChars * 0.1) {
        return 'mr';
      }
      return 'hi';
    }
    
    // Default to English
    return 'en';
  };

  // Function to load available voices
  const loadVoices = useCallback(() => {
    if ('speechSynthesis' in window) {
      // Get the available voices
      const voices = window.speechSynthesis.getVoices();
      voicesRef.current = voices;
      
      // If voices are not yet loaded, wait for the voiceschanged event
      if (voices.length === 0) {
        window.speechSynthesis.onvoiceschanged = () => {
          voicesRef.current = window.speechSynthesis.getVoices();
        };
      }
    }
  }, []);

  // Load voices when component mounts
  useEffect(() => {
    loadVoices();
  }, [loadVoices]);

  const links = [
    {
      label: "Dashboard",
      href: "/",
      icon: (
        <LayoutDashboard className="text-gray-600 dark:text-gray-300 h-5 w-5 flex-shrink-0" />
      ),
    },
    {
      label: "Chat",
      href: "/chat",
      icon: (
        <MessageSquare className="text-gray-600 dark:text-gray-300 h-5 w-5 flex-shrink-0" />
      ),
    },
    {
      label: "Arena",
      href: "/arena",
      icon: (
        <Bot className="text-gray-600 dark:text-gray-300 h-5 w-5 flex-shrink-0" />
      ),
    },
    {
      label: "Settings",
      href: "/settings",
      icon: (
        <Settings className="text-gray-600 dark:text-gray-300 h-5 w-5 flex-shrink-0" />
      ),
    },
    {
      label: "Logout",
      href: "/",
      icon: (
        <LogOut className="text-gray-600 dark:text-gray-300 h-5 w-5 flex-shrink-0" />
      ),
    },
  ];

  // Function to translate a message
  const translateMessage = async (modelType: 'gemini' | 'gpt' | 'claude', messageId: string, targetLanguage: 'en' | 'hi' | 'mr') => {
    // Find the arena message containing this message
    const arenaMessageIndex = messages.findIndex(arenaMsg => 
      arenaMsg[modelType].id === messageId
    );
    
    if (arenaMessageIndex === -1) return;

    const arenaMessage = messages[arenaMessageIndex];
    const message = arenaMessage[modelType];
    
    if (!message || message.role !== "assistant") return;

    // Check if already translated
    if (message.translatedContent?.[targetLanguage]) {
      // Update current language
      const updatedMessages = [...messages];
      updatedMessages[arenaMessageIndex] = {
        ...arenaMessage,
        [modelType]: {
          ...message,
          currentLanguage: targetLanguage
        }
      };
      setMessages(updatedMessages);
      return;
    }

    try {
      const baseUrl = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';
      const endpoint = window.location.hostname === 'localhost' ? `${baseUrl}/translate` : '/api/translate';
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: message.content,
          target_language: targetLanguage,
          source_language: message.currentLanguage || 'en'
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const translatedText = data.translated_text;

      // Update message with translated content
      const updatedMessages = [...messages];
      updatedMessages[arenaMessageIndex] = {
        ...arenaMessage,
        [modelType]: {
          ...message,
          translatedContent: {
            ...message.translatedContent,
            [targetLanguage]: translatedText
          },
          currentLanguage: targetLanguage
        }
      };
      setMessages(updatedMessages);
    } catch (error) {
      console.error("Error translating message:", error);
      toast({
        title: "Translation Error",
        description: "Failed to translate the message. Please try again.",
        variant: "destructive",
      });
    }
  };

  // Function to get display content for a message
  const getDisplayContent = (message: Message) => {
    if (message.role !== "assistant") return message.content;
    
    const currentLang = message.currentLanguage || 'en';
    if (currentLang === 'en') return message.content;
    
    return message.translatedContent?.[currentLang] || message.content;
  };

  const handleSubmit = async (value: string) => {
    if (!value.trim() || isLoading) return;

    const newMessageId = Date.now().toString();
    
    // Add user message to chat
    const newUserMessage: ArenaMessage = {
      id: newMessageId,
      gemini: {
        id: `${newMessageId}-gemini`,
        role: "user",
        content: value,
      },
      gpt: {
        id: `${newMessageId}-gpt`,
        role: "user",
        content: value,
      },
      claude: {
        id: `${newMessageId}-claude`,
        role: "user",
        content: value,
      }
    };
    
    setMessages(prev => [...prev, newUserMessage]);
    setIsLoading(true);

    try {
      // For Vercel deployment, use /api/arena endpoint
      // For local development, use port 5000 which is the default for Flask
      const baseUrl = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';
      const arenaEndpoint = window.location.hostname === 'localhost' ? `${baseUrl}/arena` : '/api/arena';
      
      // Call the arena endpoint which will call all three models simultaneously
      const arenaResponse = await fetch(arenaEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          message: value,
          language: 'en',
          translate_to: 'en',
          source_language: 'en',
          user_id: 'arena_user'  // Add user_id for context tracking
        }),
      });

      // Check content type to determine how to process the response
      const contentType = arenaResponse.headers.get('content-type');
      
      if (contentType && contentType.includes('application/json')) {
        // Handle JSON response (final responses or greetings)
        if (!arenaResponse.ok) {
          const errorText = await arenaResponse.text();
          throw new Error(`HTTP error! status: ${arenaResponse.status}, message: ${errorText}`);
        }

        const arenaData = await arenaResponse.json();
        
        const geminiText = arenaData.gemini || "Sorry, Gemini is not available right now.";
        const gptText = arenaData.gpt || "Sorry, GPT is not available right now.";
        const claudeText = arenaData.claude || "Sorry, Claude is not available right now.";
        
        // Detect language of the responses
        const geminiLanguage = detectLanguage(geminiText);
        const gptLanguage = detectLanguage(gptText);
        const claudeLanguage = detectLanguage(claudeText);
        
        // Add AI responses to chat
        const aiResponse: ArenaMessage = {
          id: (Date.now() + 1).toString(),
          gemini: {
            id: `${Date.now() + 1}-gemini`,
            role: "assistant",
            content: geminiText,
            currentLanguage: geminiLanguage
          },
          gpt: {
            id: `${Date.now() + 1}-gpt`,
            role: "assistant",
            content: gptText,
            currentLanguage: gptLanguage
          },
          claude: {
            id: `${Date.now() + 1}-claude`,
            role: "assistant",
            content: claudeText,
            currentLanguage: claudeLanguage
          }
        };
        
        // Start typing simulation for each response
        startTyping(aiResponse.gemini.id, geminiText);
        startTyping(aiResponse.gpt.id, gptText);
        startTyping(aiResponse.claude.id, claudeText);
        
        setMessages(prev => [...prev, aiResponse]);
      } else {
        // Handle plain text response (variable collection questions)
        if (!arenaResponse.ok) {
          const errorText = await arenaResponse.text();
          throw new Error(`HTTP error! status: ${arenaResponse.status}, message: ${errorText}`);
        }
        
        const questionText = await arenaResponse.text();
        
        // For variable collection questions, we'll show the same question for all models
        // In a real implementation, you might want to handle this differently
        const questionResponse: ArenaMessage = {
          id: (Date.now() + 1).toString(),
          gemini: {
            id: `${Date.now() + 1}-gemini`,
            role: "assistant",
            content: questionText,
            currentLanguage: 'en'
          },
          gpt: {
            id: `${Date.now() + 1}-gpt`,
            role: "assistant",
            content: questionText,
            currentLanguage: 'en'
          },
          claude: {
            id: `${Date.now() + 1}-claude`,
            role: "assistant",
            content: questionText,
            currentLanguage: 'en'
          },
          isVariableQuestion: true
        };
        
        setMessages(prev => [...prev, questionResponse]);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: ArenaMessage = {
        id: (Date.now() + 1).toString(),
        gemini: {
          id: `${Date.now() + 1}-gemini`,
          role: "assistant",
          content: "Sorry, I'm having trouble responding right now. Please try again later.",
          currentLanguage: 'en'
        },
        gpt: {
          id: `${Date.now() + 1}-gpt`,
          role: "assistant",
          content: "Sorry, I'm having trouble responding right now. Please try again later.",
          currentLanguage: 'en'
        },
        claude: {
          id: `${Date.now() + 1}-claude`,
          role: "assistant",
          content: "Sorry, I'm having trouble responding right now. Please try again later.",
          currentLanguage: 'en'
        }
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // State for typing simulation
  const [typingContent, setTypingContent] = useState<{[key: string]: string}>({});
  
  // Function to start typing simulation
  const startTyping = (messageId: string, content: string) => {
    setTypingContent(prev => ({ ...prev, [messageId]: content }));
  };
  
  // Function to stop typing simulation
  const stopTyping = (messageId: string) => {
    setTypingContent(prev => {
      const newTyping = { ...prev };
      delete newTyping[messageId];
      return newTyping;
    });
  };

  // Function to clean response content by removing asterisks
  const cleanResponseContent = (content: string) => {
    return content.replace(/\*/g, '');
  };

  // Function to select appropriate voice - modified to always use Heera
  const selectVoice = (text: string) => {
    const voices = voicesRef.current;
    
    if (voices.length > 0) {
      // Always look for Microsoft Heera voice regardless of text content
      const heeraVoice = voices.find(voice => 
        voice.name.includes('Heera') && voice.lang.includes('hi')
      );
      
      return heeraVoice || null;
    }
    
    return null;
  };

  // Function to handle text-to-speech for a message using Web Speech API directly
  const handleTextToSpeech = async (messageId: string, text: string) => {
    // If this message is already being spoken, stop it
    if (speakingMessageId === messageId) {
      try {
        // Stop the speech synthesis
        if (window.speechSynthesis) {
          window.speechSynthesis.cancel();
        }
        
        // Call backend to stop speech (optional)
        const baseUrl = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';
        const stopEndpoint = window.location.hostname === 'localhost' ? `${baseUrl}/stop-speech` : '/api/stop-speech';
        
        await fetch(stopEndpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ messageId }),
        });
      } catch (error) {
        console.error("Error stopping speech:", error);
      }
      setSpeakingMessageId(null);
      return;
    }

    // If another message is being spoken, stop it first
    if (speakingMessageId) {
      try {
        if (window.speechSynthesis) {
          window.speechSynthesis.cancel();
        }
        
        const baseUrl = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';
        const stopEndpoint = window.location.hostname === 'localhost' ? `${baseUrl}/stop-speech` : '/api/stop-speech';
        
        await fetch(stopEndpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ messageId: speakingMessageId }),
        });
      } catch (error) {
        console.error("Error stopping previous speech:", error);
      }
    }

    // Set this message as the one currently being spoken
    setSpeakingMessageId(messageId);

    try {
      // First, get the speech data from the backend
      const baseUrl = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';
      const ttsEndpoint = window.location.hostname === 'localhost' ? `${baseUrl}/text-to-speech` : '/api/text-to-speech';
      
      const response = await fetch(ttsEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          messageId: messageId
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Use Web Speech API directly
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Always use Hindi language and Heera voice for all text
        utterance.lang = 'hi-IN';
        
        // Select Heera voice (this will always be used now)
        const preferredVoice = selectVoice(text);
        if (preferredVoice) {
          utterance.voice = preferredVoice;
        }
        
        // Set speech properties for better quality
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        // Event handlers
        utterance.onend = () => {
          setSpeakingMessageId(null);
        };
        
        utterance.onerror = (event) => {
          console.error("Speech synthesis error:", event);
          setSpeakingMessageId(null);
        };
        
        // Speak the text
        window.speechSynthesis.speak(utterance);
      } else {
        // Fallback if Web Speech API is not supported
        console.error("Web Speech API is not supported in this browser");
        setSpeakingMessageId(null);
      }
    } catch (error) {
      console.error("Error starting speech:", error);
      setSpeakingMessageId(null);
    }
  };

  return (
    <div className="flex flex-col md:flex-row bg-white dark:bg-gray-900 w-full min-h-screen overflow-hidden">
      <Sidebar open={open} setOpen={setOpen}>
        <SidebarBody className="justify-between gap-10">
          <div className="flex flex-col flex-1 overflow-y-auto overflow-x-hidden">
            <Logo open={open} />
            <div className="mt-8 flex flex-col gap-2">
              {links.map((link, idx) => (
                <SidebarLink key={idx} link={link} />
              ))}
            </div>
          </div>
          <div>
            <SidebarLink
              link={{
                label: "AI Doctor",
                href: "/arena",
                icon: (
                  <div className="h-8 w-8 flex-shrink-0 rounded-full bg-blue-600 flex items-center justify-center">
                    <Bot className="h-5 w-5 text-white" />
                  </div>
                ),
              }}
            />
          </div>
        </SidebarBody>
      </Sidebar>
      
      <div className="flex flex-1 flex-col h-full bg-gray-50 dark:bg-gray-800">
        <div className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
          <div className="max-w-7xl mx-auto w-full">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
              {/* Gemini Column */}
              <div className="flex flex-col h-full">
                <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white p-3 md:p-4 rounded-t-xl">
                  <h2 className="text-lg md:text-xl font-bold">Gemini</h2>
                </div>
                <div className="bg-white dark:bg-gray-700 rounded-b-xl p-3 md:p-4 flex-1 min-h-[300px] md:min-h-[400px] flex flex-col">
                  <div className="flex-1 overflow-y-auto mb-4 space-y-4">
                    {messages.map((arenaMessage) => (
                      <div key={arenaMessage.gemini.id} className="mb-3">
                        {arenaMessage.gemini.role === "assistant" ? (
                          <div className="rounded-2xl px-3 py-2 md:px-4 md:py-3 bg-blue-50 dark:bg-blue-900/30 text-gray-800 dark:text-gray-200">
                            <div className="text-sm whitespace-pre-wrap">
                              {typingContent[arenaMessage.gemini.id] ? (
                                <TypingSimulation content={getDisplayContent(arenaMessage.gemini)} />
                              ) : (
                                cleanResponseContent(getDisplayContent(arenaMessage.gemini))
                              )}
                              {/* Speaker button for AI responses */}
                              <div className="mt-2 flex items-center gap-2 flex-wrap">
                                <button
                                  onClick={() => handleTextToSpeech(arenaMessage.gemini.id, getDisplayContent(arenaMessage.gemini))}
                                  className={cn(
                                    "p-2 rounded-full",
                                    speakingMessageId === arenaMessage.gemini.id
                                      ? "bg-red-100 text-red-600 hover:bg-red-200 dark:bg-red-900 dark:text-red-300 dark:hover:bg-red-800"
                                      : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-600 dark:text-gray-300 dark:hover:bg-gray-500"
                                  )}
                                >
                                  {speakingMessageId === arenaMessage.gemini.id ? (
                                    <VolumeX className="h-4 w-4" />
                                  ) : (
                                    <Volume2 className="h-4 w-4" />
                                  )}
                                </button>
                                {/* Translation buttons for AI responses (except variable questions) */}
                                {!arenaMessage.isVariableQuestion && (
                                  <div className="flex flex-wrap gap-1 md:gap-2 ml-1 md:ml-2">
                                    <button
                                      onClick={() => translateMessage('gemini', arenaMessage.gemini.id, 'en')}
                                      className={cn(
                                        "text-xs px-2 py-1 rounded-full border",
                                        arenaMessage.gemini.currentLanguage === 'en' 
                                          ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                          : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                                      )}
                                    >
                                      English
                                    </button>
                                    <button
                                      onClick={() => translateMessage('gemini', arenaMessage.gemini.id, 'hi')}
                                      className={cn(
                                        "text-xs px-2 py-1 rounded-full border",
                                        arenaMessage.gemini.currentLanguage === 'hi' 
                                          ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                          : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                                      )}
                                    >
                                      हिंदी
                                    </button>
                                    <button
                                      onClick={() => translateMessage('gemini', arenaMessage.gemini.id, 'mr')}
                                      className={cn(
                                        "text-xs px-2 py-1 rounded-full border",
                                        arenaMessage.gemini.currentLanguage === 'mr' 
                                          ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                          : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                                      )}
                                    >
                                      मराठी
                                    </button>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="flex items-start gap-2 justify-end">
                            <div className="rounded-2xl px-3 py-2 md:px-4 md:py-3 bg-blue-600 text-white max-w-[80%] md:max-w-[70%]">
                              <p className="text-sm">{arenaMessage.gemini.content}</p>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                    {isLoading && (
                      <div className="rounded-2xl px-3 py-2 md:px-4 md:py-3 bg-blue-50 dark:bg-blue-900/30 text-gray-800 dark:text-gray-200">
                        <p className="text-sm">Thinking...</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* GPT Column */}
              <div className="flex flex-col h-full">
                <div className="bg-gradient-to-r from-green-500 to-green-600 text-white p-3 md:p-4 rounded-t-xl">
                  <h2 className="text-lg md:text-xl font-bold">GPT</h2>
                </div>
                <div className="bg-white dark:bg-gray-700 rounded-b-xl p-3 md:p-4 flex-1 min-h-[300px] md:min-h-[400px] flex flex-col">
                  <div className="flex-1 overflow-y-auto mb-4 space-y-4">
                    {messages.map((arenaMessage) => (
                      <div key={arenaMessage.gpt.id} className="mb-3">
                        {arenaMessage.gpt.role === "assistant" ? (
                          <div className="rounded-2xl px-3 py-2 md:px-4 md:py-3 bg-green-50 dark:bg-green-900/30 text-gray-800 dark:text-gray-200">
                            <div className="text-sm whitespace-pre-wrap">
                              {typingContent[arenaMessage.gpt.id] ? (
                                <TypingSimulation content={getDisplayContent(arenaMessage.gpt)} />
                              ) : (
                                cleanResponseContent(getDisplayContent(arenaMessage.gpt))
                              )}
                              {/* Speaker button for AI responses */}
                              <div className="mt-2 flex items-center gap-2 flex-wrap">
                                <button
                                  onClick={() => handleTextToSpeech(arenaMessage.gpt.id, getDisplayContent(arenaMessage.gpt))}
                                  className={cn(
                                    "p-2 rounded-full",
                                    speakingMessageId === arenaMessage.gpt.id
                                      ? "bg-red-100 text-red-600 hover:bg-red-200 dark:bg-red-900 dark:text-red-300 dark:hover:bg-red-800"
                                      : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-600 dark:text-gray-300 dark:hover:bg-gray-500"
                                  )}
                                >
                                  {speakingMessageId === arenaMessage.gpt.id ? (
                                    <VolumeX className="h-4 w-4" />
                                  ) : (
                                    <Volume2 className="h-4 w-4" />
                                  )}
                                </button>
                                {/* Translation buttons for AI responses (except variable questions) */}
                                {!arenaMessage.isVariableQuestion && (
                                  <div className="flex flex-wrap gap-1 md:gap-2 ml-1 md:ml-2">
                                    <button
                                      onClick={() => translateMessage('gpt', arenaMessage.gpt.id, 'en')}
                                      className={cn(
                                        "text-xs px-2 py-1 rounded-full border",
                                        arenaMessage.gpt.currentLanguage === 'en' 
                                          ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                          : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                                      )}
                                    >
                                      English
                                    </button>
                                    <button
                                      onClick={() => translateMessage('gpt', arenaMessage.gpt.id, 'hi')}
                                      className={cn(
                                        "text-xs px-2 py-1 rounded-full border",
                                        arenaMessage.gpt.currentLanguage === 'hi' 
                                          ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                          : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                                      )}
                                    >
                                      हिंदी
                                    </button>
                                    <button
                                      onClick={() => translateMessage('gpt', arenaMessage.gpt.id, 'mr')}
                                      className={cn(
                                        "text-xs px-2 py-1 rounded-full border",
                                        arenaMessage.gpt.currentLanguage === 'mr' 
                                          ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                          : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                                      )}
                                    >
                                      मराठी
                                    </button>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="flex items-start gap-2 justify-end">
                            <div className="rounded-2xl px-3 py-2 md:px-4 md:py-3 bg-blue-600 text-white max-w-[80%] md:max-w-[70%]">
                              <p className="text-sm">{arenaMessage.gpt.content}</p>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                    {isLoading && (
                      <div className="rounded-2xl px-3 py-2 md:px-4 md:py-3 bg-green-50 dark:bg-green-900/30 text-gray-800 dark:text-gray-200">
                        <p className="text-sm">Thinking...</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Claude Column */}
              <div className="flex flex-col h-full">
                <div className="bg-gradient-to-r from-purple-500 to-purple-600 text-white p-3 md:p-4 rounded-t-xl">
                  <h2 className="text-lg md:text-xl font-bold">Claude</h2>
                </div>
                <div className="bg-white dark:bg-gray-700 rounded-b-xl p-3 md:p-4 flex-1 min-h-[300px] md:min-h-[400px] flex flex-col">
                  <div className="flex-1 overflow-y-auto mb-4 space-y-4">
                    {messages.map((arenaMessage) => (
                      <div key={arenaMessage.claude.id} className="mb-3">
                        {arenaMessage.claude.role === "assistant" ? (
                          <div className="rounded-2xl px-3 py-2 md:px-4 md:py-3 bg-purple-50 dark:bg-purple-900/30 text-gray-800 dark:text-gray-200">
                            <div className="text-sm whitespace-pre-wrap">
                              {typingContent[arenaMessage.claude.id] ? (
                                <TypingSimulation content={getDisplayContent(arenaMessage.claude)} />
                              ) : (
                                cleanResponseContent(getDisplayContent(arenaMessage.claude))
                              )}
                              {/* Speaker button for AI responses */}
                              <div className="mt-2 flex items-center gap-2 flex-wrap">
                                <button
                                  onClick={() => handleTextToSpeech(arenaMessage.claude.id, getDisplayContent(arenaMessage.claude))}
                                  className={cn(
                                    "p-2 rounded-full",
                                    speakingMessageId === arenaMessage.claude.id
                                      ? "bg-red-100 text-red-600 hover:bg-red-200 dark:bg-red-900 dark:text-red-300 dark:hover:bg-red-800"
                                      : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-600 dark:text-gray-300 dark:hover:bg-gray-500"
                                  )}
                                >
                                  {speakingMessageId === arenaMessage.claude.id ? (
                                    <VolumeX className="h-4 w-4" />
                                  ) : (
                                    <Volume2 className="h-4 w-4" />
                                  )}
                                </button>
                                {/* Translation buttons for AI responses (except variable questions) */}
                                {!arenaMessage.isVariableQuestion && (
                                  <div className="flex flex-wrap gap-1 md:gap-2 ml-1 md:ml-2">
                                    <button
                                      onClick={() => translateMessage('claude', arenaMessage.claude.id, 'en')}
                                      className={cn(
                                        "text-xs px-2 py-1 rounded-full border",
                                        arenaMessage.claude.currentLanguage === 'en' 
                                          ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                          : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                                      )}
                                    >
                                      English
                                    </button>
                                    <button
                                      onClick={() => translateMessage('claude', arenaMessage.claude.id, 'hi')}
                                      className={cn(
                                        "text-xs px-2 py-1 rounded-full border",
                                        arenaMessage.claude.currentLanguage === 'hi' 
                                          ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                          : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                                      )}
                                    >
                                      हिंदी
                                    </button>
                                    <button
                                      onClick={() => translateMessage('claude', arenaMessage.claude.id, 'mr')}
                                      className={cn(
                                        "text-xs px-2 py-1 rounded-full border",
                                        arenaMessage.claude.currentLanguage === 'mr' 
                                          ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                          : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                                      )}
                                    >
                                      मराठी
                                    </button>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="flex items-start gap-2 justify-end">
                            <div className="rounded-2xl px-3 py-2 md:px-4 md:py-3 bg-blue-600 text-white max-w-[80%] md:max-w-[70%]">
                              <p className="text-sm">{arenaMessage.claude.content}</p>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                    {isLoading && (
                      <div className="rounded-2xl px-3 py-2 md:px-4 md:py-3 bg-purple-50 dark:bg-purple-900/30 text-gray-800 dark:text-gray-200">
                        <p className="text-sm">Thinking...</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
          <AIInput onSubmit={handleSubmit} placeholder="Ask your AI Doctors..." />
        </div>
      </div>
    </div>
  );
}

const Logo = ({ open }: { open: boolean }) => {
  return (
    <div className="font-normal flex space-x-2 items-center text-lg text-gray-800 dark:text-white py-2 relative z-20">
      <div className="h-8 w-8 bg-gradient-to-b from-blue-500 to-blue-700 rounded-lg flex-shrink-0" />
      <motion.span
        initial={{ opacity: 0 }}
        animate={{ opacity: open ? 1 : 0 }}
        className="font-semibold text-gray-800 dark:text-white whitespace-pre"
      >
        AI Doctor Arena
      </motion.span>
    </div>
  );
};