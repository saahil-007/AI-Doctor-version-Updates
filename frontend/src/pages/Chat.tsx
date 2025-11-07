import { useState, useRef, useCallback, useEffect } from "react";
import { Sidebar, SidebarBody, SidebarLink } from "@/components/ui/sidebar";
import { LayoutDashboard, MessageSquare, Settings, LogOut, Bot, Languages, Volume2, VolumeX, Sword } from "lucide-react";
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

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Hello, I am Dr. Vaani, your Personal AI Doctor. I'm here to provide you with caring medical assistance. How can I help you today?",
      currentLanguage: 'en'
    }
]);
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

  // Function to pre-fetch audio for common medical phrases
  const prefetchCommonPhrases = useCallback(async () => {
    const commonPhrases = [
      "Hello! I'm Dr. Vaani, your caring AI health assistant."
    ];
    
    const baseUrl = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';
    const ttsEndpoint = window.location.hostname === 'localhost' ? `${baseUrl}/text-to-speech` : '/api/text-to-speech';
    
    for (const phrase of commonPhrases) {
      try {
        // Skip if already cached
        if (audioCache.current.has(phrase)) continue;
        
        // Pre-fetch the audio
        const response = await fetch(ttsEndpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            text: phrase,
            messageId: `prefetch-${Date.now()}`
          }),
        });
        
        if (response.ok) {
          const audioBlob = await response.blob();
          audioCache.current.set(phrase, audioBlob);
          console.log(`Prefetched audio for: ${phrase.substring(0, 30)}...`);
        }
      } catch (error) {
        console.error("Error prefetching audio:", error);
      }
    }
  }, []);

  // Pre-fetch common phrases on component mount
  useEffect(() => {
    prefetchCommonPhrases();
  }, [prefetchCommonPhrases]);

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
        <Sword className="text-gray-600 dark:text-gray-300 h-5 w-5 flex-shrink-0" />
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

  const handleSubmit = async (value: string) => {
    if (!value.trim() || isLoading) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: value,
    };
    
    // Add user message to chat
    setMessages(prev => [...prev, newMessage]);
    setIsLoading(true);

    try {
      // For Vercel deployment, use /api/chat endpoint
      // For local development, you might need to change this to 'http://localhost:5000/chat'
      const baseUrl = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';
      const endpoint = window.location.hostname === 'localhost' ? `${baseUrl}/chat` : '/api/chat';
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/plain',
        },
        body: JSON.stringify({
          message: value,
          language: 'en',
          translate_to: 'en',
          source_language: 'en',
          user_id: 'chat_user'  // Add user_id for context tracking
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Get the response as text
      const aiResponseText = await response.text();
      
      // Detect language of the response
      const detectedLanguage = detectLanguage(aiResponseText);
      
      // Add AI response to chat with detected language
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: aiResponseText,
        currentLanguage: detectedLanguage
      };
      
      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I'm having trouble responding right now. Please try again later.",
        currentLanguage: 'en'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to translate a message
  const translateMessage = async (messageId: string, targetLanguage: 'en' | 'hi' | 'mr') => {
    const messageIndex = messages.findIndex(msg => msg.id === messageId);
    if (messageIndex === -1) return;

    const message = messages[messageIndex];
    if (!message || message.role !== "assistant") return;

    // Check if already translated
    if (message.translatedContent?.[targetLanguage]) {
      // Update current language
      const updatedMessages = [...messages];
      updatedMessages[messageIndex] = {
        ...message,
        currentLanguage: targetLanguage
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
      updatedMessages[messageIndex] = {
        ...message,
        translatedContent: {
          ...message.translatedContent,
          [targetLanguage]: translatedText
        },
        currentLanguage: targetLanguage
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
    <div className="flex flex-col md:flex-row bg-white dark:bg-gray-900 w-full h-screen overflow-hidden">
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
                href: "/chat",
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
        <div className="flex-1 overflow-y-auto p-4 md:p-8">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className={cn(
                  "flex gap-3",
                  message.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                {message.role === "assistant" && (
                  <div className="h-10 w-10 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center flex-shrink-0">
                    <Bot className="h-5 w-5 text-blue-600 dark:text-blue-300" />
                  </div>
                )}
                <div
                  className={cn(
                    "rounded-2xl px-5 py-4 max-w-[85%] shadow-sm",
                    message.role === "user"
                      ? "bg-blue-600 text-white rounded-tr-none"
                      : "bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-tl-none"
                  )}
                >
                  {message.role === "assistant" ? (
                    <div className="text-sm whitespace-pre-wrap">
                      {cleanResponseContent(getDisplayContent(message))}
                      {/* Speaker button for AI responses */}
                      <div className="mt-2 flex items-center gap-2">
                        <button
                          onClick={() => handleTextToSpeech(message.id, getDisplayContent(message))}
                          className={cn(
                            "p-2 rounded-full",
                            speakingMessageId === message.id
                              ? "bg-red-100 text-red-600 hover:bg-red-200 dark:bg-red-900 dark:text-red-300 dark:hover:bg-red-800"
                              : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-600 dark:text-gray-300 dark:hover:bg-gray-500"
                          )}
                        >
                          {speakingMessageId === message.id ? (
                            <VolumeX className="h-4 w-4" />
                          ) : (
                            <Volume2 className="h-4 w-4" />
                          )}
                        </button>
                        {/* Translation buttons for AI responses */}
                        <div className="flex flex-wrap gap-2 ml-2">
                          <button
                            onClick={() => translateMessage(message.id, 'en')}
                            className={cn(
                              "text-xs px-2 py-1 rounded-full border",
                              message.currentLanguage === 'en' 
                                ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                            )}
                          >
                            English
                          </button>
                          <button
                            onClick={() => translateMessage(message.id, 'hi')}
                            className={cn(
                              "text-xs px-2 py-1 rounded-full border",
                              message.currentLanguage === 'hi' 
                                ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                            )}
                          >
                            हिंदी
                          </button>
                          <button
                            onClick={() => translateMessage(message.id, 'mr')}
                            className={cn(
                              "text-xs px-2 py-1 rounded-full border",
                              message.currentLanguage === 'mr' 
                                ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-300" 
                                : "bg-gray-100 border-gray-300 text-gray-700 dark:bg-gray-600 dark:border-gray-500 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-500"
                            )}
                          >
                            मराठी
                          </button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-start gap-2">
                      <p className="text-sm">{message.content}</p>
                      {/* Speaker button for user messages */}
                      <button
                        onClick={() => handleTextToSpeech(message.id, message.content)}
                        className={cn(
                          "p-2 rounded-full mt-1",
                          speakingMessageId === message.id
                            ? "bg-red-100 text-red-600 hover:bg-red-200 dark:bg-red-900 dark:text-red-300 dark:hover:bg-red-800"
                            : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-600 dark:text-gray-300 dark:hover:bg-gray-500"
                        )}
                      >
                        {speakingMessageId === message.id ? (
                          <VolumeX className="h-4 w-4" />
                        ) : (
                          <Volume2 className="h-4 w-4" />
                        )}
                      </button>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-3 justify-start"
              >
                <div className="h-10 w-10 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center flex-shrink-0">
                  <Bot className="h-5 w-5 text-blue-600 dark:text-blue-300" />
                </div>
                <div className="rounded-2xl px-5 py-4 max-w-[85%] bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-tl-none shadow-sm">
                  <p className="text-sm">Thinking...</p>
                </div>
              </motion.div>
            )}
          </div>
        </div>
        
        <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
          <AIInput onSubmit={handleSubmit} placeholder="Ask your AI Doctor..." />
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
        AI Doctor
      </motion.span>
    </div>
  );
};