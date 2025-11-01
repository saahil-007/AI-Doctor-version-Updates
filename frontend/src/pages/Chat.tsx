import { useState } from "react";
import { Sidebar, SidebarBody, SidebarLink } from "@/components/ui/sidebar";
import { LayoutDashboard, MessageSquare, Settings, LogOut, Bot, Languages } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AIInput } from "@/components/ui/ai-input";

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
      content: "Hello, I am Dr.Vaani, your Personal AI Doctor, How can i help you today!",
      currentLanguage: 'en'
    }
]);
  const [open, setOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

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
      // For Vercel deployment, use relative path
      // For local development, you might need to change this to 'http://localhost:5000/chat'
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: value,
          language: 'en',
          translate_to: 'en',
          source_language: 'en'
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Get the response as text
      const aiResponseText = await response.text();
      
      // Add AI response to chat
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: aiResponseText,
        currentLanguage: 'en'
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
      const response = await fetch('/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: message.content,
          target_language: targetLanguage,
          source_language: 'en' // Assuming AI responses are in English by default
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
                      {/* Translation buttons for AI responses */}
                      <div className="mt-3 flex flex-wrap gap-2">
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
                  ) : (
                    <p className="text-sm">{message.content}</p>
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