"use client";

import { CornerRightUp, Mic, MicOff } from "lucide-react";
import { useState, useRef } from "react";
import { cn } from "@/lib/utils";
import { Textarea } from "@/components/ui/textarea";
import { useAutoResizeTextarea } from "@/hooks/use-auto-resize-textarea";
import { SpeechRecognition } from "@/lib/types";

interface AIInputProps {
  id?: string
  placeholder?: string
  minHeight?: number
  maxHeight?: number
  onSubmit?: (value: string) => void
  className?: string
}

export function AIInput({
  id = "ai-input",
  placeholder = "Type your message...",
  minHeight = 52,
  maxHeight = 200,
  onSubmit,
  className
}: AIInputProps) {
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight,
    maxHeight,
  });
  const [inputValue, setInputValue] = useState("");
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const handleReset = () => {
    if (!inputValue.trim()) return;
    onSubmit?.(inputValue);
    setInputValue("");
    adjustHeight(true);
  };

  const startListening = () => {
    if (isListening) return;

    // Check if SpeechRecognition is available
    const SpeechRecognitionConstructor = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (!SpeechRecognitionConstructor) {
      alert("Speech recognition is not supported in your browser. Please try Chrome or Edge.");
      return;
    }

    // Initialize speech recognition
    recognitionRef.current = new SpeechRecognitionConstructor() as SpeechRecognition;
    recognitionRef.current.continuous = false;
    recognitionRef.current.interimResults = true;
    recognitionRef.current.lang = 'en-US';

    recognitionRef.current.onstart = () => {
      setIsListening(true);
    };

    recognitionRef.current.onresult = (event: any) => {
      const transcript = Array.from(event.results)
        .map((result: any) => result[0])
        .map((result) => result.transcript)
        .join('');
      
      setInputValue(transcript);
      adjustHeight();
    };

    recognitionRef.current.onerror = (event: any) => {
      console.error('Speech recognition error', event.error);
      setIsListening(false);
    };

    recognitionRef.current.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current.start();
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  return (
    <div className={cn("w-full py-4", className)}>
      <div className="relative max-w-4xl w-full mx-auto">
        <Textarea
          id={id}
          placeholder={placeholder}
          className={cn(
            "w-full bg-white dark:bg-gray-800 rounded-3xl pl-6 pr-16",
            "placeholder:text-gray-400 dark:placeholder:text-gray-400",
            "border border-gray-300 dark:border-gray-600",
            "text-gray-900 dark:text-white text-wrap",
            "overflow-y-auto resize-none",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:border-blue-500",
            "transition-[height] duration-100 ease-out",
            "leading-[1.2] py-[16px] shadow-sm",
            "min-h-[52px]",
            `max-h-[${maxHeight}px]`,
            "[&::-webkit-resizer]:hidden"
          )}
          ref={textareaRef}
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
            adjustHeight();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleReset();
            }
          }}
        />

        <div
          className={cn(
            "absolute top-1/2 -translate-y-1/2 rounded-xl py-2 px-2 transition-all duration-200 cursor-pointer",
            "hover:bg-gray-200 dark:hover:bg-gray-600",
            inputValue ? "right-14" : "right-3"
          )}
          onClick={toggleListening}
        >
          {isListening ? (
            <MicOff className="w-5 h-5 text-red-500" />
          ) : (
            <Mic className="w-5 h-5 text-gray-600 dark:text-gray-300" />
          )}
        </div>
        <button
          onClick={handleReset}
          type="button"
          className={cn(
            "absolute top-1/2 -translate-y-1/2 right-3",
            "rounded-xl bg-blue-600 hover:bg-blue-700 py-2 px-2",
            "transition-all duration-200",
            "shadow-md hover:shadow-lg",
            inputValue 
              ? "opacity-100 scale-100" 
              : "opacity-0 scale-95 pointer-events-none"
          )}
        >
          <CornerRightUp className="w-5 h-5 text-white" />
        </button>
      </div>
    </div>
  );
}