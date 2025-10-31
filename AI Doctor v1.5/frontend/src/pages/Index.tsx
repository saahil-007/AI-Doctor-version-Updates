import { SplineScene } from "@/components/ui/splite";
import { Card } from "@/components/ui/card";
import { Spotlight } from "@/components/ui/spotlight";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Sparkles } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="w-full min-h-screen bg-black/[0.96] relative overflow-hidden">
      <Spotlight
        className="-top-40 left-0 md:left-60 md:-top-20"
        fill="white"
      />
      
      <div className="flex flex-col lg:flex-row h-screen">
        {/* Left content */}
        <div className="flex-1 p-8 md:p-16 relative z-10 flex flex-col justify-center">
          <div className="max-w-2xl">
            <div className="flex items-center gap-2 mb-6">
              <Sparkles className="w-6 h-6 text-neutral-400" />
              <span className="text-neutral-400 text-sm font-medium">AI-Powered Healthcare</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 mb-6">
              AI Doctor
            </h1>
            
            <p className="text-lg md:text-xl text-neutral-300 mb-8 max-w-lg leading-relaxed">
              Experience the future of healthcare with our intelligent AI assistant. 
              Get instant medical insights, personalized health advice, and 24/7 support 
              powered by advanced artificial intelligence.
            </p>

            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                size="lg"
                onClick={() => navigate("/chat")}
                className="bg-gradient-to-b from-neutral-50 to-neutral-400 text-black hover:from-neutral-100 hover:to-neutral-500 font-semibold"
              >
                Get Started
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              
              <Button
                size="lg"
                variant="outline"
                className="border-neutral-700 text-neutral-300 hover:bg-neutral-900 hover:text-neutral-100"
              >
                Learn More
              </Button>
            </div>
          </div>
        </div>

        {/* Right content - 3D Scene */}
        <div className="flex-1 relative min-h-[400px] lg:min-h-0">
          <SplineScene 
            scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
            className="w-full h-full"
          />
        </div>
      </div>
    </div>
  );
};

export default Index;
