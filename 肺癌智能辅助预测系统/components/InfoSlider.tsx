
import React, { useState, useRef } from 'react';
import { COMPLEX_FEATURES } from '../constants';

const InfoSlider: React.FC = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  const scrollToIndex = (index: number) => {
    if (scrollRef.current) {
      const scrollAmount = scrollRef.current.offsetWidth * index;
      scrollRef.current.scrollTo({
        left: scrollAmount,
        behavior: 'smooth'
      });
      setCurrentIndex(index);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-100 mb-8">
      <div className="p-6 border-b border-slate-100 flex justify-between items-center">
        <h3 className="text-xl font-bold text-slate-800">
          <i className="fa-solid fa-circle-info mr-2 text-blue-600"></i>
          关键医学特征详细解析
        </h3>
        <div className="flex gap-2">
          <button 
            onClick={() => scrollToIndex(Math.max(0, currentIndex - 1))}
            disabled={currentIndex === 0}
            className={`w-8 h-8 rounded-full flex items-center justify-center border ${currentIndex === 0 ? 'text-slate-300 border-slate-200' : 'text-blue-600 border-blue-200 hover:bg-blue-50'}`}
          >
            <i className="fa-solid fa-chevron-left"></i>
          </button>
          <button 
            onClick={() => scrollToIndex(Math.min(COMPLEX_FEATURES.length - 1, currentIndex + 1))}
            disabled={currentIndex === COMPLEX_FEATURES.length - 1}
            className={`w-8 h-8 rounded-full flex items-center justify-center border ${currentIndex === COMPLEX_FEATURES.length - 1 ? 'text-slate-300 border-slate-200' : 'text-blue-600 border-blue-200 hover:bg-blue-50'}`}
          >
            <i className="fa-solid fa-chevron-right"></i>
          </button>
        </div>
      </div>

      <div 
        ref={scrollRef}
        className="flex overflow-x-hidden snap-x snap-mandatory scroll-smooth custom-scrollbar"
      >
        {COMPLEX_FEATURES.map((feature) => (
          <div key={feature.id} className="min-w-full snap-center p-8 flex flex-col md:flex-row items-center gap-6">
            <div className="w-20 h-20 bg-blue-50 rounded-2xl flex items-center justify-center text-blue-600 text-3xl shrink-0 shadow-inner">
              <i className={`fa-solid ${feature.icon}`}></i>
            </div>
            <div>
              <h4 className="text-lg font-bold text-slate-900 mb-2">{feature.title}</h4>
              <p className="text-slate-600 leading-relaxed text-sm md:text-base">
                {feature.description}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div className="flex justify-center pb-4 gap-1.5">
        {COMPLEX_FEATURES.map((_, idx) => (
          <div 
            key={idx}
            className={`w-2 h-2 rounded-full transition-all duration-300 ${currentIndex === idx ? 'w-6 bg-blue-600' : 'bg-slate-200'}`}
          ></div>
        ))}
      </div>
    </div>
  );
};

export default InfoSlider;
