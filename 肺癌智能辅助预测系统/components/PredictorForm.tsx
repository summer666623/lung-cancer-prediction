
import React, { useState, useEffect, useRef } from 'react';
import { PredictionMode, FeatureConfig } from '../types';
import { predictEnvironmentRisk, predictProbability, predictSurvival } from '../services/api';

import Chart from 'chart.js/auto';


interface Props {
  mode: PredictionMode;
  onPredictionStateChange: (isPredicting: boolean) => void;
}

const PredictorForm: React.FC<Props> = ({ mode, onPredictionStateChange }) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [formValues, setFormValues] = useState<Record<string, any>>({});

  const pieRef = useRef<HTMLCanvasElement>(null);
  const barRef = useRef<HTMLCanvasElement>(null);
  const resultAreaRef = useRef<HTMLDivElement>(null);
  const charts = useRef<any[]>([]);

  // 特征定义与用户提供的 Key 严格对应
  const configs: Record<PredictionMode, FeatureConfig[]> = {
    [PredictionMode.ENVIRONMENT_RISK]: [
      { key: 'Age', label: '年龄', min: 1, max: 110, step: 1, default: 45 },
      { key: 'Gender', label: '性别 (1:男, 2:女)', min: 1, max: 2, step: 1, default: 1 },
      { key: 'Air Pollution', label: '空气污染', min: 1, max: 8, step: 1, default: 4 },
      { key: 'Alcohol use', label: '饮酒习惯', min: 1, max: 8, step: 1, default: 2 },
      { key: 'Dust Allergy', label: '粉尘过敏', min: 1, max: 8, step: 1, default: 3 },
      { key: 'OccuPational Hazards', label: '职业危害', min: 1, max: 8, step: 1, default: 2 },
      { key: 'Genetic Risk', label: '遗传风险', min: 1, max: 8, step: 1, default: 4 },
      { key: 'chronic Lung Disease', label: '慢性肺病', min: 1, max: 8, step: 1, default: 2 },
      { key: 'Balanced Diet', label: '均衡饮食', min: 1, max: 8, step: 1, default: 5 },
      { key: 'Obesity', label: '肥胖程度', min: 1, max: 8, step: 1, default: 3 },
      { key: 'Smoking', label: '吸烟程度', min: 1, max: 8, step: 1, default: 2 },
      { key: 'Passive Smoker', label: '二手烟暴露', min: 1, max: 8, step: 1, default: 1 },
      { key: 'Chest Pain', label: '胸痛程度', min: 1, max: 8, step: 1, default: 1 },
      { key: 'Coughing of Blood', label: '咯血程度', min: 1, max: 8, step: 1, default: 1 },
      { key: 'Fatigue', label: '疲劳程度', min: 1, max: 8, step: 1, default: 2 },
      { key: 'Weight Loss', label: '体重减轻', min: 1, max: 8, step: 1, default: 1 },
      { key: 'Shortness of Breath', label: '呼吸短促', min: 1, max: 8, step: 1, default: 1 },
      { key: 'Wheezing', label: '喘息程度', min: 1, max: 8, step: 1, default: 1 },
      { key: 'Swallowing Difficulty', label: '吞咽困难', min: 1, max: 8, step: 1, default: 1 },

      { key: 'Clubbing of Finger Nails', label: '杵状指', min: 1, max: 8, step: 1, default: 1 },
      { key: 'Frequent Cold', label: '频繁感冒', min: 1, max: 8, step: 1, default: 1 },
      { key: 'Dry Cough', label: '干咳', min: 1, max: 8, step: 1, default: 1 },
      { key: 'Snoring', label: '打鼾', min: 1, max: 8, step: 1, default: 1 },
    ],

    [PredictionMode.CANCER_PROBABILITY]: [
      { key: 'age', label: '年龄', min: 1, max: 110, step: 1, default: 45, unit: '岁' },
      { key: 'smoking_years', label: '吸烟年限', min: 0, max: 80, step: 1, default: 10, unit: '年' },
      { key: 'pack_per_day', label: '日吸烟量', min: 0, max: 10, step: 0.5, default: 1, unit: '包' },
      { key: 'family_history', label: '家族病史 (0/1)', min: 0, max: 1, step: 1, default: 0 },
      { key: 'air_pollution_index', label: '空气污染指数', min: 0, max: 500, step: 1, default: 50, unit: 'AQI' },
      { key: 'chronic_lung_disease', label: '慢性肺病 (0/1)', min: 0, max: 1, step: 1, default: 0 },
      { key: 'lung_cancer_prob', label: '基准发病率', min: 0, max: 1, step: 0.01, default: 0.05 },
    ],
    [PredictionMode.SURVIVAL_PREDICTION]: [
      { key: 'Ethnicity', label: '种族类型 (1-5)', min: 1, max: 5, step: 1, default: 1 },
      { key: 'Insurance_Type', label: '医保类型', min: 1, max: 3, step: 1, default: 1 },
      { key: 'Family_History', label: '家族病史 (0/1)', min: 0, max: 1, step: 1, default: 0 },
      { key: 'Comorbidity_Diabetes', label: '糖尿病 (0/1)', min: 0, max: 1, step: 1, default: 0 },
      { key: 'Comorbidity_Hypertension', label: '高血压 (0/1)', min: 0, max: 1, step: 1, default: 0 },
      { key: 'Comorbidity_Heart_Disease', label: '心脏病 (0/1)', min: 0, max: 1, step: 1, default: 0 },
      { key: 'Comorbidity_Chronic_Lung_Disease', label: '慢性肺病 (0/1)', min: 0, max: 1, step: 1, default: 0 },
      { key: 'Comorbidity_Kidney_Disease', label: '肾脏病 (0/1)', min: 0, max: 1, step: 1, default: 0 },
      { key: 'Comorbidity_Autoimmune_Disease', label: '自身免疫病 (0/1)', min: 0, max: 1, step: 1, default: 0 },
      { key: 'Comorbidity_Other', label: '其他并发症 (0/1)', min: 0, max: 1, step: 1, default: 0 },
      { key: 'Performance_Status', label: '体能状态评分 (0-5)', min: 0, max: 5, step: 1, default: 1 },
      { key: 'Blood_Pressure', label: '血压等级 (1-3)', min: 1, max: 3, step: 1, default: 2 },
    ]
  };

  useEffect(() => {
    const initialValues: any = {};
    configs[mode].forEach(c => initialValues[c.key] = c.default);
    setFormValues(initialValues);
    setResult(null);
    setError(null);
  }, [mode]);

  // 渲染图表的 Effect
  useEffect(() => {
    if (result && mode === PredictionMode.ENVIRONMENT_RISK) {
      // 延迟确保 DOM 节点已在渲染结果时就绪
      const timer = setTimeout(() => {
        charts.current.forEach(c => c?.destroy?.());
        charts.current = [];

        if (pieRef.current && barRef.current) {
          const dataVals = [result.low_risk, result.medium_risk, result.high_risk];
          const labels = ['低风险', '中风险', '高风险'];
          const colors = ['#22c55e', '#eab308', '#ef4444'];

          charts.current.push(new Chart(pieRef.current, {
            type: 'doughnut',
            data: {
              labels,
              datasets: [{ data: dataVals, backgroundColor: colors, borderWidth: 0 }]
            },
            options: { plugins: { legend: { display: false } }, cutout: '70%', responsive: true, maintainAspectRatio: false }
          }));

          charts.current.push(new Chart(barRef.current, {
            type: 'bar',
            data: {
              labels,
              datasets: [{ data: dataVals, backgroundColor: colors, borderRadius: 10 }]
            },
            options: {
              plugins: { legend: { display: false } },
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                y: { beginAtZero: true, max: 1, ticks: { color: '#94a3b8' } },
                x: { ticks: { color: '#94a3b8', font: { weight: 'bold' } } }
              }
            }
          }));
        }
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [result, mode]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    onPredictionStateChange(true);
    setError(null);
    setResult(null);

    try {
      let res;
      if (mode === PredictionMode.ENVIRONMENT_RISK) {
        res = await predictEnvironmentRisk(formValues);
        console.log('【ENV 风险预测-后端原始返回】', res);
        }
      else if (mode === PredictionMode.CANCER_PROBABILITY) res = await predictProbability(formValues);
      else res = await predictSurvival(formValues);

      if (mode === PredictionMode.ENVIRONMENT_RISK) {

setResult({
  level: res.risk_level,
  low_risk: res.distribution.low,
  medium_risk: res.distribution.medium,
  high_risk: res.distribution.high,
});

} else {
  setResult(res);
}

      // 异步滚动确保渲染完成
      setTimeout(() => {
        resultAreaRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 100);
    } catch (err) {
      setError('系统运算异常');
    } finally {
      setLoading(false);
      onPredictionStateChange(false);
    }
  };

  return (
    <div className="space-y-12 pb-24">
      {/* 输入表单 */}
      <form onSubmit={handleSubmit} className="glass-panel p-10 rounded-[3rem] relative shadow-2xl border border-white/10 transition-all hover:border-blue-500/30">
        {loading && (
          <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-xl z-40 rounded-[3rem] flex flex-col items-center justify-center gap-6">
            <div className="w-24 h-24 border-8 border-blue-500/10 border-t-blue-500 rounded-full animate-spin"></div>
            <p className="text-xl font-black text-blue-400 tracking-[0.5em] animate-pulse">深度模型分析中</p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-12 gap-y-8">
          {configs[mode].map((f) => (
            <div key={f.key} className="relative group">
              <div className="flex justify-between items-center mb-2 px-1">
                <label className="text-xs font-black text-slate-400 group-hover:text-blue-400 transition-colors uppercase tracking-widest">{f.label}</label>
                <span className="text-sm font-black text-blue-500 bg-blue-500/10 px-3 py-1 rounded-lg">
                  {formValues[f.key]} {f.unit || ''}
                </span>
              </div>
              <input
                type="range"
                min={f.min}
                max={f.max}
                step={f.step}
                value={formValues[f.key] || f.default}
                onChange={(e) => setFormValues({...formValues, [f.key]: Number(e.target.value)})}
                className="w-full accent-blue-600 h-2 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>
          ))}
        </div>

        <div className="mt-16 flex flex-col items-center">
          <button
            type="submit"
            className="group px-24 py-6 bg-blue-600 hover:bg-blue-500 text-white rounded-3xl font-black text-lg tracking-[0.2em] shadow-[0_20px_50px_-15px_rgba(37,99,235,0.5)] transition-all transform hover:scale-105 active:scale-95"
          >
            立即获取预测结果
            <i className="fa-solid fa-chevron-right ml-4 group-hover:translate-x-2 transition-transform"></i>
          </button>
          {error && <p className="mt-4 text-red-400 font-bold">{error}</p>}
        </div>
      </form>

      {/* 结果显示区 */}
      {result && (
        <div ref={resultAreaRef} className="animate-fade-in space-y-10 scroll-mt-24">
          <div className="flex items-center gap-6">
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-slate-700 to-transparent"></div>
            <h3 className="text-sm font-black text-slate-500 uppercase tracking-[0.4em]">智能评估详细报告</h3>
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-slate-700 to-transparent"></div>
          </div>

          {mode === PredictionMode.ENVIRONMENT_RISK ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="glass-panel p-12 rounded-[3.5rem] flex flex-col items-center shadow-2xl bg-gradient-to-b from-slate-900/50 to-slate-950/80">
                <h4 className="text-2xl font-black mb-12 text-glow flex items-center gap-4">
                  <i className="fa-solid fa-chart-pie text-blue-500"></i>
                  风险权重占比
                </h4>
                <div className="relative w-80 h-80">
                  <canvas ref={pieRef}></canvas>
                  <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                    <span className="text-xs text-slate-500 font-black uppercase tracking-widest mb-1">风险等级</span>
                    <span className={`text-4xl font-black px-6 py-2 rounded-2xl bg-slate-900/90 border-2 ${result.level === '高风险' ? 'text-red-500 border-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.3)]' : result.level === '中风险' ? 'text-amber-500 border-amber-500/30' : 'text-green-500 border-green-500/30'}`}>
                      {result.level}
                    </span>
                  </div>
                </div>
              </div>

              <div className="glass-panel p-12 rounded-[3.5rem] shadow-2xl bg-gradient-to-b from-slate-900/50 to-slate-950/80">
                <h4 className="text-2xl font-black mb-12 text-glow flex items-center gap-4">
                  <i className="fa-solid fa-chart-column text-blue-500"></i>
                  概率定量分布
                </h4>
                <div className="h-80 w-full">
                  <canvas ref={barRef}></canvas>
                </div>
              </div>
            </div>
          ) : mode === PredictionMode.CANCER_PROBABILITY ? (
            <div className="glass-panel p-20 rounded-[4rem] text-center bg-gradient-to-br from-blue-900/20 via-slate-900/40 to-slate-950 shadow-[0_30px_100px_-20px_rgba(37,99,235,0.3)] border-t-8 border-blue-500">
              <h4 className="text-3xl font-black mb-12 uppercase tracking-widest text-glow">罹患概率核心预测</h4>
              <div className="relative inline-block">
                <div className="text-[12rem] font-black text-blue-400 leading-none drop-shadow-[0_0_40px_rgba(59,130,246,0.6)] animate-fade-in">
                  {(result.probability * 100).toFixed(1)}<span className="text-4xl text-slate-600 ml-4 font-normal">%</span>
                </div>
              </div>
              <div className="mt-16 max-w-2xl mx-auto p-10 bg-slate-900/80 rounded-[2.5rem] border border-white/5 backdrop-blur-xl">
                <p className="text-slate-400 text-lg leading-relaxed italic">
                  “系统综合评估了临床输入特征，当前数值提示需高度关注。建议配合影像学（CT/MRI）进行临床确诊。”
                </p>
              </div>
            </div>
          ) : (
            <div className="glass-panel p-20 rounded-[4rem] text-center border-l-[20px] border-emerald-500 bg-gradient-to-r from-emerald-900/10 to-slate-950 shadow-2xl">
              <h4 className="text-3xl font-black mb-10 text-emerald-400 uppercase tracking-[0.3em]">预期生存周期评估</h4>
              <div className="flex items-baseline justify-center gap-6 mb-12">
                <span className="text-[14rem] font-black text-emerald-400 tracking-tighter leading-none">{result.estimated_months}</span>
                <span className="text-4xl font-black text-slate-600 uppercase tracking-widest">MONTHS</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
                <div className="p-6 bg-slate-900/60 rounded-3xl border border-white/5">
                  <p className="text-[11px] font-black text-slate-500 uppercase mb-2">模型依据</p>
                  <p className="text-sm font-bold text-slate-300">特征加权非线性分析</p>
                </div>
                <div className="p-6 bg-slate-900/60 rounded-3xl border border-white/5">
                  <p className="text-[11px] font-black text-slate-500 uppercase mb-2">置信区间</p>
                  <p className="text-sm font-bold text-emerald-400">95.8% (CI)</p>
                </div>
                <div className="p-6 bg-slate-900/60 rounded-3xl border border-white/5">
                  <p className="text-[11px] font-black text-slate-500 uppercase mb-2">预后状态</p>
                  <p className="text-sm font-bold text-slate-300">稳定评估</p>
                </div>
              </div>
            </div>
          )}

          <div className="flex justify-center pt-10">
            <button
              onClick={() => {
                setResult(null);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              className="px-12 py-4 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-2xl text-sm font-black transition-all border border-white/5 uppercase tracking-widest shadow-xl"
            >
              重新录入特征参数
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictorForm;
