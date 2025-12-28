
import React, { useEffect, useState, useRef } from 'react';
import { AdminStats, UserRole } from '../types';
import { getAdminStats, registerUser } from '../services/api';

declare const Chart: any;

const AdminDashboard: React.FC = () => {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [showRegModal, setShowRegModal] = useState(false);
  const [regForm, setRegForm] = useState({ username: '', password: '' });
  const [regLoading, setRegLoading] = useState(false);
  const [regMessage, setRegMessage] = useState('');
  
  const lineChartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<any>(null);

  const fetchStats = () => {
    getAdminStats().then(setStats);
  };

  useEffect(() => {
    fetchStats();
  }, []);

  useEffect(() => {
    if (stats && lineChartRef.current && typeof Chart !== 'undefined') {
      if (chartInstance.current) chartInstance.current.destroy();
      
      const ctx = lineChartRef.current.getContext('2d');
      chartInstance.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels: stats.registrationTrend.map(t => t.month),
          datasets: [{
            label: '用户注册增长趋势',
            data: stats.registrationTrend.map(t => t.count),
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            fill: true,
            tension: 0.4,
            borderWidth: 3,
            pointBackgroundColor: '#6366f1',
            pointRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
            x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
          }
        }
      });
    }
  }, [stats]);

  const handleAdminReg = async (e: React.FormEvent) => {
    e.preventDefault();
    setRegLoading(true);
    const success = await registerUser(regForm.username, regForm.password, 'admin');
    if (success) {
      setRegMessage('管理员注册成功！');
      setRegForm({ username: '', password: '' });
      fetchStats();
      setTimeout(() => setShowRegModal(false), 1500);
    } else {
      setRegMessage('用户名已存在，注册失败。');
    }
    setRegLoading(false);
  };

  if (!stats) return <div className="loading-shimmer h-96 rounded-[3rem]"></div>;

  return (
    <div className="space-y-10 animate-fade-in pb-20">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
        <div>
          <h2 className="text-4xl font-black text-glow">监控中心 Dashboard</h2>
          <p className="text-slate-500 font-bold mt-1 uppercase tracking-widest text-xs">Real-time system health & metrics</p>
        </div>
        <button 
          onClick={() => setShowRegModal(true)}
          className="px-8 py-4 bg-indigo-600 hover:bg-indigo-500 rounded-2xl font-black text-sm transition-all flex items-center gap-3 shadow-[0_15px_30px_-10px_rgba(79,70,229,0.5)]"
        >
          <i className="fa-solid fa-user-shield"></i>
          创建特权管理员
        </button>
      </div>

      {showRegModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-950/90 backdrop-blur-md">
          <div className="glass-panel w-full max-w-md p-10 rounded-[3rem] border border-indigo-500/30">
            <div className="flex justify-between items-center mb-8">
              <h3 className="text-2xl font-black">新增管理员</h3>
              <button onClick={() => setShowRegModal(false)} className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center text-slate-400 hover:text-white"><i className="fa-solid fa-xmark"></i></button>
            </div>
            <form onSubmit={handleAdminReg} className="space-y-6">
              <div className="space-y-1">
                <label className="text-[10px] font-black text-slate-500 uppercase ml-1">管理员账号</label>
                <input 
                  type="text" placeholder="Admin ID" required
                  className="w-full bg-slate-900/50 border border-slate-700 rounded-2xl p-4 text-sm outline-none focus:border-indigo-500 transition-all"
                  value={regForm.username} onChange={e => setRegForm({...regForm, username: e.target.value})}
                />
              </div>
              <div className="space-y-1">
                <label className="text-[10px] font-black text-slate-500 uppercase ml-1">访问密码</label>
                <input 
                  type="password" placeholder="Root Password" required
                  className="w-full bg-slate-900/50 border border-slate-700 rounded-2xl p-4 text-sm outline-none focus:border-indigo-500 transition-all"
                  value={regForm.password} onChange={e => setRegForm({...regForm, password: e.target.value})}
                />
              </div>
              <button 
                type="submit" disabled={regLoading}
                className="w-full py-4 bg-indigo-600 hover:bg-indigo-500 rounded-2xl font-black text-sm active:scale-95 transition-all shadow-lg"
              >
                {regLoading ? '正在核准权限...' : '确 认 授 权'}
              </button>
              {regMessage && <p className={`text-center text-xs font-bold ${regMessage.includes('成功') ? 'text-green-400' : 'text-red-400'}`}>{regMessage}</p>}
            </form>
          </div>
        </div>
      )}

      {/* 核心统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="glass-panel p-8 rounded-[2.5rem] border-b-8 border-blue-500 group hover:-translate-y-2 transition-transform shadow-xl">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-500 text-[10px] font-black uppercase tracking-widest">活跃用户总数</p>
              <h3 className="text-6xl font-black mt-2 text-white group-hover:text-blue-400 transition-colors">{stats.totalUsers}</h3>
            </div>
            <div className="w-16 h-16 bg-blue-500/10 rounded-[1.5rem] flex items-center justify-center text-blue-500 text-2xl">
              <i className="fa-solid fa-users"></i>
            </div>
          </div>
          <p className="text-blue-400 text-xs font-bold mt-6 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
            实时数据库同步正常
          </p>
        </div>

        <div className="glass-panel p-8 rounded-[2.5rem] border-b-8 border-indigo-500 group hover:-translate-y-2 transition-transform shadow-xl">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-500 text-[10px] font-black uppercase tracking-widest">本月新增注册</p>
              <h3 className="text-6xl font-black mt-2 text-white group-hover:text-indigo-400 transition-colors">{stats.monthlyRegistrations}</h3>
            </div>
            <div className="w-16 h-16 bg-indigo-500/10 rounded-[1.5rem] flex items-center justify-center text-indigo-500 text-2xl">
              <i className="fa-solid fa-user-plus"></i>
            </div>
          </div>
          <p className="text-indigo-400 text-xs font-bold mt-6">
            增长率: {(stats.monthlyRegistrations / (stats.totalUsers || 1) * 100).toFixed(1)}%
          </p>
        </div>

        <div className="glass-panel p-8 rounded-[2.5rem] border-b-8 border-emerald-500 group hover:-translate-y-2 transition-transform shadow-xl">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-500 text-[10px] font-black uppercase tracking-widest">累计推理请求</p>
              <h3 className="text-6xl font-black mt-2 text-white group-hover:text-emerald-400 transition-colors">{stats.totalPredictions}</h3>
            </div>
            <div className="w-16 h-16 bg-emerald-500/10 rounded-[1.5rem] flex items-center justify-center text-emerald-500 text-2xl">
              <i className="fa-solid fa-bolt-lightning"></i>
            </div>
          </div>
          <p className="text-emerald-400 text-xs font-bold mt-6">
            每次预测真实累加 (Session + Store)
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* 注册趋势图 */}
        <div className="lg:col-span-8 glass-panel p-10 rounded-[3rem] shadow-2xl">
          <div className="flex justify-between items-center mb-10">
            <h4 className="text-xl font-black flex items-center gap-3">
              <i className="fa-solid fa-chart-line text-indigo-500"></i>
              年度注册用户增长曲线
            </h4>
            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Update every 5s</span>
          </div>
          <div className="h-[400px] w-full">
            <canvas ref={lineChartRef}></canvas>
          </div>
        </div>

        {/* 最近日志 */}
        <div className="lg:col-span-4 glass-panel p-10 rounded-[3rem] shadow-2xl">
          <h4 className="text-xl font-black mb-10 flex items-center gap-3">
            <i className="fa-solid fa-list-ul text-blue-500"></i>
            核心运行日志
          </h4>
          <div className="space-y-6 max-h-[400px] overflow-y-auto custom-scrollbar pr-2">
            {stats.recentPredictions.map((p) => (
              <div key={p.id} className="flex items-center justify-between p-5 bg-slate-900/50 rounded-[1.5rem] border border-white/5 hover:border-blue-500/20 transition-all">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-slate-800 rounded-2xl flex items-center justify-center shadow-inner">
                    <i className="fa-solid fa-code-commit text-slate-500 text-sm"></i>
                  </div>
                  <div>
                    <p className="font-black text-sm text-slate-300">{p.type}</p>
                    <p className="text-[10px] text-slate-500 font-bold uppercase">{p.time}</p>
                  </div>
                </div>
                <span className="px-4 py-1.5 bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded-xl text-[10px] font-black">
                  {p.result}
                </span>
              </div>
            ))}
            {stats.recentPredictions.length === 0 && (
              <p className="text-center text-slate-600 py-10 font-bold italic">暂无实时日志产生</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
