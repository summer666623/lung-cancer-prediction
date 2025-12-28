
import React, { useState, useEffect } from 'react';
import { PredictionMode, User, UserRole } from './types';
import InfoSlider from './components/InfoSlider';
import PredictorForm from './components/PredictorForm';
import AdminDashboard from './components/AdminDashboard';
import { loginUser, registerUser } from './services/api';

const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [currentMode, setCurrentMode] = useState<PredictionMode>(PredictionMode.ENVIRONMENT_RISK);
  const [loginForm, setLoginForm] = useState({ username: '', password: '' });
  const [isRegistering, setIsRegistering] = useState(false);
  const [authRole, setAuthRole] = useState<UserRole>('user');
  const [loading, setLoading] = useState(false);
  const [authError, setAuthError] = useState('');
  const [isPredicting, setIsPredicting] = useState(false);

  // Clear form helper
  const clearForm = () => {
    setLoginForm({ username: '', password: '' });
    setAuthError('');
  };

  const handleRoleSwitch = (role: UserRole) => {
    if (authRole !== role) {
      setAuthRole(role);
      clearForm();
    }
  };

  const handleModeSwitch = () => {
    setIsRegistering(!isRegistering);
    setAuthRole('user');
    clearForm();
  };

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setAuthError('');

    if (isRegistering) {
      if (authRole === 'admin') {
        setAuthError('由于安全限制，管理员账号只能由已有管理员在控制台内创建。');
        setLoading(false);
        return;
      }
      const success = await registerUser(loginForm.username, loginForm.password, 'user');
      if (success) {
        setAuthError('注册成功，请登录！');
        setIsRegistering(false);
        clearForm();
      } else {
        setAuthError('用户名已存在，注册失败。');
      }
    } else {
      const userData = await loginUser(loginForm.username, loginForm.password, authRole);
      if (userData) {
        setUser(userData);
      } else {
        setAuthError('账号或密码错误，或该账号未在对应角色下注册。');
      }
    }
    setLoading(false);
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden bg-gradient-to-br from-[#0f172a] via-[#1e293b] to-[#0f172a]">
        <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] bg-blue-500/10 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] bg-indigo-500/10 rounded-full blur-[120px]"></div>
        
        <div className="glass-panel w-full max-w-md p-10 rounded-[2.5rem] shadow-2xl relative z-10 border border-white/5">
          <div className="flex justify-center mb-8">
            <div className="w-20 h-20 bg-blue-600 rounded-[2rem] flex items-center justify-center text-white text-3xl shadow-[0_0_30px_rgba(37,99,235,0.4)] animate-pulse">
              <i className="fa-solid fa-lungs"></i>
            </div>
          </div>
          
          <h1 className="text-3xl font-black text-center mb-2 tracking-tight">AI 肺癌预测系统</h1>
          
          {/* Role Switcher */}
          {!isRegistering && (
            <div className="flex bg-slate-900/50 p-1 rounded-xl mb-6 border border-slate-800">
              <button 
                type="button"
                onClick={() => handleRoleSwitch('user')}
                className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all ${authRole === 'user' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}
              >
                普通用户
              </button>
              <button 
                type="button"
                onClick={() => handleRoleSwitch('admin')}
                className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all ${authRole === 'admin' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}
              >
                管理员入口
              </button>
            </div>
          )}
          
          <form onSubmit={handleAuth} className="space-y-6">
            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-500 uppercase ml-1">用户名</label>
              <div className="relative group">
                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-blue-500 transition-colors">
                  <i className="fa-solid fa-user"></i>
                </span>
                <input 
                  type="text" required
                  value={loginForm.username}
                  placeholder="请输入您的账号"
                  className="w-full bg-slate-900/50 border border-slate-700/50 rounded-2xl py-4 pl-12 pr-4 text-sm focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 outline-none transition-all placeholder:text-slate-700 font-medium"
                  onChange={(e) => setLoginForm({...loginForm, username: e.target.value})}
                />
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-500 uppercase ml-1">密码</label>
              <div className="relative group">
                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-blue-500 transition-colors">
                  <i className="fa-solid fa-lock"></i>
                </span>
                <input 
                  type="password" required
                  value={loginForm.password}
                  placeholder="请输入您的密码"
                  className="w-full bg-slate-900/50 border border-slate-700/50 rounded-2xl py-4 pl-12 pr-4 text-sm focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 outline-none transition-all placeholder:text-slate-700 font-medium"
                  onChange={(e) => setLoginForm({...loginForm, password: e.target.value})}
                />
              </div>
            </div>

            {authError && <p className={`text-center text-xs font-bold ${authError.includes('成功') ? 'text-green-400' : 'text-red-400'}`}>{authError}</p>}

            <button 
              type="submit" disabled={loading}
              className={`w-full py-4 rounded-2xl text-white font-black text-sm tracking-widest transition-all active:scale-95 flex items-center justify-center gap-3 ${authRole === 'admin' && !isRegistering ? 'bg-indigo-600 shadow-[0_0_20px_rgba(79,70,229,0.3)]' : 'bg-blue-600 shadow-[0_0_20px_rgba(37,99,235,0.3)]'}`}
            >
              {loading ? <i className="fa-solid fa-circle-notch fa-spin"></i> : (isRegistering ? '注 册 账 号' : '登 录')}
            </button>
          </form>
          
          <div className="mt-8 pt-8 border-t border-slate-800/50 flex flex-col items-center gap-4">
            <button 
              type="button"
              onClick={handleModeSwitch}
              className="text-slate-500 text-xs font-bold hover:text-blue-400 transition-colors"
            >
              {isRegistering ? '已有普通用户账号？返回登录' : '没有账号？普通用户请先注册'}
            </button>
            <p className="text-[10px] text-slate-600 font-medium tracking-tighter">管理员 hzq 初始密码 123</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200">
      <nav className={`glass-panel sticky top-0 z-50 border-b border-white/5 transition-opacity ${isPredicting ? 'opacity-70 pointer-events-none' : 'opacity-100'}`}>
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(37,99,235,0.3)]">
              <i className="fa-solid fa-brain"></i>
            </div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-black tracking-tight">肺癌 AI 决策中心</h1>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-slate-900/50 p-1.5 rounded-2xl border border-white/5">
            {user.role === 'admin' ? (
              <span className="text-xs font-black text-amber-500 px-3 flex items-center gap-2">
                <i className="fa-solid fa-shield-halved"></i>
                控制台模式: {user.username}
              </span>
            ) : (
              (Object.keys(PredictionMode) as Array<keyof typeof PredictionMode>).map((key) => (
                <button
                  key={key}
                  disabled={isPredicting}
                  onClick={() => setCurrentMode(PredictionMode[key])}
                  className={`px-5 py-2 rounded-xl text-[10px] font-black tracking-widest transition-all ${currentMode === PredictionMode[key] ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'} ${isPredicting ? 'cursor-not-allowed opacity-50' : ''}`}
                >
                  {key === 'ENVIRONMENT_RISK' ? '风险级' : key === 'CANCER_PROBABILITY' ? '患病率' : '存活预测'}
                </button>
              ))
            )}
          </div>

          <button 
            onClick={() => setUser(null)}
            className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center text-slate-400 hover:bg-red-500/20 hover:text-red-400 transition-all border border-white/5"
          >
            <i className="fa-solid fa-power-off"></i>
          </button>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {user.role === 'admin' ? (
          <AdminDashboard />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
            <div className="lg:col-span-4 space-y-8">
              <div className="glass-panel p-8 rounded-[2rem]">
                <h2 className="text-2xl font-black mb-6 flex items-center gap-3">
                  <span className="w-1.5 h-6 bg-blue-500 rounded-full"></span>
                  系统运行状态
                </h2>
                <div className={`p-4 rounded-2xl border transition-all ${isPredicting ? 'bg-amber-500/10 border-amber-500/30' : 'bg-green-500/10 border-green-500/30'}`}>
                  <p className="text-xs font-bold flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${isPredicting ? 'bg-amber-500 animate-pulse' : 'bg-green-500'}`}></span>
                    {isPredicting ? '核心算法正在计算...' : 'AI 模型已就绪'}
                  </p>
                  <p className="text-[10px] text-slate-500 mt-1 leading-relaxed">
                    {isPredicting ? '正在调用 Flask 后端接口进行深度非线性回归分析，请稍候。' : '当前已连接三个机器学习 pkl 核心镜像。点击下方按钮开始快速预测。'}
                  </p>
                </div>
              </div>
              <InfoSlider />
            </div>

            <div className="lg:col-span-8">
              <div className={`mb-10 transition-opacity ${isPredicting ? 'opacity-50' : 'opacity-100'}`}>
                <h3 className="text-4xl font-black text-white mb-2">
                  {currentMode === PredictionMode.ENVIRONMENT_RISK ? '环境风险等级分析' : 
                   currentMode === PredictionMode.CANCER_PROBABILITY ? '患病概率智能预测' : '存活时间智能预测'}
                </h3>
                <p className="text-slate-500 text-lg leading-relaxed">
                  {isPredicting ? '预测任务执行中，系统已临时锁定导航功能以保证计算完整性。' : '拖动滑块调整特征等级。等级越高，对应的环境暴露或生理风险越大。'}
                </p>
              </div>

              <PredictorForm 
                mode={currentMode} 
                onPredictionStateChange={(state) => setIsPredicting(state)} 
              />
            </div>
          </div>
        )}
      </main>
      
      <footer className="max-w-7xl mx-auto px-6 py-12 border-t border-white/5 text-center">
        <p className="text-[10px] text-slate-600 font-bold uppercase tracking-[0.4em]">
          © 2025 LUNG INTELLIGENCE MEDICAL CENTER | 深 度 学 习 生 物 医 学 研 究 室
        </p>
      </footer>
    </div>
  );
};

export default App;
