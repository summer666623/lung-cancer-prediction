import axios from "axios";
import {
  RiskLevelResponse,
  ProbabilityResponse,
  SurvivalResponse,
  AdminStats,
  User,
  UserRole,
  RegistrationTrend
} from "../types";

/* =========================
   后端真实 API（Flask）
========================= */

const realApi = axios.create({
  baseURL: "http://127.0.0.1:5000/api",
  timeout: 10000
});

/* =========================
   本地 Mock / LocalStorage
========================= */

const getUsersFromDB = (): any[] => {
  const users = localStorage.getItem("app_users");
  return users
    ? JSON.parse(users)
    : [{ username: "hzq", password: "123", role: "admin", createdAt: new Date().toISOString() }];
};

const incrementPredictionCount = () => {
  const count = Number(localStorage.getItem("total_predictions") || "0");
  localStorage.setItem("total_predictions", (count + 1).toString());

  const recent = JSON.parse(localStorage.getItem("recent_history") || "[]");
  recent.unshift({
    id: Date.now().toString(),
    type: "肺癌风险预测",
    result: "已完成",
    time: "刚刚"
  });
  localStorage.setItem("recent_history", JSON.stringify(recent.slice(0, 10)));
};

/* =========================
   用户 / 管理端（继续 mock）
========================= */

export const loginUser = async (
  username: string,
  password: string,
  role: UserRole
): Promise<User | null> => {
  await new Promise(r => setTimeout(r, 100));
  const users = getUsersFromDB();
  const found = users.find(
    u => u.username === username && u.password === password && u.role === role
  );
  return found ? { username: found.username, role: found.role } : null;
};

export const registerUser = async (
  username: string,
  password: string,
  role: UserRole
): Promise<boolean> => {
  await new Promise(r => setTimeout(r, 100));
  const users = getUsersFromDB();
  if (users.find(u => u.username === username)) return false;

  users.push({ username, password, role, createdAt: new Date().toISOString() });
  localStorage.setItem("app_users", JSON.stringify(users));
  return true;
};

export const getAdminStats = async (): Promise<AdminStats> => {
  const users = getUsersFromDB();
  const totalPredictions = Number(localStorage.getItem("total_predictions") || "0");
  const recentHistory = JSON.parse(localStorage.getItem("recent_history") || "[]");

  const months = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"];
  const trend: RegistrationTrend[] = months.map(m => ({ month: m, count: 0 }));

  users.forEach(u => {
    const m = new Date(u.createdAt).getMonth();
    trend[m].count += 1;
  });

  return {
    totalUsers: users.length,
    monthlyRegistrations: users.filter(u => {
      const d = new Date(u.createdAt);
      return d.getMonth() === new Date().getMonth();
    }).length,
    totalPredictions,
    recentPredictions: recentHistory,
    registrationTrend: trend
  };
};

/* =========================
   🔥 核心：真实模型预测
========================= */

export const predictEnvironmentRisk = async (
  data: any
): Promise<RiskLevelResponse> => {
  incrementPredictionCount();

  try {
    const res = await realApi.post("/predict", data);

    // ✅ 关键：把 distribution 原样返回
    return {
      risk_code: res.data.risk_code,
      risk_level: res.data.risk_level,
      distribution: res.data.distribution,
      using_mock: false
    };
  } catch (err) {
    console.error("❌ 后端预测失败", err);
    throw err;
  }
};


/* =========================
   下面两个：暂时保留 mock
   （你后端还没做）
========================= */

export const predictProbability = async (
  data: any
): Promise<ProbabilityResponse> => {
  incrementPredictionCount();

  try {
    const res = await realApi.post("/predict/incidence", data);

    return {
      probability: res.data.incidence_probability,
      using_mock: false
    };
  } catch (err) {
    console.error("❌ 患病率预测失败", err);
    throw err;
  }
};

/* =========================
export const predictSurvival = async (
  data: any
): Promise<SurvivalResponse> => {
  incrementPredictionCount();
  await new Promise(r => setTimeout(r, 200));

  return { estimated_months: 36 + Math.floor(Math.random() * 24), using_mock: true };
};
========================= */
export const predictSurvival = async (
  data: any
): Promise<SurvivalResponse> => {
  incrementPredictionCount();

  try {
    // 调用 Flask 后端真正的生存时间接口
    const res = await realApi.post("/predict/survival", data);

    // 返回 estimated_survival_months
    return {
      estimated_months: res.data.estimated_survival_months,
      using_mock: false
    };
  } catch (err) {
    console.error("❌ 生存时间预测失败，使用 mock 数据", err);

    // fallback：仍然可以返回 mock 数据
    return {
      estimated_months: 36 + Math.floor(Math.random() * 24),
      using_mock: true
    };
  }
};
