
export enum PredictionMode {
  ENVIRONMENT_RISK = 'ENVIRONMENT_RISK',
  CANCER_PROBABILITY = 'CANCER_PROBABILITY',
  SURVIVAL_PREDICTION = 'SURVIVAL_PREDICTION'
}

export type UserRole = 'user' | 'admin';

export interface User {
  username: string;
  role: UserRole;
}

export interface RegistrationTrend {
  month: string;
  count: number;
}

export interface AdminStats {
  totalUsers: number;
  monthlyRegistrations: number;
  totalPredictions: number;
  recentPredictions: Array<{
    id: string;
    type: string;
    result: string;
    time: string;
  }>;
  registrationTrend: RegistrationTrend[];
}

export interface RiskLevelResponse {
  low_risk: number;
  medium_risk: number;
  high_risk: number;
  level: '低风险' | '中风险' | '高风险';
  using_mock?: boolean;
}

export interface ProbabilityResponse {
  probability: number;
  using_mock?: boolean;
}

export interface SurvivalResponse {
  estimated_months: number;
  using_mock?: boolean;
}

export interface FeatureConfig {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  unit?: string;
}

export interface FeatureInfo {
  id: string;
  title: string;
  description: string;
  icon: string;
}
