
import { FeatureInfo } from './types';

export const COMPLEX_FEATURES: FeatureInfo[] = [
  {
    id: 'genetic_risk',
    title: '遗传风险 (Genetic Risk)',
    description: '指家族中是否存在肺癌病史或其他相关癌症遗传基因。具有一级亲属肺癌史的人群，其患病风险显著高于普通人群。',
    icon: 'fa-dna'
  },
  {
    id: 'occupational_hazards',
    title: '职业危害 (OccuPational Hazards)',
    description: '长期暴露于石棉、氡、砷、铬、镍及其化合物等致癌物质的环境中。这些物质进入呼吸道后可能引发细胞变异。',
    icon: 'fa-mask-ventilator'
  },
  {
    id: 'air_pollution',
    title: '空气污染 (Air Pollution)',
    description: '包括室外PM2.5工业废气和室内厨房油烟、煤烟等。长期吸入细颗粒物是导致非吸烟者患肺癌的重要因素。',
    icon: 'fa-smog'
  },
  {
    id: 'performance_status',
    title: '体能状态评分 (Performance Status)',
    description: '通常采用ECOG或Karnofsky评分，用于评估患者日常生活能力及对治疗的耐受程度，是预测生存期的关键指标。',
    icon: 'fa-person-walking'
  },
  {
    id: 'clubbing',
    title: '杵状指 (Clubbing of Finger Nails)',
    description: '手指末端增生、肥厚，呈杵状膨大。常与慢性缺氧及肺部肿瘤分泌的生长因子有关，是呼吸系统疾病的典型体征。',
    icon: 'fa-hand'
  },
  {
    id: 'passive_smoker',
    title: '被动吸烟 (Passive Smoker)',
    description: '长期处于二手烟环境。二手烟中含有多种与原烟类似的致癌物，对非吸烟者的肺部健康构成巨大威胁。',
    icon: 'fa-ban-smoking'
  }
];

export const LUNG_CANCER_INTRO = "肺癌是全球范围内发病率和死亡率最高的恶性肿瘤之一。其发生发展与吸烟、环境污染、职业暴露、遗传及生活习惯等多种因素密切相关。早期筛查和科学评估对于提高患者生存率至关重要。本系统旨在利用先进的机器学习模型，为医疗决策提供数据支持。";
