export interface ModelWeights {
  feature_cols: string[];
  scaler_mean: number[];
  scaler_scale: number[];
  coefficients: number[];
  intercept: number;
}

export interface ReplayEvent {
  t: string; // ISO timestamp
  u: string; // user id
  a: string; // artist name
  tr: string; // track name
}

export interface FeatureComparisonRow {
  feature: string;
  retained_avg: number;
  disengaging_avg: number;
}

export interface EngagementSummary {
  n_users: number;
  n_events: number;
  n_sessions: number;
  n_labeled_user_weeks: number;
  disengagement_rate: number;
  date_range: [string, string];
  feature_comparison: FeatureComparisonRow[];
}

export interface ModelMetrics {
  roc_auc: number;
  pr_auc: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
}

export interface ModelComparison {
  n_labeled_user_weeks: number;
  n_users: number;
  n_train_rows: number;
  n_test_rows: number;
  n_train_users: number;
  n_test_users: number;
  disengagement_rate: number;
  results: Record<string, ModelMetrics>;
}

export interface UserFeatureVector {
  sessions_count: number;
  tracks_count: number;
  unique_artists: number;
  repeat_rate: number;
  avg_session_minutes: number;
  rolling_sessions: number;
  rolling_tracks: number;
  rolling_unique_artists: number;
  rolling_repeat_rate: number;
  sessions_trend: number;
  weeks_since_active: number;
}

export interface UserRiskState extends UserFeatureVector {
  user_id: string;
  churn_probability: number;
  last_event_time: string;
}
