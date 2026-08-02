import weights from "../../public/data/model_weights.json";
import { ModelWeights, UserFeatureVector } from "./types";

const W = weights as ModelWeights;

/**
 * Exact TypeScript port of the trained Logistic Regression pipeline
 * (StandardScaler + LogisticRegression) from ml/src/export_web_replay.py.
 * export_web_replay.py asserts these exported weights reproduce sklearn's
 * predict_proba to within 1e-6 before writing them to disk -- this is not
 * an approximation of the model, it IS the model's decision function.
 */
export function scoreFeatureVector(features: UserFeatureVector): number {
  let logit = W.intercept;
  W.feature_cols.forEach((name, i) => {
    const raw = (features as unknown as Record<string, number>)[name];
    const scaled = (raw - W.scaler_mean[i]) / W.scaler_scale[i];
    logit += W.coefficients[i] * scaled;
  });
  return 1 / (1 + Math.exp(-logit));
}
