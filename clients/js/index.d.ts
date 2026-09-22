export type DistanceMetric =
  | "euclidean"
  | "cosine"
  | "dot_product"
  | "manhattan"
  | "hamming";

export const Distance: {
  Euclidean: "euclidean";
  Cosine: "cosine";
  DotProduct: "dot_product";
  Manhattan: "manhattan";
  Hamming: "hamming";
};

export interface Health {
  status: string;
  vector_count: number;
}

export interface AddResponse {
  success: boolean;
  inserted: number;
  indices: number[];
}

export interface SearchResult {
  id: number;
  distance: number;
  data: number[];
}

export interface GigaVectorOptions {
  apiKey?: string;
  fetch?: typeof fetch;
}

export class GigaVectorError extends Error {
  status: number;
  code?: string;
}

export class GigaVector {
  constructor(baseURL: string, opts?: GigaVectorOptions);
  health(): Promise<Health>;
  addVector(data: number[], metadata?: Record<string, string>): Promise<AddResponse>;
  search(query: number[], k: number, distance?: DistanceMetric): Promise<SearchResult[]>;
  stats(): Promise<Record<string, unknown>>;
}

export default GigaVector;
