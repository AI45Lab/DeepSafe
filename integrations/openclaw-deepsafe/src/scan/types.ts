declare function require(name: string): any;

const fs = require("fs");

import { Finding, FindingCategory, ModuleStatus } from "../report/schema";

export type ModuleResult = {
  name: FindingCategory;
  status: ModuleStatus;
  score: number;
  findings: Finding[];
  error?: string;
};

export function safeReadJson(filePath: string): any {
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw);
}
