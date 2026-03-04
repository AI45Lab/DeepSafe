declare function require(name: string): any;
import { ScanReport, Finding, maxSeverity } from "./schema";

const MODULE_LABELS: Record<string, string> = {
  posture: "Deployment Posture",
  skill: "Skill / MCP",
  model: "Model Safety",
  memory: "Memory",
};

const MODULE_ICONS: Record<string, string> = {
  posture: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>`,
  skill: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>`,
  model: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a8 8 0 0 0-8 8c0 3.4 2.1 6.3 5 7.4V20h6v-2.6c2.9-1.1 5-4 5-7.4a8 8 0 0 0-8-8z"/><line x1="10" y1="22" x2="14" y2="22"/></svg>`,
  memory: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 12h.01M10 12h.01M14 12h.01M18 12h.01"/></svg>`,
};

const MODULE_DESCRIPTIONS: Record<string, string> = {
  posture: "Infrastructure & deployment configuration",
  skill: "Tool access & MCP integration security",
  model: "Model behavior & safety alignment",
  memory: "Data persistence & memory isolation",
};

const STATUS_LABELS: Record<string, string> = {
  ok: "Passed",
  warn: "Warning",
  error: "Error",
  skipped: "Skipped",
  not_implemented: "N/A",
};

const STATUS_DOT_COLORS: Record<string, string> = {
  ok: "#22c55e",
  warn: "#eab308",
  error: "#ef4444",
  skipped: "#64748b",
  not_implemented: "#475569",
};

const SEVERITY_COLORS: Record<string, string> = {
  CRITICAL: "#ef4444",
  HIGH: "#f97316",
  MEDIUM: "#eab308",
  LOW: "#22c55e",
};

const SEVERITY_BG: Record<string, string> = {
  CRITICAL: "rgba(239,68,68,0.12)",
  HIGH: "rgba(249,115,22,0.10)",
  MEDIUM: "rgba(234,179,8,0.08)",
  LOW: "rgba(34,197,94,0.06)",
};

const SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const;

function scoreColor(score: number): string {
  if (score >= 85) return "#22c55e";
  if (score >= 65) return "#eab308";
  if (score >= 40) return "#f97316";
  return "#ef4444";
}

function scoreColorDim(score: number): string {
  if (score >= 85) return "rgba(34,197,94,0.15)";
  if (score >= 65) return "rgba(234,179,8,0.15)";
  if (score >= 40) return "rgba(249,115,22,0.15)";
  return "rgba(239,68,68,0.15)";
}

function riskLabel(score: number): string {
  if (score >= 85) return "LOW RISK";
  if (score >= 65) return "MEDIUM RISK";
  if (score >= 40) return "HIGH RISK";
  return "CRITICAL RISK";
}

function escHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function durationStr(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`;
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return escHtml(iso);
  }
}

const DEEPSAFE_LOGO = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 241.25 49.18" class="ds-logo-svg"><defs><style>.ds-logo{fill:#fff}</style></defs><g><path class="ds-logo" d="M36.92,34.89c-.74.57-1.56.83-2.47.49-.48-.17-.79-.05-1.13.29-1.75,1.78-3.51,3.54-5.28,5.3-.27.27-.24.43.02.67.86.84,1.71,1.69,2.55,2.55.25.26.45.25.71.03,1.44-1.22,2.8-2.51,3.99-3.97,2.47-3.02,4.24-6.43,5.52-10.1,1.42-4.07,2.13-8.27,2.44-12.54,0-.11.04-.24-.1-.32-.18,0-.28.15-.39.26-1.64,1.63-3.29,3.27-4.93,4.91-.23.23-.44.47-.39.84.13,1.06-.79,2.12-1.88,2.08-.47-.02-.8.13-1.12.45-5.2,5.22-10.41,10.42-15.62,15.63-.2.2-.3.5-.56.64-.21-.09-.34-.26-.49-.4-1.3-1.3-2.59-2.6-3.88-3.9q-.42-.42,0-.85c1.45-1.45,2.9-2.91,4.36-4.35.3-.3.44-.62.44-1.05-.02-1.14,0-2.29,0-3.43,0-.25-.02-.5-.3-.57-.43-.1-.84-.58-1.34.41-.74,1.45-1.91,2.62-3.35,3.37-.63.33-.82.73-.73,1.25.23,1.42-1.11,2.54-2.48,2.09-.9-.3-1.45-1.23-1.29-2.19.16-.96.96-1.57,2.01-1.52.31.02.57-.02.8-.25,1.4-1.4,2.8-2.8,4.2-4.2.25-.25.33-.5.24-.87-.46-1.96,1.29-3.72,3.2-3.23,1.07.27,1.75.96,1.98,2.02.23,1.05-.06,1.99-.94,2.66-.33.25-.45.53-.45.93.02,1.33,0,2.67,0,4.01,0,.55-.17.98-.57,1.37-1.23,1.2-2.43,2.43-3.65,3.64-.26.26-.25.42.11.71.59.48,1.12,1.01,1.61,1.6.36.43.56.49.9.14,4.92-4.93,9.85-9.86,14.78-14.78.24-.24.33-.46.26-.82-.26-1.34.91-2.53,2.25-2.27.31.06.55.02.78-.21,2.1-2.11,4.21-4.22,6.32-6.32.22-.22.33-.47.31-.79-.06-1.27-.12-2.55-.16-3.82-.01-.35-.2-.49-.51-.55-.46-.08-.93-.14-1.38-.26-.47-.12-.8,0-1.14.34-1.37,1.39-2.75,2.77-4.13,4.15-2.26,2.26-4.52,4.52-6.8,6.77-.28.28-.39.52-.22.92.22.51.21,1.07.07,1.6-.4,1.47-1.86,2.27-3.34,1.86-1.34-.37-2.17-1.84-1.82-3.23.36-1.43,1.76-2.28,3.22-1.91.38.1.64.02.9-.25,3.45-3.46,6.91-6.91,10.36-10.37.37-.37.34-.46-.17-.63-1.28-.43-2.57-.83-3.85-1.28-.42-.15-.88-.04-1.19.27-3.58,3.62-7.19,7.21-10.78,10.82-.32.32-.67.47-1.15.44-1.36-.08-2.72-.09-4.08,0-.63.04-1.07-.13-1.49-.56-1.43-1.47-2.89-2.92-4.33-4.38-.23-.23-.46-.37-.81-.32-1.4.22-2.39-.72-2.25-2.11.03-.32.07-.64-.28-.88-.69-.48-1.33-1.05-1.77-1.77-.2-.32-.47-.42-.78-.35-1.61.36-3.23.74-4.84,1.09-.34.08-.52.27-.51.61,0,.72,0,1.44-.1,2.15-.11.8.21,1.36.78,1.92,2.99,2.94,5.95,5.92,8.91,8.89.45.45.42.42.92-.01.39-.34.53-.87.38-1.37-.12-.4-.1-.85,0-1.27.34-1.32,1.76-2.15,3.12-1.83,1.43.34,2.35,1.74,2.04,3.12-.35,1.58-1.81,2.47-3.37,2.03-.43-.12-.71-.04-1,.3-.27.32-.56.63-.87.92-.9.84-.76.98-1.29.45-3.05-3.05-6.1-6.1-9.15-9.15-.11-.11-.19-.27-.37-.26-.12.11-.11.25-.09.38.2,2.04.34,4.08.67,6.1.68,4.1,1.85,8.04,3.72,11.76,1.47,2.92,3.32,5.57,5.66,7.85,3.14,3.06,6.84,5.21,10.99,6.61.36.12.7.15,1.07.03,2.46-.79,4.78-1.87,6.93-3.29.15-.1.34-.18.4-.44-1.37-1.31-2.76-2.63-4.11-4.03.24-.21.45-.39.65-.58,2.09-2.08,4.17-4.17,6.27-6.24.34-.34.44-.65.31-1.13-.25-.94.03-1.77.72-2.45.72-.7,1.59-.96,2.55-.68,2.1.61,2.66,3.05,1.04,4.61M22.43,16.43c3.21-3.21,6.46-6.39,9.7-9.69-.28-.14-.43-.23-.59-.3-3.33-1.5-6.45-3.35-9.34-5.59-.51-.39-.52-.39-1.05.02-2.64,2.05-5.5,3.75-8.5,5.2-.58.28-.6.36-.13.82,2,2,4.01,4,6,6.01.32.32.5.25.84-.15.82-.94,1.71-1.81,2.65-2.62.55-.47.73-.84.5-1.52-.61-1.79.94-3.71,2.97-3.36,1.5.25,2.45,1.62,2.18,3.14-.26,1.43-1.66,2.39-3.12,2.09-.43-.09-.72.03-1.01.33-1.38,1.4-2.77,2.79-4.16,4.18q-.57.57-1.12.02c-1.2-1.21-2.4-2.41-3.61-3.61-1.3-1.3-2.6-2.59-3.89-3.89-.27-.27-.52-.36-.9-.2-.55.24-1.12.43-1.68.65-.45.18-.48.32-.15.66.38.4.79.78,1.17,1.19.2.21.48.33.76.32,1.38-.03,2.14.74,2.14,2.12,0,.42.14.72.62,1,1.44.85,2.62,2.06,3.49,3.48.36.59.8.94,1.39.87.75-.08,1.5-.11,2.24-.03,1.08.11,1.92-.21,2.6-1.13M36.22,33.28c.21-.7-.05-1.3-.66-1.5-.56-.19-1.22.15-1.41.73-.17.52.08,1.08.58,1.33.54.27,1.07.09,1.49-.56M18.93,26.1c.52.06.86-.21,1.11-.63.23-.38.2-.76-.06-1.11-.26-.36-.62-.52-1.07-.46-.49.07-.88.49-.92.98-.05.59.23.97.94,1.21M25.78,7.11c-.44-.34-.89-.4-1.3-.16-.4.24-.63.69-.55,1.11.09.49.44.85.92.95.37.08.85-.15,1.09-.51.27-.4.23-.83-.17-1.39M14.66,21.55c-.03-.13-.03-.26-.08-.38-.18-.41-.49-.65-.94-.68-.48-.04-.8.2-1.04.6-.23.39-.21.76.05,1.11.25.34.61.56,1.06.49.53-.09.85-.46.95-1.13M25.5,23.13c.04.08.07.16.12.23.37.53,1.06.64,1.57.26.48-.36.58-1.05.22-1.52-.39-.53-1.05-.6-1.57-.18-.38.3-.42.71-.35,1.22Z"/></g><g><g><path class="ds-logo" d="M59.79,36.55V6.45h10.06c3.47,0,6.35.62,8.64,1.87s3.99,3,5.1,5.25c1.1,2.25,1.66,4.9,1.66,7.93s-.55,5.68-1.66,7.93c-1.1,2.25-2.8,4-5.1,5.25s-5.17,1.87-8.64,1.87h-10.06ZM65.6,31.56h3.96c2.44,0,4.37-.4,5.81-1.2,1.43-.8,2.45-1.96,3.05-3.46s.9-3.3.9-5.4-.3-3.93-.9-5.44c-.6-1.51-1.62-2.66-3.05-3.46-1.43-.8-3.37-1.2-5.81-1.2h-3.96v20.17Z"/><path class="ds-logo" d="M97.89,37.07c-2.18,0-4.11-.47-5.78-1.42-1.68-.95-2.98-2.26-3.91-3.93-.93-1.68-1.4-3.6-1.4-5.78s.47-4.26,1.4-5.98c.93-1.72,2.23-3.08,3.89-4.08,1.66-1,3.6-1.5,5.81-1.5s4.03.47,5.63,1.42c1.6.95,2.85,2.21,3.74,3.81.89,1.59,1.33,3.42,1.33,5.48,0,.29,0,.61-.02.97-.02.36-.04.72-.06,1.1h-17.54v-3.57h11.78c-.09-1.35-.58-2.44-1.48-3.27-.9-.83-2.01-1.25-3.33-1.25-1,0-1.91.24-2.73.71-.82.47-1.47,1.16-1.96,2.06-.49.9-.73,2.04-.73,3.42v1.25c0,1.18.22,2.2.67,3.07.44.88,1.07,1.56,1.87,2.04.8.49,1.73.73,2.8.73s1.97-.22,2.64-.67c.67-.44,1.2-1.04,1.57-1.78h5.89c-.37,1.35-1.04,2.57-2,3.68-.96,1.1-2.12,1.96-3.48,2.58-1.36.62-2.89.92-4.58.92Z"/><path class="ds-logo" d="M121.2,37.07c-2.18,0-4.11-.47-5.78-1.42-1.68-.95-2.98-2.26-3.91-3.93-.93-1.68-1.4-3.6-1.4-5.78s.47-4.26,1.4-5.98c.93-1.72,2.23-3.08,3.89-4.08,1.66-1,3.6-1.5,5.81-1.5s4.03.47,5.63,1.42c1.6.95,2.85,2.21,3.74,3.81.89,1.59,1.33,3.42,1.33,5.48,0,.29,0,.61-.02.97-.02.36-.04.72-.06,1.1h-17.54v-3.57h11.78c-.09-1.35-.58-2.44-1.48-3.27-.9-.83-2.01-1.25-3.33-1.25-1,0-1.91.24-2.73.71-.82.47-1.47,1.16-1.96,2.06-.49.9-.73,2.04-.73,3.42v1.25c0,1.18.22,2.2.67,3.07.44.88,1.07,1.56,1.87,2.04.8.49,1.73.73,2.8.73s1.97-.22,2.64-.67c.67-.44,1.2-1.04,1.57-1.78h5.89c-.37,1.35-1.04,2.57-2,3.68-.96,1.1-2.12,1.96-3.48,2.58-1.36.62-2.89.92-4.58.92Z"/><path class="ds-logo" d="M134.1,46.01V14.88h5.16l.65,2.97c.43-.6.97-1.17,1.61-1.7.65-.53,1.4-.96,2.28-1.29.87-.33,1.88-.49,3.03-.49,2.06,0,3.87.49,5.42,1.48,1.55.99,2.77,2.34,3.68,4.06.9,1.72,1.35,3.67,1.35,5.85s-.46,4.12-1.38,5.83c-.92,1.71-2.15,3.05-3.7,4.02-1.55.97-3.33,1.46-5.33,1.46-1.58,0-2.95-.29-4.11-.88-1.16-.59-2.11-1.4-2.86-2.43v12.25h-5.81ZM145.53,31.99c1.12,0,2.11-.26,2.99-.77.87-.52,1.56-1.24,2.06-2.17.5-.93.75-2.01.75-3.25s-.25-2.37-.75-3.33c-.5-.96-1.19-1.71-2.06-2.24-.88-.53-1.87-.8-2.99-.8s-2.11.27-2.99.8c-.88.53-1.56,1.27-2.04,2.21-.49.95-.73,2.05-.73,3.31s.24,2.32.73,3.25c.49.93,1.17,1.66,2.04,2.19.87.53,1.87.8,2.99.8Z"/><path class="ds-logo" d="M170.17,37.07c-2.18,0-4.12-.37-5.83-1.12-1.71-.75-3.05-1.83-4.02-3.27-.97-1.43-1.49-3.17-1.55-5.2h6.15c.03.89.26,1.69.69,2.41.43.72,1.02,1.28,1.78,1.7.76.42,1.67.62,2.73.62.92,0,1.71-.15,2.39-.45.67-.3,1.2-.72,1.59-1.27.39-.54.58-1.19.58-1.93,0-.86-.22-1.58-.65-2.15-.43-.57-1.01-1.06-1.74-1.46-.73-.4-1.58-.77-2.54-1.1-.96-.33-1.96-.67-2.99-1.01-2.41-.8-4.23-1.85-5.46-3.14-1.23-1.29-1.85-3-1.85-5.12,0-1.81.43-3.35,1.29-4.64s2.06-2.28,3.61-2.97c1.55-.69,3.31-1.03,5.29-1.03s3.82.35,5.35,1.05c1.53.7,2.75,1.71,3.65,3.03.9,1.32,1.38,2.87,1.44,4.64h-6.23c-.03-.66-.22-1.28-.58-1.87-.36-.59-.85-1.06-1.48-1.42-.63-.36-1.38-.54-2.24-.54-.75-.03-1.43.09-2.04.34-.62.26-1.1.64-1.44,1.14-.34.5-.52,1.13-.52,1.87s.17,1.32.52,1.81c.34.49.83.91,1.46,1.27.63.36,1.38.69,2.24.99.86.3,1.79.61,2.79.92,1.52.52,2.9,1.13,4.15,1.83,1.25.7,2.24,1.61,2.99,2.73.75,1.12,1.12,2.6,1.12,4.43,0,1.58-.41,3.04-1.23,4.39-.82,1.35-2.01,2.44-3.59,3.27-1.58.83-3.53,1.25-5.85,1.25Z"/><path class="ds-logo" d="M190.38,37.07c-1.78,0-3.25-.29-4.43-.88-1.18-.59-2.06-1.38-2.64-2.37s-.88-2.09-.88-3.29c0-1.35.34-2.52,1.01-3.53.67-1,1.7-1.78,3.08-2.34,1.38-.56,3.12-.84,5.25-.84h5.29c0-1.09-.14-1.98-.41-2.67s-.72-1.2-1.33-1.55c-.62-.34-1.43-.52-2.43-.52-1.09,0-2.01.23-2.77.69-.76.46-1.23,1.19-1.4,2.19h-5.63c.14-1.58.65-2.92,1.53-4.04.87-1.12,2.04-1.99,3.48-2.62,1.45-.63,3.06-.95,4.84-.95,2.04,0,3.81.34,5.31,1.01,1.51.67,2.65,1.66,3.44,2.97.79,1.3,1.18,2.89,1.18,4.75v13.46h-4.86l-.64-3.31c-.32.57-.7,1.09-1.14,1.55-.45.46-.96.86-1.55,1.2-.59.34-1.23.61-1.91.79-.69.19-1.48.28-2.36.28ZM191.8,32.59c.72,0,1.37-.14,1.96-.41.59-.27,1.1-.65,1.53-1.14.43-.49.77-1.03,1.03-1.63.26-.6.43-1.26.52-1.98v-.04h-4.43c-.92,0-1.66.11-2.21.32-.56.22-.97.52-1.23.92-.26.4-.39.86-.39,1.38,0,.54.13,1.01.39,1.4.26.39.64.68,1.14.88.5.2,1.07.3,1.7.3Z"/><path class="ds-logo" d="M203.71,19.74v-4.86h13.46v4.86h-13.46ZM206.68,36.55V12.94c0-1.81.3-3.24.9-4.3.6-1.06,1.47-1.83,2.6-2.32,1.13-.49,2.46-.73,3.98-.73h2.67v4.94h-1.76c-.92,0-1.58.18-1.98.54-.4.36-.6.97-.6,1.83v23.65h-5.8Z"/><path class="ds-logo" d="M228.61,37.07c-2.18,0-4.11-.47-5.78-1.42-1.68-.95-2.98-2.26-3.91-3.93-.93-1.68-1.4-3.6-1.4-5.78s.46-4.26,1.4-5.98c.93-1.72,2.23-3.08,3.89-4.08,1.66-1,3.6-1.5,5.8-1.5s4.03.47,5.63,1.42,2.85,2.21,3.74,3.81c.89,1.59,1.33,3.42,1.33,5.48,0,.29,0,.61-.02.97-.01.36-.04.72-.06,1.1h-17.54v-3.57h11.78c-.09-1.35-.58-2.44-1.48-3.27-.9-.83-2.01-1.25-3.33-1.25-1,0-1.91.24-2.73.71-.82.47-1.47,1.16-1.96,2.06s-.73,2.04-.73,3.42v1.25c0,1.18.22,2.2.67,3.07.44.88,1.07,1.56,1.87,2.04.8.49,1.73.73,2.79.73s1.97-.22,2.64-.67c.67-.44,1.2-1.04,1.57-1.78h5.89c-.37,1.35-1.04,2.57-2,3.68-.96,1.1-2.12,1.96-3.48,2.58s-2.89.92-4.58.92Z"/></g></g></svg>`;

export function toHtml(report: ScanReport): string {
  const s = report.scores;
  const totalColor = scoreColor(s.total);
  const totalColorDim = scoreColorDim(s.total);
  const highest = maxSeverity(report.findings) ?? "NONE";
  const critCount = report.findings.filter((f) => f.severity === "CRITICAL").length;
  const highCount = report.findings.filter((f) => f.severity === "HIGH").length;
  const medCount = report.findings.filter((f) => f.severity === "MEDIUM").length;
  const lowCount = report.findings.filter((f) => f.severity === "LOW").length;
  const totalFindings = report.findings.length;

  const severityCounts: Record<string, number> = {
    CRITICAL: critCount,
    HIGH: highCount,
    MEDIUM: medCount,
    LOW: lowCount,
  };

  const totalGauge = (() => {
    const radius = 80;
    const circumference = 2 * Math.PI * radius;
    const pct = Math.max(0, Math.min(100, s.total)) / 100;
    const dashLen = circumference * pct;
    const gapLen = circumference - dashLen;
    return `<svg class="hero-gauge-svg" width="220" height="220" viewBox="0 0 200 200">
      <defs>
        <filter id="gauge-glow">
          <feGaussianBlur stdDeviation="4" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <linearGradient id="gauge-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="${totalColor}"/>
          <stop offset="100%" stop-color="${totalColor}" stop-opacity="0.6"/>
        </linearGradient>
      </defs>
      <circle cx="100" cy="100" r="${radius}" fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="14"/>
      <circle cx="100" cy="100" r="${radius}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="14"
        stroke-dasharray="2 8" stroke-linecap="round" transform="rotate(-90 100 100)"/>
      <circle class="gauge-arc" cx="100" cy="100" r="${radius}" fill="none"
        stroke="url(#gauge-grad)" stroke-width="14"
        stroke-dasharray="${dashLen} ${gapLen}"
        stroke-linecap="round" transform="rotate(-90 100 100)" filter="url(#gauge-glow)"
        style="--target-dash: ${dashLen} ${gapLen}; --circumference: ${circumference}"/>
      <text x="100" y="88" text-anchor="middle" fill="${totalColor}" font-size="52" font-weight="800" font-family="-apple-system,BlinkMacSystemFont,sans-serif">${s.total}</text>
      <text x="100" y="110" text-anchor="middle" fill="rgba(255,255,255,0.35)" font-size="14" font-family="-apple-system,BlinkMacSystemFont,sans-serif">out of 100</text>
      <text x="100" y="134" text-anchor="middle" fill="${totalColor}" font-size="13" font-weight="700" font-family="-apple-system,BlinkMacSystemFont,sans-serif" letter-spacing="1.5">${riskLabel(s.total)}</text>
    </svg>`;
  })();

  const moduleCards = (["posture", "skill", "model", "memory"] as const)
    .map((key, idx) => {
      const mod = report.modules.find((m) => m.name === key);
      const score = mod && mod.status !== "error" && mod.status !== "skipped" ? mod.score : 0;
      const status = mod ? mod.status : "skipped";
      const color = scoreColor(score);
      const colorDim = scoreColorDim(score);
      const radius = 36;
      const circumference = 2 * Math.PI * radius;
      const pct = Math.max(0, Math.min(100, score)) / 100;
      const dashLen = circumference * pct;
      const gapLen = circumference - dashLen;
      const modFindings = mod ? mod.findings : 0;
      const duration = mod ? durationStr(mod.durationMs) : "-";
      const dotColor = STATUS_DOT_COLORS[status] ?? "#475569";
      const statusLabel = STATUS_LABELS[status] ?? status;

      return `<div class="module-card fade-in" style="animation-delay:${idx * 0.1}s">
        <div class="module-card-inner">
          <div class="module-card-header">
            <span class="module-icon" style="color:${color}">${MODULE_ICONS[key] ?? ""}</span>
            <div class="module-card-title-group">
              <div class="module-card-title">${escHtml(MODULE_LABELS[key] ?? key)}</div>
              <div class="module-card-desc">${escHtml(MODULE_DESCRIPTIONS[key] ?? "")}</div>
            </div>
          </div>
          <div class="module-card-gauge">
            <svg width="90" height="90" viewBox="0 0 90 90">
              <circle cx="45" cy="45" r="${radius}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="6"/>
              <circle class="gauge-arc-sm" cx="45" cy="45" r="${radius}" fill="none" stroke="${color}" stroke-width="6"
                stroke-dasharray="${dashLen} ${gapLen}"
                stroke-linecap="round" transform="rotate(-90 45 45)"
                style="--target-dash: ${dashLen} ${gapLen}; --circumference: ${circumference}; animation-delay:${0.3 + idx * 0.15}s"/>
              <text x="45" y="42" text-anchor="middle" fill="${color}" font-size="20" font-weight="800" font-family="-apple-system,BlinkMacSystemFont,sans-serif">${score}</text>
              <text x="45" y="56" text-anchor="middle" fill="rgba(255,255,255,0.35)" font-size="9" font-family="-apple-system,BlinkMacSystemFont,sans-serif">/100</text>
            </svg>
          </div>
          <div class="module-card-meta">
            <span class="module-status-dot" style="background:${dotColor}"></span>
            <span class="module-status-label">${escHtml(statusLabel)}</span>
            <span class="module-meta-sep"></span>
            <span class="module-meta-item">${modFindings} finding${modFindings !== 1 ? "s" : ""}</span>
            <span class="module-meta-sep"></span>
            <span class="module-meta-item">${escHtml(duration)}</span>
          </div>
        </div>
      </div>`;
    })
    .join("\n");

  const severityBar = (() => {
    if (totalFindings === 0) {
      return `<div class="sev-bar-empty">No findings detected</div>`;
    }
    const segments = SEVERITY_ORDER.map((sev) => {
      const count = severityCounts[sev] ?? 0;
      if (count === 0) return "";
      const pct = (count / totalFindings) * 100;
      return `<div class="sev-bar-seg" style="width:${pct}%;background:${SEVERITY_COLORS[sev]}" title="${sev}: ${count}">
        ${pct > 12 ? `<span class="sev-bar-label">${count}</span>` : ""}
      </div>`;
    }).join("");
    const legend = SEVERITY_ORDER.map((sev) => {
      const count = severityCounts[sev] ?? 0;
      if (count === 0) return "";
      return `<div class="sev-legend-item">
        <span class="sev-legend-dot" style="background:${SEVERITY_COLORS[sev]}"></span>
        <span class="sev-legend-text">${sev} (${count})</span>
      </div>`;
    }).join("");
    return `<div class="sev-bar-track">${segments}</div>
    <div class="sev-legend">${legend}</div>`;
  })();

  const findingsByModule: Record<string, Finding[]> = {};
  for (const f of report.findings) {
    if (!findingsByModule[f.category]) findingsByModule[f.category] = [];
    findingsByModule[f.category].push(f);
  }

  let findingsHtml = "";
  for (const modName of ["posture", "skill", "model", "memory"] as const) {
    const modFindings = findingsByModule[modName];
    if (!modFindings || !modFindings.length) continue;

    const modSeverity = maxSeverity(modFindings);
    const modSevColor = modSeverity ? SEVERITY_COLORS[modSeverity] : "#64748b";

    findingsHtml += `<div class="findings-module-group fade-in">
      <div class="findings-module-header">
        <div class="findings-module-title-row">
          <span class="module-icon-sm" style="color:${modSevColor}">${MODULE_ICONS[modName] ?? ""}</span>
          <h3 class="findings-module-name">${escHtml(MODULE_LABELS[modName] ?? modName)}</h3>
          <span class="findings-module-count" style="background:${modSevColor}20;color:${modSevColor}">${modFindings.length} finding${modFindings.length !== 1 ? "s" : ""}</span>
        </div>
      </div>`;

    for (const f of modFindings) {
      const sevColor = SEVERITY_COLORS[f.severity] ?? "#64748b";
      const sevBg = SEVERITY_BG[f.severity] ?? "rgba(255,255,255,0.03)";
      const evidence = escHtml(f.evidence).replace(/\n/g, "<br>");
      const remediation = escHtml(f.remediation).replace(/\n/g, "<br>");

      findingsHtml += `<div class="finding-card" style="border-left-color:${sevColor}">
        <div class="finding-top">
          <span class="sev-badge" style="background:${sevColor}">${f.severity}</span>
          <span class="finding-title">${escHtml(f.title)}</span>
          <span class="finding-id">${escHtml(f.id)}</span>
        </div>
        <details class="finding-details">
          <summary class="finding-summary">View Evidence &amp; Remediation</summary>
          <div class="finding-body">
            <div class="finding-section-label">Evidence</div>
            <div class="finding-evidence">${evidence}</div>
            <div class="finding-section-label">Remediation</div>
            <div class="finding-remediation">${remediation}</div>
          </div>
        </details>
      </div>`;
    }

    findingsHtml += `</div>`;
  }

  if (!findingsHtml) {
    findingsHtml = `<div class="no-findings fade-in">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
      <div class="no-findings-text">No security issues found</div>
      <div class="no-findings-sub">All modules passed without findings</div>
    </div>`;
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepSafe Security Report</title>
<style>
:root {
  --bg-primary: #0a0e1a;
  --bg-card: rgba(255,255,255,0.03);
  --bg-card-hover: rgba(255,255,255,0.05);
  --border-subtle: rgba(255,255,255,0.06);
  --border-medium: rgba(255,255,255,0.1);
  --text-primary: #e2e8f0;
  --text-secondary: rgba(255,255,255,0.55);
  --text-muted: rgba(255,255,255,0.35);
  --accent-blue: #3b82f6;
  --accent-purple: #8b5cf6;
  --accent-gradient: linear-gradient(135deg, #3b82f6, #8b5cf6);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-xl: 20px;
}

*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}

body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  min-height: 100vh;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 0 24px 80px;
}

/* ── Hero Section ── */
.hero {
  position: relative;
  text-align: center;
  padding: 56px 24px 48px;
  margin-bottom: 40px;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(59,130,246,0.08) 0%, rgba(139,92,246,0.04) 60%, transparent 100%);
}

.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  mask-image: radial-gradient(ellipse 70% 60% at 50% 40%, black 30%, transparent 100%);
  -webkit-mask-image: radial-gradient(ellipse 70% 60% at 50% 40%, black 30%, transparent 100%);
  pointer-events: none;
}

.hero::after {
  content: "";
  position: absolute;
  top: 20%;
  left: 50%;
  transform: translateX(-50%);
  width: 600px;
  height: 600px;
  background: radial-gradient(circle, ${totalColorDim} 0%, transparent 70%);
  pointer-events: none;
  opacity: 0.5;
}

.hero-logo {
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: center;
  margin-bottom: 8px;
}

.ds-logo-svg { width: 200px; height: auto; }

.hero-subtitle {
  position: relative;
  z-index: 1;
  color: var(--text-muted);
  font-size: 13px;
  letter-spacing: 0.5px;
  margin-bottom: 32px;
}

.hero-subtitle span {
  display: inline-block;
  padding: 4px 12px;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border-subtle);
  border-radius: 20px;
  font-size: 12px;
}

.hero-gauge-wrap {
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: center;
  margin-bottom: 0;
}

.hero-gauge-svg {
  filter: drop-shadow(0 0 30px ${totalColorDim});
}

/* Gauge arc animation */
@keyframes gauge-fill {
  from { stroke-dasharray: 0 9999; }
}

.gauge-arc {
  animation: gauge-fill 1.4s cubic-bezier(0.23, 1, 0.32, 1) forwards;
}

.gauge-arc-sm {
  animation: gauge-fill 1.2s cubic-bezier(0.23, 1, 0.32, 1) forwards;
}

/* Fade in */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

.fade-in {
  opacity: 0;
  animation: fadeInUp 0.5s ease forwards;
}

/* ── Stats Row ── */
.stats-section {
  margin-bottom: 36px;
}

.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 12px;
}

.stat-pill {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 16px 12px;
  text-align: center;
  transition: background 0.2s, border-color 0.2s;
}

.stat-pill:hover {
  background: var(--bg-card-hover);
  border-color: var(--border-medium);
}

.stat-val {
  font-size: 24px;
  font-weight: 800;
  line-height: 1.2;
  letter-spacing: -0.5px;
}

.stat-lbl {
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-top: 4px;
}

/* ── Section Headers ── */
.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}

.section-header h2 {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.3px;
}

.section-header .section-line {
  flex: 1;
  height: 1px;
  background: var(--border-subtle);
}

/* ── Module Cards ── */
.modules-section {
  margin-bottom: 40px;
}

.modules-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

@media (max-width: 600px) {
  .modules-grid { grid-template-columns: 1fr; }
}

.module-card {
  opacity: 0;
  animation: fadeInUp 0.5s ease forwards;
}

.module-card-inner {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 20px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: border-color 0.25s, box-shadow 0.25s, background 0.25s;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.module-card-inner:hover {
  border-color: var(--border-medium);
  background: var(--bg-card-hover);
  box-shadow: 0 4px 24px rgba(0,0,0,0.2);
}

.module-card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}

.module-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.05);
  flex-shrink: 0;
}

.module-card-title-group { flex: 1; min-width: 0; }

.module-card-title {
  font-size: 14px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.module-card-desc {
  font-size: 11px;
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.module-card-gauge {
  display: flex;
  justify-content: center;
}

.module-card-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-secondary);
  flex-wrap: wrap;
  justify-content: center;
}

.module-status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}

.module-status-label { font-weight: 600; }

.module-meta-sep {
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: rgba(255,255,255,0.2);
  flex-shrink: 0;
}

.module-meta-item { color: var(--text-muted); }

/* ── Severity Bar ── */
.severity-section {
  margin-bottom: 40px;
}

.sev-bar-track {
  display: flex;
  height: 28px;
  border-radius: 14px;
  overflow: hidden;
  background: rgba(255,255,255,0.04);
  margin-bottom: 12px;
}

.sev-bar-seg {
  display: flex;
  align-items: center;
  justify-content: center;
  transition: width 0.8s cubic-bezier(0.23, 1, 0.32, 1);
  position: relative;
}

.sev-bar-label {
  font-size: 11px;
  font-weight: 700;
  color: #fff;
  text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

.sev-bar-empty {
  height: 28px;
  border-radius: 14px;
  background: rgba(34,197,94,0.08);
  border: 1px solid rgba(34,197,94,0.15);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  color: #22c55e;
  margin-bottom: 12px;
}

.sev-legend {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  justify-content: center;
}

.sev-legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.sev-legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  flex-shrink: 0;
}

.sev-legend-text {
  font-size: 12px;
  color: var(--text-secondary);
  font-weight: 500;
}

/* ── Findings ── */
.findings-section {
  margin-bottom: 40px;
}

.findings-module-group {
  margin-bottom: 28px;
}

.findings-module-header {
  margin-bottom: 14px;
}

.findings-module-title-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.module-icon-sm {
  display: flex;
  align-items: center;
}

.findings-module-name {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: -0.2px;
}

.findings-module-count {
  font-size: 11px;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 20px;
  letter-spacing: 0.3px;
}

.finding-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-left: 3px solid;
  border-radius: var(--radius-md);
  margin-bottom: 10px;
  transition: background 0.2s, border-color 0.2s;
}

.finding-card:hover {
  background: var(--bg-card-hover);
}

.finding-top {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px;
  flex-wrap: wrap;
}

.sev-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 10px;
  border-radius: 6px;
  font-size: 10px;
  font-weight: 800;
  color: #fff;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  flex-shrink: 0;
}

.finding-title {
  font-size: 14px;
  font-weight: 600;
  flex: 1;
  min-width: 0;
}

.finding-id {
  font-size: 11px;
  color: var(--text-muted);
  font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace;
  flex-shrink: 0;
}

.finding-details {
  border-top: 1px solid var(--border-subtle);
}

.finding-summary {
  padding: 10px 16px;
  font-size: 12px;
  font-weight: 600;
  color: var(--accent-blue);
  cursor: pointer;
  list-style: none;
  user-select: none;
  transition: color 0.15s;
}

.finding-summary::-webkit-details-marker { display: none; }

.finding-summary::before {
  content: "\\25B6";
  display: inline-block;
  margin-right: 8px;
  font-size: 9px;
  transition: transform 0.2s;
}

details[open] > .finding-summary::before {
  transform: rotate(90deg);
}

.finding-summary:hover {
  color: #60a5fa;
}

.finding-body {
  padding: 0 16px 16px;
}

.finding-section-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-muted);
  margin-bottom: 6px;
  margin-top: 12px;
}

.finding-section-label:first-child { margin-top: 0; }

.finding-evidence {
  font-size: 12px;
  color: var(--text-secondary);
  background: rgba(0,0,0,0.3);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-x: auto;
}

.finding-remediation {
  font-size: 13px;
  color: var(--text-primary);
  line-height: 1.6;
  padding: 8px 0 0;
}

/* No Findings */
.no-findings {
  text-align: center;
  padding: 48px 24px;
  background: rgba(34,197,94,0.04);
  border: 1px solid rgba(34,197,94,0.1);
  border-radius: var(--radius-lg);
}

.no-findings svg { margin-bottom: 16px; }

.no-findings-text {
  font-size: 20px;
  font-weight: 700;
  color: #22c55e;
  margin-bottom: 4px;
}

.no-findings-sub {
  font-size: 13px;
  color: var(--text-muted);
}

/* ── Footer ── */
.footer {
  text-align: center;
  padding-top: 40px;
  border-top: 1px solid var(--border-subtle);
}

.footer-logo {
  display: flex;
  justify-content: center;
  margin-bottom: 16px;
  opacity: 0.5;
  transition: opacity 0.2s;
}

.footer-logo:hover { opacity: 0.8; }

.footer-logo .ds-logo-svg { width: 140px; }

.footer-tagline {
  font-size: 13px;
  color: var(--text-muted);
  margin-bottom: 20px;
}

.footer-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 24px;
}

.btn-star {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 24px;
  background: var(--accent-gradient);
  color: #fff;
  border-radius: 10px;
  font-weight: 700;
  font-size: 14px;
  text-decoration: none;
  transition: transform 0.15s, box-shadow 0.15s;
  box-shadow: 0 2px 12px rgba(59,130,246,0.3);
}

.btn-star:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 20px rgba(59,130,246,0.4);
}

.btn-share {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 24px;
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--border-medium);
  color: var(--text-primary);
  border-radius: 10px;
  font-weight: 600;
  font-size: 14px;
  text-decoration: none;
  cursor: default;
  transition: background 0.15s;
}

.btn-share:hover {
  background: rgba(255,255,255,0.08);
}

.footer-meta {
  font-size: 11px;
  color: rgba(255,255,255,0.2);
  line-height: 1.8;
}

.footer-meta a {
  color: rgba(255,255,255,0.35);
  text-decoration: none;
  transition: color 0.15s;
}

.footer-meta a:hover { color: rgba(255,255,255,0.6); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ── Print ── */
@media print {
  body {
    background: #0a0e1a !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
    color-adjust: exact;
  }
  .hero { break-inside: avoid; }
  .module-card-inner { break-inside: avoid; }
  .finding-card { break-inside: avoid; }
  details { open: true; }
  .finding-details[open] .finding-body { display: block !important; }
}

/* ── Responsive ── */
@media (max-width: 480px) {
  .hero { padding: 36px 16px 32px; }
  .ds-logo-svg { width: 150px; }
  .hero-gauge-svg { width: 180px; height: 180px; }
  .stats-row { grid-template-columns: repeat(3, 1fr); }
  .finding-top { padding: 12px 14px; }
  .finding-id { display: none; }
}
</style>
</head>
<body>
<div class="container">

  <!-- Hero -->
  <div class="hero">
    <div class="hero-logo">${DEEPSAFE_LOGO}</div>
    <div class="hero-subtitle">
      <span>Preflight Security Scan &middot; ${escHtml(report.metadata.profile.toUpperCase())} mode &middot; ${escHtml(formatDate(report.metadata.generatedAt))}</span>
    </div>
    <div class="hero-gauge-wrap">${totalGauge}</div>
  </div>

  <!-- Stats -->
  <div class="stats-section fade-in">
    <div class="stats-row">
      <div class="stat-pill">
        <div class="stat-val">${totalFindings}</div>
        <div class="stat-lbl">Findings</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val" style="color:${highest !== "NONE" ? SEVERITY_COLORS[highest] ?? "var(--text-primary)" : "#22c55e"}">${escHtml(highest)}</div>
        <div class="stat-lbl">Highest Sev</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val" style="color:${SEVERITY_COLORS.CRITICAL}">${critCount}</div>
        <div class="stat-lbl">Critical</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val" style="color:${SEVERITY_COLORS.HIGH}">${highCount}</div>
        <div class="stat-lbl">High</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val" style="color:${SEVERITY_COLORS.MEDIUM}">${medCount}</div>
        <div class="stat-lbl">Medium</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">${escHtml(durationStr(report.metadata.durationMs))}</div>
        <div class="stat-lbl">Duration</div>
      </div>
    </div>
  </div>

  <!-- Modules -->
  <div class="modules-section">
    <div class="section-header">
      <h2>Module Scores</h2>
      <div class="section-line"></div>
    </div>
    <div class="modules-grid">
      ${moduleCards}
    </div>
  </div>

  <!-- Severity Breakdown -->
  <div class="severity-section fade-in" style="animation-delay:0.2s">
    <div class="section-header">
      <h2>Severity Breakdown</h2>
      <div class="section-line"></div>
    </div>
    ${severityBar}
  </div>

  <!-- Findings -->
  <div class="findings-section">
    <div class="section-header">
      <h2>Detailed Findings</h2>
      <div class="section-line"></div>
    </div>
    ${findingsHtml}
  </div>

  <!-- Footer -->
  <div class="footer">
    <div class="footer-logo">${DEEPSAFE_LOGO}</div>
    <div class="footer-tagline">AI Safety Evaluation Framework</div>
    <div class="footer-actions">
      <a class="btn-star" href="https://github.com/AI45Lab/DeepSafe" target="_blank" rel="noopener noreferrer">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
        Star on GitHub
      </a>
      <span class="btn-share">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
        Share this report
      </span>
    </div>
    <div class="footer-meta">
      Generated by <a href="https://github.com/AI45Lab/DeepSafe">openclaw-deepsafe</a> v${escHtml(report.metadata.pluginVersion)}<br>
      ${escHtml(formatDate(report.metadata.generatedAt))} &middot; Fingerprint: ${escHtml(report.metadata.fingerprint.slice(0, 12))}&hellip; &middot; Cache TTL: ${report.metadata.ttlDays}d
    </div>
  </div>

</div>
</body>
</html>`;
}
