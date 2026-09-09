// Lite-AI backend proxy
//
// Why this exists: the API key must never live in the phone app. Anyone can
// unzip an APK or view page source and pull a key straight out of client
// code. This server is the only thing that talks to Gemini; the app talks
// only to this server.
//
// It also streams the model's response back to the client as it's
// generated (Server-Sent Events), instead of making the phone wait for the
// full answer — important on the slow/unstable cellular connections this
// project targets.

const express = require("express");
const cors = require("cors");

const app = express();

// --- Config ---------------------------------------------------------------
const PORT = process.env.PORT || 3000;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-flash-latest";

// Hard caps to protect both your API bill and low-end phones from huge
// payloads. Tune as needed.
const MAX_JSON_BODY = "12mb"; // raw request body limit (base64 inflates ~33%)
const MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024; // 8MB decoded file size cap
const MAX_HISTORY_TURNS = 20; // don't let a client send unbounded history

if (!GEMINI_API_KEY) {
  console.error("FATAL: GEMINI_API_KEY environment variable is not set.");
  process.exit(1);
}

app.use(cors()); // lock this down to your app's origin in production
app.use(express.json({ limit: MAX_JSON_BODY }));

// --- Health check (useful for uptime monitoring / load balancer probes) ---
app.get("/health", (req, res) => res.json({ ok: true }));

// --- Main chat endpoint -----------------------------------------------------
// Expects:
// {
//   "message": "user text",
//   "history": [ { "role": "user"|"model", "text": "..." }, ... ],
//   "file": { "mimeType": "image/png", "data": "<base64, no data: prefix>" } // optional
// }
//
// Responds with a text/event-stream of raw text chunks as they arrive from
// Gemini, terminated by a final "event: done" message.
app.post("/chat", async (req, res) => {
  try {
    const { message, history, file } = req.body || {};

    if ((!message || typeof message !== "string" || !message.trim()) && !file) {
      return res.status(400).json({ error: "message or file is required" });
    }

    if (file) {
      const validation = validateFile(file);
      if (validation.error) {
        return res.status(400).json({ error: validation.error });
      }
    }

    const contents = buildGeminiContents(history, message, file);

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    await streamFromGemini(contents, res);
  } catch (err) {
    console.error("Chat error:", err);
    // If headers are already sent (we were mid-stream), end the stream
    // instead of trying to send a second response.
    if (res.headersSent) {
      res.write(`event: error\ndata: ${JSON.stringify({ error: "upstream_failure" })}\n\n`);
      res.end();
    } else {
      res.status(500).json({ error: "Internal server error" });
    }
  }
});

// --- Helpers ----------------------------------------------------------------

function validateFile(file) {
  if (!file.mimeType || !file.data) {
    return { error: "file requires mimeType and data" };
  }
  const allowed = ["image/png", "image/jpeg", "image/webp", "application/pdf", "audio/mpeg", "audio/wav", "audio/mp4"];
  if (!allowed.includes(file.mimeType)) {
    return { error: `unsupported file type: ${file.mimeType}` };
  }
  // base64 length -> approx decoded byte size
  const approxBytes = (file.data.length * 3) / 4;
  if (approxBytes > MAX_ATTACHMENT_BYTES) {
    return { error: `file too large (max ${MAX_ATTACHMENT_BYTES / 1024 / 1024}MB)` };
  }
  return { error: null };
}

function buildGeminiContents(history, message, file) {
  const trimmedHistory = Array.isArray(history) ? history.slice(-MAX_HISTORY_TURNS) : [];

  const contents = trimmedHistory.map((turn) => ({
    role: turn.role === "model" ? "model" : "user",
    parts: [{ text: String(turn.text || "") }],
  }));

  const parts = [];
  if (message && message.trim()) parts.push({ text: message.trim() });
  if (file) {
    parts.push({
      inlineData: {
        mimeType: file.mimeType,
        data: file.data, // already base64, no prefix
      },
    });
  }

  contents.push({ role: "user", parts });
  return contents;
}

async function streamFromGemini(contents, res) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:streamGenerateContent` +
    `?alt=sse&key=${GEMINI_API_KEY}`;

  const upstream = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contents }),
  });

  if (!upstream.ok || !upstream.body) {
    const errText = await upstream.text().catch(() => "");
    console.error("Gemini upstream error:", upstream.status, errText);
    res.write(`event: error\ndata: ${JSON.stringify({ error: "model_unavailable" })}\n\n`);
    res.end();
    return;
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Gemini's SSE stream sends "data: {...}\n\n" blocks
    const lines = buffer.split("\n\n");
    buffer = lines.pop(); // keep any incomplete trailing chunk

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const jsonStr = trimmed.slice(5).trim();
      if (!jsonStr) continue;

      try {
        const parsed = JSON.parse(jsonStr);
        const text = parsed?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (text) {
          // relay just the text delta to the client, our own simple protocol
          res.write(`data: ${JSON.stringify({ text })}\n\n`);
        }
      } catch (e) {
        // ignore malformed/partial JSON fragments, they'll complete on next read
      }
    }
  }

  res.write("event: done\ndata: {}\n\n");
  res.end();
}

app.listen(PORT, () => {
  console.log(`Lite-AI backend listening on port ${PORT}`);
});
