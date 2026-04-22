import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "";

// ── Markdown-like renderer (no external deps) ──────────────────────────────
function renderMarkdown(text) {
  if (!text) return "";
  return text
    // Bold **text**
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    // Italic *text*
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    // Code `inline`
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // Headings ### H3 ## H2 # H1
    .replace(/^### (.+)$/gm, '<h3 class="md-h3">$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 class="md-h2">$1</h2>')
    .replace(/^# (.+)$/gm, '<h1 class="md-h1">$1</h1>')
    // Unordered list
    .replace(/^[-•] (.+)$/gm, '<li class="md-li">$1</li>')
    .replace(/(<li class="md-li">.*<\/li>\n?)+/g, (m) => `<ul class="md-ul">${m}</ul>`)
    // Numbered list
    .replace(/^\d+\. (.+)$/gm, '<li class="md-oli">$1</li>')
    .replace(/(<li class="md-oli">.*<\/li>\n?)+/g, (m) => `<ol class="md-ol">${m}</ol>`)
    // Horizontal rule
    .replace(/^---$/gm, "<hr />")
    // Paragraphs (double newline)
    .replace(/\n\n+/g, '</p><p class="md-p">')
    // Wrap in paragraph
    .replace(/^(.+)$/, (m) => (m.startsWith("<") ? m : `<p class="md-p">${m}</p>`))
    // Single newline → <br>
    .replace(/\n/g, "<br />");
}

// ── Typing dots ────────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="typing-indicator" aria-label="PSYCH.AI is thinking">
      <span></span><span></span><span></span>
    </div>
  );
}

// ── Source badge ──────────────────────────────────────────────────────────
function SourceBadge({ source, sources, similarity }) {
  if (source === "rag") {
    return (
      <div className="source-badge rag-badge">
        <span className="badge-icon">📚</span>
        <span>
          From IGNOU material · {Math.round(similarity * 100)}% match
          {sources?.length > 0 && (
            <span className="source-files">
              {" "}· {sources.slice(0, 2).join(", ")}
              {sources.length > 2 && ` +${sources.length - 2} more`}
            </span>
          )}
        </span>
      </div>
    );
  }
  return (
    <div className="source-badge gemini-badge">
      <span className="badge-icon">✨</span>
      <span>PSYCH.AI expert knowledge</span>
    </div>
  );
}

// ── Single message bubble ─────────────────────────────────────────────────
function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`msg-row ${isUser ? "msg-row--user" : "msg-row--ai"}`}>
      {!isUser && (
        <div className="avatar ai-avatar" aria-hidden="true">
          <svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="18" cy="18" r="18" fill="url(#grad)" />
            <text x="18" y="24" textAnchor="middle" fontSize="16" fill="white">🧠</text>
            <defs>
              <linearGradient id="grad" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
                <stop stopColor="#7c3aed"/>
                <stop offset="1" stopColor="#4f46e5"/>
              </linearGradient>
            </defs>
          </svg>
        </div>
      )}
      <div className={`bubble ${isUser ? "bubble--user" : "bubble--ai"}`}>
        {isUser ? (
          <p>{msg.content}</p>
        ) : (
          <div
            className="ai-content"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
          />
        )}
        {/* Only show SourceBadge for Gemini answers, never for RAG */}
        {!isUser && msg.source === "gemini" && (
          <SourceBadge
            source={msg.source}
            sources={msg.rag_sources}
            similarity={msg.rag_similarity}
          />
        )}
        <span className="msg-time">{msg.time}</span>
      </div>
      {isUser && (
        <div className="avatar user-avatar" aria-hidden="true">
          <span>U</span>
        </div>
      )}
    </div>
  );
}

// ── Suggested prompts ─────────────────────────────────────────────────────
const SUGGESTED = [
  "Explain Freud's structural model of personality",
  "What is cognitive dissonance? Give examples",
  "How do I prepare for MPC-007 exams?",
  "Compare CBT and Psychoanalysis",
  "Explain operant conditioning with examples",
  "What are the defence mechanisms in psychology?",
];

// ── Friendly error message mapper ─────────────────────────────────────────
function getFriendlyError(raw) {
  const msg = raw || "";
  if (msg.includes("500"))
    return "PSYCH.AI hit a server hiccup — please try again in a moment.";
  if (msg.includes("429") || msg.toLowerCase().includes("rate limit"))
    return "We're getting a lot of questions right now! Please wait a few seconds and try again.";
  if (msg.includes("Failed to fetch") || msg.includes("NetworkError") || msg.includes("network"))
    return "Can't reach the server — please check your internet connection.";
  if (msg.includes("4000") || msg.toLowerCase().includes("too long"))
    return "Your message is too long. Please shorten it and try again.";
  if (msg.includes("timeout") || msg.includes("Timeout"))
    return "The request timed out — PSYCH.AI is thinking hard! Please try again.";
  if (msg.includes("503") || msg.toLowerCase().includes("unavailable"))
    return "PSYCH.AI is temporarily unavailable. Please try again in a moment.";
  return "Something went wrong. Please try again.";
}

// ── Main Chatbot Component ────────────────────────────────────────────────
export default function PsychChatbot() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "**Welcome to PSYCH.AI** 👋\n\nI'm your dedicated psychology tutor for IGNOU's MA Psychology programme. I'm here 24/7 to help you with:\n\n- **Concept explanations** across all MPC courses\n- **Exam preparation** and model answers\n- **Theory comparisons** (Freud vs Rogers, CBT vs Psychoanalysis, etc.)\n- **Research methods** and statistics help\n- **Case studies** and applied psychology\n\nWhat would you like to explore today?",
      source: null,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    },
  ]);
  const [input, setInput]     = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [ragChunks, setRagChunks] = useState(null);

  const bottomRef   = useRef(null);
  const inputRef    = useRef(null);
  const abortRef    = useRef(null);

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
    // Check RAG health
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.json())
      .then((d) => setRagChunks(d.rag_chunks))
      .catch(() => {});
  }, []);

  const sendMessage = useCallback(
    async (text) => {
      const userText = (text || input).trim();
      if (!userText || loading) return;

      setInput("");
      setError(null);

      const userMsg = {
        id: Date.now().toString(),
        role: "user",
        content: userText,
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);

      // Build history for API (exclude welcome message)
      const apiHistory = messages
        .filter((m) => m.id !== "welcome")
        .map((m) => ({ role: m.role === "assistant" ? "assistant" : "user", content: m.content }));

      abortRef.current = new AbortController();

      try {
        const res = await fetch(`${API_BASE}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortRef.current.signal,
          body: JSON.stringify({
            message: userText,
            history: apiHistory,
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server error ${res.status}`);
        }

        const data = await res.json();

        const aiMsg = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.reply,
          source: data.source,
          rag_sources: data.rag_sources || [],
          rag_similarity: data.rag_similarity || 0,
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        };

        setMessages((prev) => [...prev, aiMsg]);
      } catch (err) {
        if (err.name === "AbortError") return;
        // ── Friendly error messages ────────────────────────────────────
        setError(getFriendlyError(err.message));
        // ── end friendly error ─────────────────────────────────────────
        setMessages((prev) => prev.filter((m) => m.id !== userMsg.id));
        setInput(userText); // restore input
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [input, loading, messages]
  );

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setLoading(false);
  };

  const clearChat = () => {
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content:
          "Chat cleared. Let's start fresh! What psychology topic would you like to explore?",
        source: null,
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
    setError(null);
  };

  const showSuggestions = messages.length <= 1 && !loading;

  return (
    <>
      <style>{CSS}</style>
      <div className="psych-app">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <header className="chat-header">
          <div className="header-left">
            <div className="logo-mark">
              <span>🧠</span>
            </div>
            <div className="header-titles">
              <h1 className="header-name">PSYCH<span className="dot-ai">.AI</span></h1>
              <p className="header-sub">IGNOU MA Psychology Tutor</p>
            </div>
          </div>
          <div className="header-right">
            {ragChunks !== null && (
              <div className={`rag-status ${ragChunks > 0 ? "rag-active" : "rag-empty"}`}>
                <span className="rag-dot"></span>
                <span>{ragChunks > 0 ? `${ragChunks.toLocaleString()} chunks indexed` : "No docs — add PDFs"}</span>
              </div>
            )}
            <button className="icon-btn" onClick={clearChat} title="Clear conversation">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6"/>
                <path d="M19 6l-1 14H6L5 6"/>
                <path d="M10 11v6M14 11v6"/>
                <path d="M9 6V4h6v2"/>
              </svg>
            </button>
          </div>
        </header>

        {/* ── Messages ────────────────────────────────────────────────── */}
        <main className="messages-area" role="log" aria-live="polite">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} msg={msg} />
          ))}

          {loading && (
            <div className="msg-row msg-row--ai">
              <div className="avatar ai-avatar" aria-hidden="true">
                <svg viewBox="0 0 36 36" fill="none">
                  <circle cx="18" cy="18" r="18" fill="url(#grad2)" />
                  <text x="18" y="24" textAnchor="middle" fontSize="16" fill="white">🧠</text>
                  <defs>
                    <linearGradient id="grad2" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
                      <stop stopColor="#7c3aed"/>
                      <stop offset="1" stopColor="#4f46e5"/>
                    </linearGradient>
                  </defs>
                </svg>
              </div>
              <div className="bubble bubble--ai bubble--thinking">
                <TypingIndicator />
              </div>
            </div>
          )}

          {error && (
            <div className="error-bar">
              <span>⚠ {error}</span>
              <button onClick={() => setError(null)}>✕</button>
            </div>
          )}

          {/* Suggested prompts */}
          {showSuggestions && (
            <div className="suggestions">
              <p className="suggestions-label">Try asking…</p>
              <div className="suggestions-grid">
                {SUGGESTED.map((s) => (
                  <button
                    key={s}
                    className="suggestion-chip"
                    onClick={() => sendMessage(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </main>

        {/* ── Input bar ───────────────────────────────────────────────── */}
        <footer className="input-bar">
          <div className="input-wrap">
            <textarea
              ref={inputRef}
              className="input-field"
              placeholder="Ask me anything about psychology…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={loading}
              aria-label="Message input"
            />
            {loading ? (
              <button className="send-btn stop-btn" onClick={handleStop} title="Stop">
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="6" width="12" height="12" rx="2"/>
                </svg>
              </button>
            ) : (
              <button
                className="send-btn"
                onClick={() => sendMessage()}
                disabled={!input.trim()}
                title="Send (Enter)"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="22" y1="2" x2="11" y2="13"/>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                </svg>
              </button>
            )}
          </div>
          <p className="input-hint">
            Enter to send · Shift+Enter for new line · Responses grounded in IGNOU course material
          </p>
        </footer>
      </div>
    </>
  );
}

// ── Styles (all-in-one) ────────────────────────────────────────────────────
const CSS = `
  :root {
    --bg:        #0f0f1a;
    --surface:   #16162a;
    --surface2:  #1e1e35;
    --border:    #2a2a4a;
    --purple:    #7c3aed;
    --indigo:    #4f46e5;
    --violet:    #8b5cf6;
    --text:      #e2e8f0;
    --muted:     #94a3b8;
    --user-bg:   #1d1b4b;
    --ai-bg:     #1a1a2e;
    --error:     #ef4444;
    --success:   #22c55e;
    --radius:    14px;
    --font:      'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body, #root { height: 100%; }

  .psych-app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    overflow: hidden;
  }

  /* ── Header ── */
  .chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    backdrop-filter: blur(12px);
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .logo-mark {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--purple), var(--indigo));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 18px rgba(124,58,237,0.35);
  }
  .header-name {
    font-size: 20px; font-weight: 700; letter-spacing: -0.5px;
    color: #fff;
  }
  .dot-ai { color: var(--violet); }
  .header-sub { font-size: 11px; color: var(--muted); margin-top: 1px; }
  .header-right { display: flex; align-items: center; gap: 10px; }

  .rag-status {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--muted);
    background: var(--surface2); border: 1px solid var(--border);
    padding: 5px 10px; border-radius: 20px;
  }
  .rag-dot {
    width: 7px; height: 7px; border-radius: 50%;
  }
  .rag-active .rag-dot { background: var(--success); }
  .rag-empty  .rag-dot { background: #f59e0b; }

  .icon-btn {
    width: 36px; height: 36px; border-radius: 8px;
    background: transparent; border: 1px solid var(--border);
    color: var(--muted); cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all .15s;
  }
  .icon-btn svg { width: 16px; height: 16px; }
  .icon-btn:hover { background: var(--surface2); color: var(--text); }

  /* ── Messages area ── */
  .messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }
  .messages-area::-webkit-scrollbar { width: 5px; }
  .messages-area::-webkit-scrollbar-track { background: transparent; }
  .messages-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* ── Message rows ── */
  .msg-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    max-width: 820px;
    animation: fadeUp .2s ease;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .msg-row--user {
    flex-direction: row-reverse;
    align-self: flex-end;
  }
  .msg-row--ai { align-self: flex-start; }

  .avatar {
    width: 36px; height: 36px; border-radius: 50%;
    flex-shrink: 0; overflow: hidden;
  }
  .ai-avatar svg { width: 100%; height: 100%; }
  .user-avatar {
    background: linear-gradient(135deg, var(--indigo), var(--purple));
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 600; color: #fff;
  }

  /* ── Bubbles ── */
  .bubble {
    padding: 12px 16px;
    border-radius: var(--radius);
    max-width: 680px;
    line-height: 1.65;
    font-size: 14.5px;
    position: relative;
  }
  .bubble--user {
    background: var(--user-bg);
    border: 1px solid rgba(79,70,229,0.35);
    border-bottom-right-radius: 4px;
    color: #c7d2fe;
  }
  .bubble--ai {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .bubble--thinking {
    padding: 16px 20px;
    min-width: 80px;
  }

  /* ── AI markdown content ── */
  .ai-content p.md-p { margin-bottom: 10px; }
  .ai-content h1.md-h1, .ai-content h2.md-h2, .ai-content h3.md-h3 {
    color: #c4b5fd; font-weight: 600; margin: 14px 0 8px;
  }
  .ai-content h1.md-h1 { font-size: 17px; }
  .ai-content h2.md-h2 { font-size: 15.5px; }
  .ai-content h3.md-h3 { font-size: 14.5px; }
  .ai-content ul.md-ul, .ai-content ol.md-ol {
    padding-left: 18px; margin: 8px 0;
  }
  .ai-content li.md-li, .ai-content li.md-oli { margin-bottom: 4px; }
  .ai-content strong { color: #e0d7ff; }
  .ai-content em { color: var(--muted); }
  .ai-content code {
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.3);
    padding: 1px 6px; border-radius: 4px;
    font-size: 12.5px; font-family: 'Fira Code', monospace;
    color: #c4b5fd;
  }
  .ai-content hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }

  /* ── Source badge ── */
  .source-badge {
    display: flex; align-items: center; gap: 6px;
    margin-top: 10px; padding: 6px 10px;
    border-radius: 8px; font-size: 11px; color: var(--muted);
  }
  .rag-badge { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2); }
  .gemini-badge { background: rgba(124,58,237,0.08); border: 1px solid rgba(124,58,237,0.2); }
  .badge-icon { font-size: 13px; }
  .source-files { opacity: .75; }

  .msg-time {
    display: block; text-align: right;
    font-size: 10px; color: var(--muted);
    margin-top: 6px; opacity: .6;
  }

  /* ── Typing indicator ── */
  .typing-indicator {
    display: flex; align-items: center; gap: 5px;
    height: 20px;
  }
  .typing-indicator span {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--violet); opacity: .5;
    animation: bounce 1.2s infinite;
  }
  .typing-indicator span:nth-child(2) { animation-delay: .2s; }
  .typing-indicator span:nth-child(3) { animation-delay: .4s; }
  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity:.5; }
    30%            { transform: translateY(-6px); opacity:1; }
  }

  /* ── Suggestions ── */
  .suggestions { margin-top: 8px; }
  .suggestions-label {
    font-size: 12px; color: var(--muted);
    margin-bottom: 10px; text-transform: uppercase; letter-spacing: .05em;
  }
  .suggestions-grid {
    display: flex; flex-wrap: wrap; gap: 8px;
  }
  .suggestion-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 8px 14px; border-radius: 20px;
    font-size: 13px; cursor: pointer;
    transition: all .15s;
  }
  .suggestion-chip:hover {
    border-color: var(--violet);
    color: var(--text);
    background: rgba(124,58,237,0.1);
  }

  /* ── Error bar ── */
  .error-bar {
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    padding: 10px 14px; border-radius: 10px;
    font-size: 13px; color: #fca5a5;
    animation: fadeUp .2s ease;
  }
  .error-bar button {
    background: none; border: none; color: #fca5a5;
    cursor: pointer; font-size: 14px; padding: 0 4px;
  }

  /* ── Input bar ── */
  .input-bar {
    padding: 16px 20px 12px;
    background: var(--surface);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }
  .input-wrap {
    display: flex; align-items: flex-end; gap: 10px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 12px;
    transition: border-color .15s;
  }
  .input-wrap:focus-within { border-color: var(--violet); }
  .input-field {
    flex: 1; resize: none; border: none; outline: none;
    background: transparent; color: var(--text);
    font-family: var(--font); font-size: 14.5px;
    line-height: 1.5; max-height: 160px; overflow-y: auto;
    scrollbar-width: thin;
  }
  .input-field::placeholder { color: var(--muted); }
  .input-field:disabled { opacity: .5; }

  .send-btn {
    width: 38px; height: 38px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--purple), var(--indigo));
    border: none; color: #fff; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: opacity .15s, transform .1s;
  }
  .send-btn svg { width: 18px; height: 18px; }
  .send-btn:hover:not(:disabled) { opacity: .88; transform: scale(1.04); }
  .send-btn:disabled { opacity: .35; cursor: not-allowed; }
  .stop-btn { background: linear-gradient(135deg, #ef4444, #b91c1c); }

  .input-hint {
    text-align: center; font-size: 11px;
    color: var(--muted); margin-top: 8px; opacity: .6;
  }

  /* ── Scrollbar hint on mobile ── */
  @media (max-width: 600px) {
    .chat-header { padding: 12px 14px; }
    .messages-area { padding: 16px 12px; }
    .input-bar { padding: 12px 14px 10px; }
    .rag-status { display: none; }
    .bubble { font-size: 14px; }
  }
`;