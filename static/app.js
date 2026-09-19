/* Search page behaviour: consumes the pipeline's SSE event stream and renders
 * the answer, its sources, and the decision trace.
 *
 * The trace is the interesting part. The pipeline reports which cache
 * candidates it considered, their similarity, and why it accepted or rejected
 * them -- so the page can show that it rejected a 0.954 cosine match because
 * the query said "bad" where the cached one said "good".
 */
(function () {
  "use strict";

  var form = document.getElementById("searchForm");
  var input = document.getElementById("queryInput");
  var button = document.getElementById("searchButton");
  var forceRefresh = document.getElementById("forceRefresh");
  var alertBox = document.getElementById("alert");
  var progressCard = document.getElementById("progressCard");
  var progressFill = document.getElementById("progressFill");
  var stagesBox = document.getElementById("stages");
  var result = document.getElementById("result");
  var badges = document.getElementById("badges");
  var traceBox = document.getElementById("traceBox");
  var traceBody = document.getElementById("traceBody");
  var summaryBox = document.getElementById("summary");
  var sourcesCard = document.getElementById("sourcesCard");
  var sourcesBox = document.getElementById("sources");
  var historyCard = document.getElementById("historyCard");
  var historyBox = document.getElementById("history");
  var selectAll = document.getElementById("selectAll");
  var deleteSelected = document.getElementById("deleteSelected");
  var clearHistory = document.getElementById("clearHistory");
  var copyBtn = document.getElementById("copyBtn");
  var refreshBtn = document.getElementById("refreshBtn");

  var source = null;
  var lastQuery = "";
  var lastSummary = "";

  var STAGE_LABELS = {
    validating: "Checking the query",
    classified: "Classified",
    cache: "Checking the cache",
    cache_miss: "Cache decision",
    searching: "Searching the web",
    found: "Results found",
    scraping: "Reading pages",
    read: "Pages read",
    summarizing: "Summarising",
    caching: "Saving",
    complete: "Done"
  };

  var VOLATILITY_BLURB = {
    static: "the answer does not really change",
    slow: "the answer drifts over weeks",
    dynamic: "the answer changes within a day",
    realtime: "the answer changes continuously"
  };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function host(url) {
    try { return new URL(url).hostname.replace(/^www\./, ""); } catch (e) { return url; }
  }

  function human(seconds) {
    if (!seconds || seconds <= 0) return "not cached";
    if (seconds < 120) return seconds + "s";
    if (seconds < 7200) return Math.round(seconds / 60) + "m";
    if (seconds < 172800) return Math.round(seconds / 3600) + "h";
    return Math.round(seconds / 86400) + "d";
  }

  /* Minimal markdown: bold, italic, inline code, links, bullets, paragraphs.
   * Everything is escaped first, so this never injects markup from a scraped
   * page. A full parser would be a dependency for very little gain. */
  function renderMarkdown(text) {
    var blocks = esc(text).trim().split(/\n{2,}/);
    return blocks.map(function (block) {
      var lines = block.split("\n");
      var isList = lines.every(function (l) { return /^\s*[-*+]\s+/.test(l); });
      var inline = function (s) {
        return s
          .replace(/`([^`]+)`/g, "<code>$1</code>")
          .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
          .replace(/(^|\s)\*([^*]+)\*/g, "$1<em>$2</em>")
          .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
                   '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
      };
      if (isList) {
        return "<ul>" + lines.map(function (l) {
          return "<li>" + inline(l.replace(/^\s*[-*+]\s+/, "")) + "</li>";
        }).join("") + "</ul>";
      }
      if (/^#{1,6}\s/.test(lines[0])) {
        return "<h3>" + inline(lines[0].replace(/^#{1,6}\s*/, "")) + "</h3>" +
               (lines.length > 1 ? "<p>" + inline(lines.slice(1).join(" ")) + "</p>" : "");
      }
      return "<p>" + inline(lines.join(" ")) + "</p>";
    }).join("");
  }

  // --- progress ---

  var seenStages = [];

  function resetUI() {
    alertBox.classList.add("hidden");
    result.classList.add("hidden");
    progressCard.classList.remove("hidden");
    progressFill.style.width = "0%";
    stagesBox.innerHTML = "";
    seenStages = [];
    traceBody.innerHTML = "";
    badges.innerHTML = "";
    sourcesBox.innerHTML = "";
    sourcesCard.classList.add("hidden");
  }

  function setBusy(busy) {
    button.disabled = busy;
    button.textContent = busy ? "Working…" : "Search";
  }

  function pushStage(stage, message) {
    if (stage === "complete" || stage === "error") return;
    var prev = stagesBox.querySelector(".stage.active");
    if (prev) { prev.classList.remove("active"); prev.classList.add("done"); }
    if (seenStages.indexOf(stage) !== -1) return;
    seenStages.push(stage);
    var el = document.createElement("div");
    el.className = "stage active";
    el.innerHTML = '<span class="dot"></span><span>' + esc(message) + "</span>";
    stagesBox.appendChild(el);
  }

  function finishStages() {
    Array.prototype.forEach.call(stagesBox.querySelectorAll(".stage"), function (el) {
      el.classList.remove("active");
      el.classList.add("done");
    });
  }

  function showError(message) {
    alertBox.innerHTML = '<span class="ico" aria-hidden="true">✕</span><span>' +
                         esc(message) + "</span>";
    alertBox.classList.remove("hidden");
    progressCard.classList.add("hidden");
    setBusy(false);
  }

  // --- rendering ---

  function renderBadges(data) {
    var v = data.verdict || {};
    var cache = data.cache || {};
    var out = [];

    if (v.volatility) {
      out.push('<span class="badge vol-' + esc(v.volatility) + '">' +
               '<span class="swatch" aria-hidden="true"></span>' +
               esc(v.volatility) + "</span>");
    }
    if (v.ttl_seconds != null) {
      out.push('<span class="badge">cache ' + esc(human(v.ttl_seconds)) + "</span>");
    }

    /* Status badges always pair a glyph with a word. Green and red measure
     * deltaE 4.1 apart under deuteranopia, so the colour cannot be the only
     * thing distinguishing "reused" from "fresh". */
    if (data.is_cached) {
      var pct = data.similarity != null
        ? " " + Math.round(data.similarity * 1000) / 10 + "%" : "";
      out.push('<span class="badge good"><span aria-hidden="true">✓</span>reused' +
               esc(pct) + "</span>");
    } else {
      out.push('<span class="badge"><span aria-hidden="true">↻</span>fresh answer</span>');
    }

    if (data.answered === false) {
      out.push('<span class="badge warning"><span aria-hidden="true">!</span>' +
               "incomplete — not cached</span>");
    }
    if (v.degraded) {
      out.push('<span class="badge warning"><span aria-hidden="true">!</span>' +
               "no LLM — deterministic fallback</span>");
    }
    var calls = (v.llm_calls || 0) + (cache.llm_calls || 0);
    out.push('<span class="badge">' + calls + " LLM call" +
             (calls === 1 ? "" : "s") + "</span>");

    badges.innerHTML = out.join("");
  }

  function renderTrace(data) {
    var v = data.verdict || {};
    var cache = data.cache || {};
    var steps = [];

    if (v.volatility) {
      var blurb = VOLATILITY_BLURB[v.volatility] || "";
      var via = v.source ? " via <em>" + esc(v.source) + "</em>" : "";
      steps.push(
        '<div class="trace-step"><div class="label">1 · Volatility</div>' +
        '<div class="detail">Classified <em>' + esc(v.volatility) + "</em>" + via +
        (blurb ? " — " + esc(blurb) : "") + ". Cacheable for <em>" +
        esc(human(v.ttl_seconds)) + "</em>." +
        (v.reason ? " " + esc(v.reason) : "") + "</div></div>"
      );
      if (v.escalated_from) {
        steps.push(
          '<div class="trace-step"><div class="label">Escalation</div>' +
          '<div class="detail">The model said <em>' + esc(v.escalated_from) +
          "</em>, but a pattern match indicated <em>" + esc(v.heuristic_floor || "") +
          "</em>. Heuristics may only raise volatility, so <em>" + esc(v.volatility) +
          "</em> wins.</div></div>"
        );
      }
    }

    var cands = cache.candidates || [];
    var detail = esc(cache.decision || cache.reason || "no cache lookup");
    if (cache.expired_skipped) {
      detail += " <em>" + cache.expired_skipped + " expired entr" +
                (cache.expired_skipped === 1 ? "y" : "ies") + " skipped.</em>";
    }
    var verifier = cache.verifier
      ? ' Verifier: <em>' + esc(cache.verifier) + "</em>." : "";

    var candHtml = "";
    if (cands.length) {
      candHtml = '<div class="cands">' + cands.map(function (c, i) {
        var chosen = data.is_cached && i === 0 &&
                     (cache.verifier === "auto_accept" || cache.verifier === "llm_accept");
        var pct = Math.max(0, Math.min(1, c.similarity)) * 100;
        var mark = chosen
          ? '<span class="pick" title="reused">\u2713 reused</span>'
          : '<span class="drop" title="not reused">\u2715</span>';
        return '<div class="cand' + (chosen ? " is-chosen" : "") + '">' +
               '<div class="row"><span class="q">' + esc(c.query) + "</span>" +
               mark +
               '<span class="score">' + c.similarity.toFixed(3) + "</span></div>" +
               '<div class="meter"><i style="width:' + pct.toFixed(1) + '%"></i></div>' +
               "</div>";
      }).join("") + "</div>";
    }

    steps.push(
      '<div class="trace-step"><div class="label">2 · Cache</div>' +
      '<div class="detail">' + detail + verifier + "</div>" + candHtml + "</div>"
    );

    if (!data.is_cached && data.pages_scraped) {
      var note = "";
      if (data.answered === false) {
        note = " The pages did not actually answer the question, so this was " +
               "<em>not stored</em> — a retry may find better sources.";
      } else if (cache.stored) {
        note = " Stored for <em>" + esc(human(v.ttl_seconds)) + "</em>.";
      }
      steps.push(
        '<div class="trace-step"><div class="label">3 · Fetch</div>' +
        '<div class="detail">Read <em>' + data.pages_scraped +
        "</em> pages, <em>" + (data.total_content_length || 0).toLocaleString() +
        "</em> characters, then summarised against the query." + note + "</div></div>"
      );
    }

    traceBody.innerHTML = steps.join("");
    traceBox.classList.remove("hidden");
  }

  function renderSources(sources) {
    if (!sources || !sources.length) { sourcesCard.classList.add("hidden"); return; }
    sourcesBox.innerHTML = sources.map(function (s) {
      var url = typeof s === "string" ? s : s.url;
      var title = (typeof s === "object" && s.title) ? s.title : host(url);
      if (!url) return "";
      var via = (typeof s === "object" && s.via) ? s.via : "";
      var h = host(url);
      // Show the hostname once: on a cache hit there may be no stored title,
      // in which case title already is the hostname and repeating it reads
      // like a rendering bug.
      var sub = (title === h) ? "" : '<span class="host">' + esc(h) + "</span>";
      return '<a class="source" href="' + esc(url) + '" target="_blank" rel="noopener noreferrer">' +
             '<span class="favicon" aria-hidden="true">' + esc(h.charAt(0).toUpperCase()) + "</span>" +
             '<span class="meta"><span class="title">' + esc(title) + "</span>" + sub + "</span>" +
             (via ? '<span class="via">' + esc(via) + "</span>" : "") +
             "</a>";
    }).join("");
    sourcesCard.classList.remove("hidden");
  }

  function showResult(data) {
    lastSummary = data.summary || "";
    summaryBox.innerHTML = renderMarkdown(lastSummary);
    renderBadges(data);
    renderTrace(data);
    renderSources(data.sources);
    progressCard.classList.add("hidden");
    result.classList.remove("hidden");
    setBusy(false);
    rememberQuery(lastQuery);
  }

  // --- history ---

  function loadHistory() {
    try { return JSON.parse(localStorage.getItem("history") || "[]"); }
    catch (e) { return []; }
  }

  function rememberQuery(q) {
    if (!q) return;
    var items = loadHistory().filter(function (x) { return x !== q; });
    items.unshift(q);
    saveHistory(items.slice(0, 12));
  }

  function saveHistory(items) {
    try { localStorage.setItem("history", JSON.stringify(items)); } catch (e) {}
    renderHistory();
  }

  function selectedQueries() {
    return Array.prototype.slice
      .call(historyBox.querySelectorAll("input[type=checkbox]:checked"))
      .map(function (box) { return box.value; });
  }

  function syncTools() {
    var boxes = historyBox.querySelectorAll("input[type=checkbox]");
    var chosen = selectedQueries().length;
    deleteSelected.disabled = chosen === 0;
    deleteSelected.textContent = chosen
      ? "Delete selected (" + chosen + ")" : "Delete selected";
    selectAll.checked = boxes.length > 0 && chosen === boxes.length;
    // Distinct from both checked and unchecked, so "some selected" is visible
    // rather than looking like "none selected".
    selectAll.indeterminate = chosen > 0 && chosen < boxes.length;
  }

  function renderHistory() {
    var items = loadHistory();
    if (!items.length) {
      historyCard.classList.add("hidden");
      return;
    }
    historyBox.innerHTML = items.map(function (q) {
      return '<div class="history-row">' +
             '<input type="checkbox" value="' + esc(q) +
             '" aria-label="Select \u201c' + esc(q) + '\u201d">' +
             '<button type="button" class="history-query">' + esc(q) + "</button>" +
             '<button type="button" class="history-remove" data-remove="' + esc(q) +
             '" aria-label="Remove \u201c' + esc(q) + '\u201d" title="Remove">' +
             "\u2715</button></div>";
    }).join("");
    historyCard.classList.remove("hidden");
    syncTools();
  }

  // --- run ---

  function run(query, force) {
    if (!query) return;
    lastQuery = query;
    input.value = query;
    resetUI();
    setBusy(true);
    if (source) source.close();

    var url = "/search_progress?query=" + encodeURIComponent(query) +
              (force ? "&refresh=1" : "");
    source = new EventSource(url);

    source.onmessage = function (event) {
      var data;
      try { data = JSON.parse(event.data); } catch (e) { return; }

      if (data.progress != null) progressFill.style.width = data.progress + "%";

      if (data.stage === "error") {
        showError(data.message);
        source.close();
        return;
      }
      if (data.stage === "complete") {
        progressFill.style.width = "100%";
        finishStages();
        showResult(data);
        source.close();
        return;
      }
      pushStage(data.stage, STAGE_LABELS[data.stage] || data.message);
    };

    source.onerror = function () {
      // A clean end-of-stream also lands here, so only complain if we never
      // produced a result.
      if (result.classList.contains("hidden") && !alertBox.classList.contains("hidden") === false) {
        showError("Connection lost before the answer arrived. Try again.");
      }
      if (source) source.close();
      setBusy(false);
    };
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    run(input.value.trim(), forceRefresh.checked);
  });

  document.getElementById("examples").addEventListener("click", function (e) {
    if (e.target.classList.contains("chip")) run(e.target.textContent.trim(), false);
  });

  historyBox.addEventListener("click", function (e) {
    var remove = e.target.getAttribute("data-remove");
    if (remove !== null) {
      saveHistory(loadHistory().filter(function (q) { return q !== remove; }));
      return;
    }
    if (e.target.classList.contains("history-query")) {
      run(e.target.textContent.trim(), false);
    }
  });

  historyBox.addEventListener("change", syncTools);

  selectAll.addEventListener("change", function () {
    Array.prototype.forEach.call(
      historyBox.querySelectorAll("input[type=checkbox]"),
      function (box) { box.checked = selectAll.checked; }
    );
    syncTools();
  });

  deleteSelected.addEventListener("click", function () {
    var doomed = selectedQueries();
    if (!doomed.length) return;
    saveHistory(loadHistory().filter(function (q) {
      return doomed.indexOf(q) === -1;
    }));
  });

  clearHistory.addEventListener("click", function () {
    if (loadHistory().length > 2 &&
        !window.confirm("Clear all recent queries?")) return;
    saveHistory([]);
  });

  refreshBtn.addEventListener("click", function () { run(lastQuery, true); });

  copyBtn.addEventListener("click", function () {
    navigator.clipboard.writeText(lastSummary).then(function () {
      copyBtn.textContent = "Copied";
      setTimeout(function () { copyBtn.textContent = "Copy"; }, 1400);
    });
  });

  window.addEventListener("beforeunload", function () { if (source) source.close(); });

  // Warn up front if search is not configured -- otherwise the first query
  // fails with something that looks like a bug.
  fetch("/healthz").then(function (r) { return r.json(); }).then(function (h) {
    var note = document.getElementById("backendNote");
    if (!h.search_configured) {
      showError("No TAVILY_API_KEY configured, so web search is unavailable. " +
                "Add one to .env — the free tier needs no card.");
    } else if (h.llm_provider === "none" || h.llm_provider === "") {
      note.textContent = "No LLM key — running deterministic fallbacks.";
      note.classList.remove("hidden");
    }
  }).catch(function () {});

  renderHistory();
}());
