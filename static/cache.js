/* Cache explorer. Reads the /cache-* endpoints, which already existed but had
 * no interface at all. */
(function () {
  "use strict";

  var tiles = document.getElementById("tiles");
  var volBars = document.getElementById("volBars");
  var entriesBox = document.getElementById("entries");
  var emptyBox = document.getElementById("empty");
  var filterInput = document.getElementById("filterInput");
  var purgeBtn = document.getElementById("purgeBtn");
  var alertBox = document.getElementById("alert");

  var ORDER = ["static", "slow", "dynamic", "realtime", "unknown"];
  var all = [];

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function human(seconds) {
    if (seconds == null || seconds < 0) return "unknown";
    if (seconds < 120) return seconds + "s";
    if (seconds < 7200) return Math.round(seconds / 60) + "m";
    if (seconds < 172800) return Math.round(seconds / 3600) + "h";
    return Math.round(seconds / 86400) + "d";
  }

  function showError(msg) {
    alertBox.innerHTML = '<span class="ico" aria-hidden="true">✕</span><span>' +
                         esc(msg) + "</span>";
    alertBox.classList.remove("hidden");
  }

  /* Headline counts are stat tiles, not a chart: five independent numbers with
   * no shared scale would make a misleading bar chart. */
  function renderTiles(stats) {
    var cells = [
      ["Entries", stats.total_queries || 0],
      ["Fresh", stats.fresh || 0],
      ["Expired", stats.expired || 0],
      ["Avg age", human(stats.avg_age_seconds)]
    ];
    if (stats.legacy_entries) cells.push(["Pre-TTL", stats.legacy_entries]);
    tiles.innerHTML = cells.map(function (c) {
      return '<div class="tile"><div class="v">' + esc(c[1]) +
             '</div><div class="k">' + esc(c[0]) + "</div></div>";
    }).join("");
  }

  /* Volatility is an ordered scale, so this uses one hue with monotone
   * lightness rather than four unrelated categorical colours. Each bar is
   * directly labelled, so no legend box is needed. */
  function renderVolBars(stats) {
    var by = stats.by_volatility || {};
    var present = ORDER.filter(function (k) { return by[k]; });
    if (!present.length) {
      volBars.innerHTML = '<div class="empty">No entries yet.</div>';
      return;
    }
    var max = Math.max.apply(null, present.map(function (k) { return by[k]; }));
    volBars.innerHTML = present.map(function (k) {
      var pct = max ? (by[k] / max) * 100 : 0;
      var colour = k === "unknown" ? "var(--ink-3)" : "var(--vol-" + k + ")";
      return '<div class="vol-bar"><span class="name">' + esc(k) + "</span>" +
             '<span class="track"><i style="width:' + pct.toFixed(1) +
             "%;background:" + colour + '"></i></span>' +
             '<span class="n">' + by[k] + "</span></div>";
    }).join("");
  }

  function renderEntries(items) {
    if (!items.length) {
      entriesBox.innerHTML = "";
      emptyBox.classList.remove("hidden");
      return;
    }
    emptyBox.classList.add("hidden");
    entriesBox.innerHTML = items.map(function (it) {
      var state = it.is_expired
        ? '<span class="badge critical"><span aria-hidden="true">✕</span>expired</span>'
        : '<span class="badge good"><span aria-hidden="true">✓</span>fresh</span>';
      var sources = (it.source_urls || []).length;
      return '<div class="entry' + (it.is_expired ? " expired" : "") + '">' +
        '<div class="top"><span class="q">' + esc(it.query) + "</span></div>" +
        '<div class="s">' + esc(it.summary) + "</div>" +
        '<div class="foot">' +
          '<span class="badge vol-' + esc(it.volatility) + '">' +
            '<span class="swatch" aria-hidden="true"></span>' + esc(it.volatility) +
          "</span>" + state +
          "<span>age " + esc(human(it.age_seconds)) + "</span><span class='sep'>·</span>" +
          "<span>ttl " + esc(human(it.ttl_seconds)) + "</span><span class='sep'>·</span>" +
          "<span>" + (it.hit_count || 0) + " hit" + (it.hit_count === 1 ? "" : "s") + "</span>" +
          (sources ? "<span class='sep'>·</span><span>" + sources + " source" +
                     (sources === 1 ? "" : "s") + "</span>" : "") +
          "<button class='btn-ghost' data-del='" + esc(it.id) + "'>Delete</button>" +
        "</div></div>";
    }).join("");
  }

  function applyFilter() {
    var term = filterInput.value.trim().toLowerCase();
    renderEntries(term
      ? all.filter(function (it) { return it.query.toLowerCase().indexOf(term) !== -1; })
      : all);
  }

  function load() {
    Promise.all([
      fetch("/cache-stats").then(function (r) { return r.json(); }),
      fetch("/cache-view").then(function (r) { return r.json(); })
    ]).then(function (res) {
      var stats = res[0], view = res[1];
      if (stats.error) { showError(stats.error); return; }
      renderTiles(stats);
      renderVolBars(stats);
      all = view.items || [];
      applyFilter();
    }).catch(function (e) { showError("Could not load the cache: " + e.message); });
  }

  filterInput.addEventListener("input", applyFilter);

  document.getElementById("filterForm").addEventListener("submit", function (e) {
    e.preventDefault();
  });

  entriesBox.addEventListener("click", function (e) {
    var id = e.target.getAttribute("data-del");
    if (!id) return;
    fetch("/cache-delete-item", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: id })
    }).then(load);
  });

  purgeBtn.addEventListener("click", function () {
    purgeBtn.disabled = true;
    fetch("/cache-purge", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function () { purgeBtn.disabled = false; load(); })
      .catch(function () { purgeBtn.disabled = false; });
  });

  load();
}());
