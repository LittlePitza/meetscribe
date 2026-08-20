/* MeetScribe panel client.
 *
 * Three things this file is careful about:
 *
 * 1. RECONNECTION. EventSource reconnects on its own and replays from the last
 *    event id, so a refresh, a sleeping laptop or a throttled background tab all
 *    resume exactly where they left off. Nothing here re-fetches the transcript on
 *    reconnect, because the server's replay buffer already covers it.
 *
 * 2. DRAFT REPLACEMENT. A line is keyed by "track:seq". A draft renders in place
 *    and is overwritten by the final version carrying the same key, so provisional
 *    text never accumulates alongside the confirmed text that supersedes it.
 *
 * 3. NO innerHTML WITH SERVER DATA. Every piece of transcript text goes through
 *    textContent. The transcript contains whatever people said, and a meeting where
 *    someone reads out an HTML tag must not be able to alter the page.
 */

(() => {
  "use strict";

  const TOKEN = new URLSearchParams(location.search).get("t") || "";
  const LED_COUNT = 14;

  const el = {
    rec: document.getElementById("rec"),
    recLabel: document.getElementById("rec-label"),
    sessionName: document.getElementById("session-name"),
    elapsed: document.getElementById("elapsed"),
    stop: document.getElementById("stop"),
    transcript: document.getElementById("transcript"),
    transcriptEmpty: document.getElementById("transcript-empty"),
    speakers: document.getElementById("speakers"),
    actions: document.getElementById("actions"),
    actionsEmpty: document.getElementById("actions-empty"),
    actionCount: document.getElementById("action-count"),
    notices: document.getElementById("notices"),
    summary: document.getElementById("summary"),
    summaryBody: document.getElementById("summary-body"),
    summaryMeta: document.getElementById("summary-meta"),
    autoscroll: document.getElementById("autoscroll"),
    dialog: document.getElementById("rename-dialog"),
    renameForm: document.getElementById("rename-form"),
    renameInput: document.getElementById("rename-input"),
    renameCurrent: document.getElementById("rename-current"),
  };

  const state = {
    lines: new Map(),      // "track:seq" -> <li>
    speakers: new Map(),   // speaker_id -> {display_name, track, is_local, count}
    actions: new Set(),    // fingerprints already rendered
    elapsed: 0,
    running: false,
    renaming: null,
  };

  // ------------------------------------------------------------------ helpers

  function timestamp(seconds) {
    seconds = Math.max(0, Math.floor(seconds || 0));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    const pad = (n) => String(n).padStart(2, "0");
    return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
  }

  function buildLeds(container) {
    for (let i = 0; i < LED_COUNT; i += 1) {
      container.appendChild(document.createElement("i"));
    }
  }

  document.querySelectorAll(".leds").forEach(buildLeds);

  function post(path, body) {
    return fetch(path + "?t=" + encodeURIComponent(TOKEN), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // A custom header is what makes this request impossible to forge from
        // another origin: it forces a CORS preflight the server never answers.
        "X-MeetScribe-Token": TOKEN,
      },
      body: JSON.stringify(body || {}),
    });
  }

  // ------------------------------------------------------------------ notices

  function notice(level, message, code) {
    if (code) {
      const existing = el.notices.querySelector(`[data-code="${CSS.escape(code)}"]`);
      if (existing) existing.remove();
    }
    const box = document.createElement("div");
    box.className = "notice " + (level || "info");
    if (code) box.dataset.code = code;

    const text = document.createElement("span");
    text.className = "msg";
    text.textContent = message;

    // A real <button>, so the notice can be dismissed from the keyboard. The
    // glyph is decorative; the accessible name comes from aria-label.
    const close = document.createElement("button");
    close.type = "button";
    close.className = "close";
    close.textContent = "×";
    close.setAttribute("aria-label", "Descartar aviso");
    close.addEventListener("click", () => box.remove());

    box.append(text, close);
    el.notices.appendChild(box);
    if (level === "info") setTimeout(() => box.remove(), 12000);
  }

  // --------------------------------------------------------------- transcript

  function renderSegment(data) {
    const key = `${data.track}:${data.seq}`;
    let li = state.lines.get(key);
    if (!li) {
      li = document.createElement("li");
      li.className = "line " + data.track;
      const time = document.createElement("span");
      time.className = "t";
      const body = document.createElement("span");
      body.className = "body";
      const who = document.createElement("span");
      who.className = "who";
      const text = document.createElement("span");
      text.className = "text";
      body.append(who, text);
      li.append(time, body);
      state.lines.set(key, li);
      insertOrdered(li, data.start_s);
      el.transcriptEmpty.hidden = true;
    }
    li.dataset.start = String(data.start_s);
    li.classList.toggle("draft", data.state === "draft");
    li.querySelector(".t").textContent = timestamp(data.start_s);
    li.querySelector(".who").textContent = data.speaker_name + ":";
    li.querySelector(".text").textContent = data.text;
    li.dataset.speaker = data.speaker_id;
    autoscroll();
  }

  function insertOrdered(li, startS) {
    // Utterances usually arrive in order, but the two tracks are transcribed
    // through one queue, so a short remote turn can land after a long local one
    // that began earlier. Insert by start time, scanning from the end because
    // that is where the answer almost always is.
    const children = el.transcript.children;
    for (let i = children.length - 1; i >= 0; i -= 1) {
      if (parseFloat(children[i].dataset.start || "0") <= startS) {
        children[i].after(li);
        return;
      }
    }
    el.transcript.prepend(li);
  }

  function autoscroll() {
    if (!el.autoscroll.checked) return;
    el.transcript.scrollTop = el.transcript.scrollHeight;
  }

  // ----------------------------------------------------------------- speakers

  function renderSpeaker(data) {
    state.speakers.set(data.speaker_id, data);
    let li = el.speakers.querySelector(`[data-id="${CSS.escape(data.speaker_id)}"]`);
    if (!li) {
      li = document.createElement("li");
      li.dataset.id = data.speaker_id;
      const swatch = document.createElement("span");
      swatch.className = "swatch " + data.track;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "name-btn";
      button.addEventListener("click", () => openRename(data.speaker_id));
      const count = document.createElement("span");
      count.className = "count";
      li.append(swatch, button, count);
      el.speakers.appendChild(li);
    }
    li.querySelector(".name-btn").textContent = data.display_name;
    li.querySelector(".count").textContent =
      data.utterance_count ? `${data.utterance_count}` : "";

    // A rename must reach every line already on screen, not only new ones.
    el.transcript
      .querySelectorAll(`[data-speaker="${CSS.escape(data.speaker_id)}"] .who`)
      .forEach((node) => { node.textContent = data.display_name + ":"; });
  }

  function openRename(speakerId) {
    const speaker = state.speakers.get(speakerId);
    if (!speaker) return;
    state.renaming = speakerId;
    el.renameCurrent.textContent = `Actual: ${speaker.display_name}`;
    el.renameInput.value = speaker.display_name;
    el.dialog.showModal();
    el.renameInput.focus();
    el.renameInput.select();
  }

  el.renameForm.addEventListener("submit", (event) => {
    const action = event.submitter && event.submitter.value;
    const name = el.renameInput.value.trim();
    if (action !== "ok" || !name || !state.renaming) return;
    post("/api/speaker", { speaker_id: state.renaming, name })
      .then((r) => {
        if (!r.ok) throw new Error("HTTP " + r.status);
        const speaker = state.speakers.get(state.renaming);
        if (speaker) renderSpeaker({ ...speaker, display_name: name });
      })
      .catch(() => notice("error", "No se pudo cambiar el nombre.", "rename"));
  });

  // ------------------------------------------------------------------ actions

  function renderAction(data) {
    if (state.actions.has(data.fingerprint)) return;
    state.actions.add(data.fingerprint);
    const li = document.createElement("li");
    if (data.owner) {
      const owner = document.createElement("span");
      owner.className = "owner";
      owner.textContent = data.owner + ": ";
      li.appendChild(owner);
    }
    li.appendChild(document.createTextNode(data.text));
    if (data.due) {
      const due = document.createElement("span");
      due.className = "due";
      due.textContent = " — " + data.due;
      li.appendChild(due);
    }
    el.actions.appendChild(li);
    el.actionsEmpty.hidden = true;
    el.actionCount.textContent = String(state.actions.size);
  }

  // ------------------------------------------------------------------- levels

  function renderLevels(data) {
    Object.entries(data.tracks || {}).forEach(([track, reading]) => {
      const leds = document.querySelector(`.leds[data-track="${track}"]`);
      const label = document.querySelector(`[data-db="${track}"]`);
      if (!leds || !label) return;
      // -60 dBFS (inaudible) to 0 dBFS (clipping), mapped onto the LED strip.
      const filled = Math.round(
        Math.min(1, Math.max(0, (reading.dbfs + 60) / 60)) * LED_COUNT
      );
      Array.from(leds.children).forEach((led, index) => {
        led.classList.toggle("on", index < filled);
        led.classList.toggle("hot", index >= LED_COUNT - 2 && index < filled);
      });
      label.textContent = reading.dbfs <= -90 ? "--" : `${Math.round(reading.dbfs)} dB`;
      const meter = leds.closest(".meter");
      const silent = reading.dbfs <= -80;
      if (meter) meter.classList.toggle("silent", silent);
      // The wording lives in the DOM rather than in a CSS ::after, so a screen
      // reader announces it and the string stays with the rest of the copy.
      const warning = document.querySelector(`[data-warn="${track}"]`);
      if (warning) warning.hidden = !silent;
    });
  }

  function renderVad(data) {
    document.querySelectorAll(`.swatch.${data.track}`).forEach((node) => {
      node.classList.toggle("talking", !!data.speaking);
    });
  }

  // ------------------------------------------------------------------ session

  function renderSession(data) {
    el.sessionName.textContent = data.name || "MeetScribe";
    document.title = (data.name ? data.name + " — " : "") + "MeetScribe";
    state.elapsed = data.elapsed_s || 0;
    state.running = data.state === "running";

    const labels = {
      idle: "En espera",
      starting: "Iniciando",
      running: "Grabando",
      stopping: "Finalizando",
      stopped: "Finalizada",
      failed: "Con errores",
    };
    el.recLabel.textContent = labels[data.state] || data.state;
    el.rec.className =
      "rec " +
      (data.state === "running"
        ? "rec-on"
        : data.state === "stopped"
        ? "rec-done"
        : data.state === "failed"
        ? "rec-error"
        : "rec-off");
    el.stop.disabled = !state.running;
    if (data.offline_only) {
      notice("info", "Sesión solo-local: ningún texto sale de este equipo.", "offline");
    }
  }

  // ------------------------------------------------------------------ summary

  function list(title, items) {
    if (!items || !items.length) return null;
    const fragment = document.createDocumentFragment();
    const heading = document.createElement("h3");
    heading.textContent = title;
    const ul = document.createElement("ul");
    items.forEach((item) => {
      const li = document.createElement("li");
      if (typeof item === "string") {
        li.textContent = item;
      } else {
        if (item.owner) {
          const owner = document.createElement("span");
          owner.className = "owner";
          owner.textContent = item.owner + ": ";
          li.appendChild(owner);
        }
        li.appendChild(document.createTextNode(item.text));
        if (item.due) {
          const due = document.createElement("span");
          due.className = "due";
          due.textContent = " — " + item.due;
          li.appendChild(due);
        }
      }
      ul.appendChild(li);
    });
    fragment.append(heading, ul);
    return fragment;
  }

  function renderSummary(data) {
    el.summaryBody.replaceChildren();
    const body = document.createElement("div");
    body.className = "body";

    const exec = document.createElement("p");
    exec.className = "exec";
    exec.textContent = data.executive || "";
    body.appendChild(exec);

    [
      ["Temas tratados", data.topics],
      ["Decisiones", data.decisions],
      ["Compromisos", data.action_items],
      ["Preguntas abiertas", data.open_questions],
      ["Seguimiento", data.follow_ups],
    ].forEach(([title, items]) => {
      const block = list(title, items);
      if (block) body.appendChild(block);
    });

    el.summaryBody.appendChild(body);
    el.summaryMeta.textContent =
      data.model + (data.partial ? " · por fragmentos" : "");
    el.summary.hidden = false;
    (data.action_items || []).forEach(renderAction);
    el.summary.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // -------------------------------------------------------------------- clock

  setInterval(() => {
    if (state.running) state.elapsed += 1;
    el.elapsed.textContent = timestamp(state.elapsed);
  }, 1000);

  el.stop.addEventListener("click", () => {
    el.stop.disabled = true;
    post("/api/stop", {}).catch(() =>
      notice("error", "No se pudo detener la sesión desde el panel.", "stop")
    );
  });

  // --------------------------------------------------------------- connection

  function connect() {
    const source = new EventSource("/events?t=" + encodeURIComponent(TOKEN));

    const handlers = {
      session: renderSession,
      segment: renderSegment,
      speaker: renderSpeaker,
      level: renderLevels,
      vad: renderVad,
      action_item: renderAction,
      summary: renderSummary,
      notice: (d) => notice(d.level, d.message, d.code),
    };

    Object.entries(handlers).forEach(([name, handler]) => {
      source.addEventListener(name, (event) => {
        try {
          handler(JSON.parse(event.data).data);
        } catch (err) {
          console.error("evento", name, err);
        }
      });
    });

    source.addEventListener("open", () => {
      const stale = el.notices.querySelector('[data-code="conn"]');
      if (stale) stale.remove();
    });

    source.addEventListener("error", () => {
      // EventSource retries on its own and resumes from Last-Event-ID, so this is
      // informational: nothing is lost while it is disconnected.
      if (source.readyState === EventSource.CLOSED) {
        notice("warn", "Conexión con el panel cerrada.", "conn");
      }
    });
  }

  if (!TOKEN) {
    notice(
      "error",
      "Falta el token de acceso. Abre el panel con el enlace que imprimió " +
        "'meetscribe start'."
    );
  } else {
    connect();
  }
})();
