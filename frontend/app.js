/* Retask Console — voice/text instruction -> TaskSpec.
   Deliberately unbound: SPACEBAR. It belongs to E-STOP once the car bridge
   lands, and a demo where space does two things is a demo that hurts someone. */
(function () {
  'use strict';

  /* Stages the perception pipeline actually emits, in the order its state
     machine produces them. model_loading / model_loaded only appear when the
     spec names a model that is not resident, so they are often skipped. The
     first two are ours: the pipeline knows nothing about the LLM compile. */
  var STAGES = ['received', 'compiled', 'model_loading', 'model_loaded', 'prepared', 'applied', 'active'];
  var OPTIONAL_STAGES = ['model_loading', 'model_loaded'];
  var BAD_STAGES = ['rejected', 'failed', 'no_target'];

  /* TargetState.phase — level-triggered, one per frame. Only TRACKING drives. */
  var PHASE_CLASS = {
    TRACKING: 'pass', ACQUIRING: 'pending', LOST: 'warn', NO_TARGET: 'warn',
    FAULT: 'fail', BOOTING: 'idle', IDLE: 'idle', SWITCHING: 'pending', LOADING_MODEL: 'pending'
  };
  var HISTORY_KEY = 'retask.history.v1';
  var HISTORY_MAX = 8;

  var CHIPS = [
    'track the pencil',
    'track the pencil and the eraser',
    'follow the person in red shoes',
    'follow the dog'
  ];

  var el = {};
  [
    'modePill', 'hdot', 'hlabel', 'ttsBtn', 'settingsBtn',
    'micBtn', 'micLabel', 'micStatus', 'level', 'instruction', 'chips',
    'sendBtn', 'clearBtn', 'validBadge', 'timer', 'timerLabel', 'timerSub',
    'stages', 'json', 'notices', 'copyBtn', 'downloadBtn',
    'history', 'historyCount', 'drawer', 'drawerScrim',
    'setMode', 'setApiBase', 'setWsUrl', 'setLang', 'setTts', 'setAutoSend',
    'setOpenaiModel', 'keyStatus', 'saveSettings', 'closeSettings', 'derivedWs',
    'setVideoUrl', 'videoFeed', 'videoPlaceholder', 'videoObjective', 'videoBadge',
    'videoReconnect', 'videoHint', 'cdot', 'clabel',
    'phaseChip', 'pipeReadout', 'targetReadout', 'stopBtn',
    'setPipelineBase', 'setSpecModel', 'setSendToPipeline'
  ].forEach(function (id) { el[id] = document.getElementById(id); });

  var settings = window.Api.loadSettings();
  var api = window.Api.create(settings);
  var pipeline = new window.Api.PipelineClient(settings);
  var socket = null;            // orchestrator /ws/status
  var eventSocket = null;       // pipeline /ws/events
  var targetSocket = null;      // pipeline /ws/target
  var knownModels = [];         // names from GET /models
  var lastPhase = null;

  var currentSpec = null;
  var pending = null;          // { id, text, t0, settled }
  var timerHandle = null;
  var history = [];
  var generation = 0;          // drops in-flight events from a superseded compile
  var attachedVideoUrl = null; // so a settings save only restarts the stream if the URL moved

  /* ================= settings ================= */

  var MODE_ORDER = ['openai', 'live', 'mock'];

  function applySettings() {
    api = window.Api.create(settings);

    var live = settings.mode === 'live';
    var openai = settings.mode === 'openai';
    el.modePill.textContent = openai ? 'OPENAI' : (live ? 'LIVE' : 'MOCK');
    el.modePill.classList.toggle('live', live);
    el.modePill.classList.toggle('openai', openai);
    el.hlabel.textContent = openai ? settings.openaiModel
      : (live ? settings.apiBase.replace(/^https?:\/\//, '') : 'mock adapter');

    el.ttsBtn.textContent = settings.tts ? 'VOICE ON' : 'VOICE OFF';
    el.ttsBtn.setAttribute('aria-pressed', settings.tts ? 'true' : 'false');

    if (settings.videoUrl !== attachedVideoUrl) {
      attachedVideoUrl = settings.videoUrl;
      videoBackoff = 1500;
      attachVideo();
    }

    if (socket) { socket.close(); socket = null; }
    if (live) {
      socket = new window.Api.StatusSocket(settings, { onEvent: onStatusEvent });
      socket.connect();
    }

    // The pipeline is the source of truth for stages and phase whether or not
    // the orchestrator is in the loop, so these stay connected in every mode.
    if (eventSocket) { eventSocket.close(); eventSocket = null; }
    if (targetSocket) { targetSocket.close(); targetSocket = null; }
    pipeline.settings = settings;
    eventSocket = new window.Api.StatusSocket(pipeline.eventsUrl(), { onEvent: onStatusEvent });
    eventSocket.connect();
    targetSocket = new window.Api.StatusSocket(pipeline.targetUrl(), { onEvent: onTargetState });
    targetSocket.connect();

    refreshModels();
    pollHealth();
  }

  function refreshModels() {
    pipeline.models().then(function (list) {
      knownModels = list.map(function (m) { return m.name; }).filter(Boolean);
      var sel = el.setSpecModel;
      if (!sel) return;
      sel.innerHTML = '';
      if (!knownModels.length) {
        sel.appendChild(new Option(settings.specModel + ' (registry unreachable)', settings.specModel));
        return;
      }
      list.forEach(function (m) {
        var label = m.name
          + (m.open_vocab ? ' · open vocab' : (m.classes ? ' · ' + m.classes.length + ' classes' : ''))
          + (m.available === false ? ' · UNAVAILABLE' : (m.loaded ? ' · loaded' : ''));
        var opt = new Option(label, m.name);
        opt.disabled = m.available === false;
        sel.appendChild(opt);
      });
      sel.value = settings.specModel;
    });
  }

  function openDrawer() {
    el.setMode.value = settings.mode;
    el.setOpenaiModel.value = settings.openaiModel;
    el.keyStatus.textContent = settings.openaiKey
      ? 'loaded, ' + settings.openaiKey.length + ' chars, ends ' + settings.openaiKey.slice(-4)
      : 'missing — create frontend/config.local.js';
    el.setPipelineBase.value = settings.pipelineBase;
    el.setSendToPipeline.checked = settings.sendToPipeline;
    el.setVideoUrl.value = settings.videoUrl;
    el.setApiBase.value = settings.apiBase;
    refreshModels();
    el.setWsUrl.value = settings.wsUrl;
    el.setLang.value = settings.lang;
    el.setTts.checked = settings.tts;
    el.setAutoSend.checked = settings.autoSend;
    el.derivedWs.textContent = window.Api.statusUrl({ apiBase: settings.apiBase, wsUrl: '' });
    el.drawer.hidden = false;
    el.drawerScrim.hidden = false;
  }

  function closeDrawer() {
    el.drawer.hidden = true;
    el.drawerScrim.hidden = true;
  }

  el.settingsBtn.addEventListener('click', openDrawer);
  el.closeSettings.addEventListener('click', closeDrawer);
  el.drawerScrim.addEventListener('click', closeDrawer);

  el.setApiBase.addEventListener('input', function () {
    el.derivedWs.textContent = window.Api.statusUrl({ apiBase: el.setApiBase.value.replace(/\/+$/, ''), wsUrl: '' });
  });

  el.saveSettings.addEventListener('click', function () {
    settings.mode = el.setMode.value;
    settings.openaiModel = el.setOpenaiModel.value.trim() || 'gpt-4o-mini';
    settings.pipelineBase = el.setPipelineBase.value.trim().replace(/\/+$/, '');
    settings.sendToPipeline = el.setSendToPipeline.checked;
    if (el.setSpecModel.value) settings.specModel = el.setSpecModel.value;
    settings.videoUrl = el.setVideoUrl.value.trim() || (settings.pipelineBase + '/video');
    settings.apiBase = el.setApiBase.value.trim().replace(/\/+$/, '');
    settings.wsUrl = el.setWsUrl.value.trim();
    settings.lang = el.setLang.value.trim() || 'en-US';
    settings.tts = el.setTts.checked;
    settings.autoSend = el.setAutoSend.checked;
    window.Api.saveSettings(settings);
    applySettings();
    closeDrawer();
  });

  el.modePill.addEventListener('click', function () {
    var i = MODE_ORDER.indexOf(settings.mode);
    settings.mode = MODE_ORDER[(i + 1) % MODE_ORDER.length];
    window.Api.saveSettings(settings);
    applySettings();
  });

  el.ttsBtn.addEventListener('click', function () {
    settings.tts = !settings.tts;
    window.Api.saveSettings(settings);
    if (!settings.tts && window.speechSynthesis) window.speechSynthesis.cancel();
    applySettings();
  });

  /* ================= chips ================= */

  CHIPS.forEach(function (text) {
    var b = document.createElement('button');
    b.className = 'chip';
    b.type = 'button';
    b.textContent = text;
    b.addEventListener('click', function () {
      el.instruction.value = text;
      el.instruction.focus();
      compile(text);
    });
    el.chips.appendChild(b);
  });

  /* ================= speech to text ================= */

  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  var recognition = null;
  var listening = false;
  var finalBuffer = '';
  var silenceTimer = null;

  if (!SR) {
    el.micBtn.disabled = true;
    el.micStatus.textContent = 'Speech recognition needs Chrome or Edge. Typing works everywhere.';
    el.micStatus.classList.add('error');
  }

  function setMicUi(on, message, isError) {
    listening = on;
    el.micBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    el.micLabel.textContent = on ? 'LISTENING' : 'SPEAK';
    if (message !== undefined) el.micStatus.textContent = message;
    el.micStatus.classList.toggle('error', !!isError);
  }

  function startListening() {
    if (!SR || listening) return;
    if (window.speechSynthesis) window.speechSynthesis.cancel();  // don't transcribe ourselves

    recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = settings.lang;
    recognition.maxAlternatives = 1;

    finalBuffer = el.instruction.value.trim();
    if (finalBuffer) finalBuffer += ' ';

    recognition.onstart = function () {
      setMicUi(true, 'Listening… click again to stop.');
      startLevelMeter();
    };

    recognition.onresult = function (event) {
      var interim = '';
      for (var i = event.resultIndex; i < event.results.length; i++) {
        var chunk = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalBuffer += chunk.trim() + ' ';
        } else {
          interim += chunk;
        }
      }
      el.instruction.value = (finalBuffer + interim).replace(/\s+/g, ' ').trim();
      el.instruction.classList.toggle('interim', interim.length > 0);

      if (settings.autoSend) {
        clearTimeout(silenceTimer);
        silenceTimer = setTimeout(function () {
          if (listening && el.instruction.value.trim()) {
            stopListening();
            compile(el.instruction.value);
          }
        }, 1500);
      }
    };

    recognition.onerror = function (event) {
      var messages = {
        'not-allowed': 'Microphone blocked. Allow it in the address bar, then try again.',
        'service-not-allowed': 'Microphone blocked by policy. Type instead.',
        'no-speech': 'Heard nothing. Try again, closer to the mic.',
        'audio-capture': 'No microphone found.',
        'network': 'Speech service unreachable — Chrome sends audio to Google, so this needs internet.'
      };
      if (event.error === 'aborted') return;   // our own stop() call
      setMicUi(false, messages[event.error] || ('Speech error: ' + event.error), true);
      stopLevelMeter();
    };

    recognition.onend = function () {
      clearTimeout(silenceTimer);
      el.instruction.classList.remove('interim');
      if (listening) setMicUi(false, el.instruction.value.trim() ? 'Got it — press Enter to compile.' : 'Click to speak, or just type below.');
      stopLevelMeter();
    };

    try {
      recognition.start();
    } catch (e) {
      setMicUi(false, 'Could not start the mic: ' + e.message, true);
    }
  }

  function stopListening() {
    if (!recognition) return;
    var wasListening = listening;
    listening = false;
    try { recognition.stop(); } catch (e) { /* already stopped */ }
    if (wasListening) setMicUi(false, el.instruction.value.trim() ? 'Got it — press Enter to compile.' : 'Click to speak, or just type below.');
    stopLevelMeter();
  }

  el.micBtn.addEventListener('click', function () {
    if (listening) stopListening(); else startListening();
  });

  /* ---- input level meter (cosmetic, fails soft) ---- */

  var audioCtx = null, analyser = null, micStream = null, meterRaf = null;
  var bars = Array.prototype.slice.call(el.level.querySelectorAll('i'));

  function startLevelMeter() {
    if (analyser || !navigator.mediaDevices) return;
    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stream) {
      micStream = stream;
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      var source = audioCtx.createMediaStreamSource(stream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      var data = new Uint8Array(analyser.frequencyBinCount);

      (function draw() {
        if (!analyser) return;
        meterRaf = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(data);
        var step = Math.floor(data.length / bars.length);
        var loud = 0;
        bars.forEach(function (bar, i) {
          var v = data[i * step] / 255;
          loud += v;
          bar.style.height = (3 + v * 23).toFixed(1) + 'px';
        });
        el.level.classList.toggle('hot', loud / bars.length > 0.06);
      })();
    }).catch(function () {
      /* No meter is fine — recognition still works. */
    });
  }

  function stopLevelMeter() {
    if (meterRaf) cancelAnimationFrame(meterRaf);
    meterRaf = null;
    analyser = null;
    el.level.classList.remove('hot');
    bars.forEach(function (bar) { bar.style.height = '3px'; });
    if (micStream) { micStream.getTracks().forEach(function (t) { t.stop(); }); micStream = null; }
    if (audioCtx) { audioCtx.close().catch(function () {}); audioCtx = null; }
  }

  /* ================= text to speech ================= */

  function speak(text) {
    if (!settings.tts || !window.speechSynthesis || !text) return;
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.rate = 1.05;
    u.lang = settings.lang;
    window.speechSynthesis.speak(u);
  }

  /* ================= timer ================= */

  function fmt(seconds) { return seconds.toFixed(2); }

  function renderTimer(seconds) {
    el.timer.innerHTML = fmt(seconds) + '<small>s</small>';
  }

  function startTimer() {
    stopTimer();
    el.timer.classList.remove('failed');
    el.timer.classList.add('running');
    el.timerLabel.textContent = 'compiling…';
    el.timerSub.textContent = '';
    var t0 = performance.now();
    pending.t0 = t0;
    renderTimer(0);
    timerHandle = setInterval(function () {
      renderTimer((performance.now() - t0) / 1000);
    }, 40);
  }

  function stopTimer() {
    if (timerHandle) clearInterval(timerHandle);
    timerHandle = null;
    el.timer.classList.remove('running');
  }

  function elapsed() {
    return pending ? (performance.now() - pending.t0) / 1000 : 0;
  }

  /* ================= stage strip ================= */

  function resetStages() {
    el.stages.innerHTML = '';
    STAGES.forEach(function (s) {
      var li = document.createElement('li');
      li.textContent = s.replace('model_', '');
      li.dataset.stage = s;
      li.title = s;
      // Only emitted when the spec names a model that is not already resident.
      if (OPTIONAL_STAGES.indexOf(s) !== -1) li.classList.add('optional');
      el.stages.appendChild(li);
    });
  }

  function markStage(stage) {
    var li = el.stages.querySelector('[data-stage="' + stage + '"]');
    if (li) { li.classList.add('done'); return; }
    if (BAD_STAGES.indexOf(stage) !== -1 || stage === 'clarify') {
      var extra = document.createElement('li');
      extra.textContent = stage;
      extra.dataset.stage = stage;
      extra.classList.add('bad');
      el.stages.appendChild(extra);
    }
  }

  /* ================= notices ================= */

  function clearNotices() { el.notices.innerHTML = ''; }

  function notice(kind, text, path) {
    var div = document.createElement('div');
    div.className = 'notice ' + kind;
    if (path) {
      var span = document.createElement('span');
      span.className = 'path';
      span.textContent = path;
      div.appendChild(span);
    }
    div.appendChild(document.createTextNode(text));
    el.notices.appendChild(div);
  }

  /* ================= compile ================= */

  function setBadge(cls, text) {
    el.validBadge.className = 'badge ' + cls;
    el.validBadge.textContent = text;
  }

  function compile(text) {
    text = String(text || '').trim();
    if (!text) {
      el.instruction.focus();
      return;
    }
    if (listening) stopListening();

    currentSpec = null;
    el.copyBtn.disabled = true;
    el.downloadBtn.disabled = true;
    el.sendBtn.classList.add('busy');
    el.json.innerHTML = '<span class="j-dim">// compiling…</span>';
    clearNotices();
    resetStages();
    setBadge('pending', 'COMPILING');

    if (pending) clearTimeout(pending.acquireTimer);
    pending = { id: null, text: text, t0: performance.now(), settled: false, compiled: false };
    var gen = ++generation;
    startTimer();
    markStage('received');

    api.compile(text, function (ev) {
      if (gen === generation) onStatusEvent(ev);
    }).then(function (res) {
      if (!pending || pending.settled) return;
      if (res.instruction_id) pending.id = res.instruction_id;
      if (res.spec) {
        deliverSpec(res.spec);
      } else {
        // Orchestrator acknowledged but the spec comes over /ws/status.
        setBadge('pending', 'AWAITING SPEC');
        el.json.innerHTML = '<span class="j-dim">// accepted as ' +
          (res.instruction_id || 'unknown id') +
          ' — waiting for the compiled event on ' + window.Api.statusUrl(settings) + '</span>';
      }
    }).catch(function (err) {
      fail(err.message || String(err));
    }).then(function () {
      el.sendBtn.classList.remove('busy');
    });
  }

  function dispatchEnabled() {
    // In live mode the orchestrator posts the spec on our behalf.
    return settings.sendToPipeline && settings.mode !== 'live';
  }

  function deliverSpec(spec) {
    if (!pending || pending.settled || pending.compiled) return;
    pending.compiled = true;
    pending.compileSeconds = elapsed();

    markStage('compiled');
    currentSpec = spec;
    el.json.innerHTML = window.TaskSpec.highlight(spec);
    setObjective(spec);
    el.copyBtn.disabled = false;
    el.downloadBtn.disabled = false;

    var result = window.TaskSpec.validate(spec, { models: knownModels });
    clearNotices();
    result.errors.forEach(function (e) { notice('error', e.msg, e.path); });
    result.warnings.forEach(function (w) { notice('warn', w.msg, w.path); });

    if (!result.ok) {
      // Never post a spec we already know the pipeline will 422.
      settle(pending.compileSeconds, false, 'invalid');
      setBadge('fail', 'INVALID');
      speak('Spec rejected. ' + result.errors.length + ' contract '
        + (result.errors.length === 1 ? 'error' : 'errors') + '.');
      return;
    }

    setBadge(result.warnings.length ? 'warn' : 'pass',
      result.warnings.length ? 'VALID · ' + result.warnings.length + ' WARN' : 'VALID');
    speak(window.TaskSpec.summarize(spec));
    el.timerSub.textContent = 'compiled in ' + fmt(pending.compileSeconds) + 's';

    if (!dispatchEnabled()) {
      settle(pending.compileSeconds, result.ok, 'compile time');
      return;
    }

    // Keep the clock running: the headline number is instruction -> active,
    // and the pipeline emits `active` once per spec when it reaches TRACKING.
    el.timerLabel.textContent = 'acquiring…';
    pipeline.sendSpec(pending.id, spec).then(function () {
      // If the pipeline never acquires, stop counting rather than run forever.
      pending.acquireTimer = setTimeout(function () {
        if (pending && !pending.settled) {
          settle(pending.compileSeconds, result.ok, 'compiled (never acquired)');
          notice('warn', 'no `active` event within 25s — the pipeline saw the spec but never reached TRACKING');
        }
      }, 25000);
    }).catch(function (e) {
      fail(e.message || String(e));
    });
  }

  /* Stop the clock and record the run. `seconds` is whatever number this run
     earned: retask time if the pipeline acquired, compile time otherwise. */
  function settle(seconds, ok, label) {
    if (!pending || pending.settled) return;
    pending.settled = true;
    clearTimeout(pending.acquireTimer);
    stopTimer();
    renderTimer(seconds);
    el.timerLabel.textContent = label;
    if (!ok) el.timer.classList.add('failed');
    pushHistory({
      text: pending.text,
      seconds: seconds,
      compileSeconds: pending.compileSeconds,
      spec: currentSpec,
      ok: ok,
      acquired: label === 'retask time'
    });
  }

  function fail(detail) {
    if (!pending || pending.settled) return;
    var seconds = elapsed();
    pending.settled = true;
    clearTimeout(pending.acquireTimer);
    stopTimer();
    renderTimer(seconds);
    el.timer.classList.add('failed');
    el.timerLabel.textContent = 'failed';
    setBadge('fail', 'ERROR');
    if (!pending.compiled) el.json.innerHTML = '<span class="j-dim">// no spec</span>';
    notice('error', detail || 'compile failed');
    speak('Failed.');
    pushHistory({ text: pending.text, seconds: seconds, spec: pending.compiled ? currentSpec : null, ok: false });
  }

  function clarify(question) {
    if (!pending) return;
    pending.settled = true;
    stopTimer();
    el.timerLabel.textContent = 'needs clarification';
    setBadge('warn', 'CLARIFY');
    clearNotices();
    notice('ask', question || 'The orchestrator needs more detail.');
    el.instruction.focus();
    el.instruction.select();
    speak(question || 'Which one did you mean?');
  }

  /* ================= status events ================= */

  function onStatusEvent(ev) {
    if (!ev || !ev.stage) return;
    // /ws/events replays its recent backlog on connect, so events from a spec
    // that predates this page load must not light up the strip.
    if (!pending) return;
    if (pending.id && ev.instruction_id && ev.instruction_id !== pending.id) return;

    markStage(ev.stage);

    if (ev.stage === 'compiled') {
      var spec = window.Api.extractSpec(ev.data) || (ev.data && ev.data.targets ? ev.data : null);
      if (spec) deliverSpec(spec);

    } else if (ev.stage === 'clarify') {
      clarify(ev.detail);

    } else if (ev.stage === 'active') {
      // The retask metric. Emitted once per spec, on reaching TRACKING.
      settle(elapsed(), true, 'retask time');
      setBadge('pass', 'TRACKING');
      speak('Target acquired.');

    } else if (ev.stage === 'no_target') {
      // Not a failure: the spec is live, nothing matching is in frame yet.
      notice('warn', ev.detail || 'nothing matching in frame — the spec is loaded and still looking');
      if (!pending.settled) settle(pending.compileSeconds || elapsed(), false, 'no target');
      setBadge('warn', 'NO TARGET');

    } else if (BAD_STAGES.indexOf(ev.stage) !== -1) {
      fail(ev.detail || ('pipeline stage: ' + ev.stage));
    }
  }

  /* One per frame off /ws/target. Level-triggered, so a dropped message
     self-corrects; never used to drive the stage strip. */
  function onTargetState(st) {
    if (!st || !st.phase) return;
    if (st.phase !== lastPhase) {
      lastPhase = st.phase;
      el.phaseChip.className = 'badge ' + (PHASE_CLASS[st.phase] || 'idle');
      el.phaseChip.textContent = st.phase;
    }
    var bits = [];
    if (st.label) bits.push(st.label);
    if (st.visible) {
      bits.push('cx ' + st.cx.toFixed(2), 'area ' + (st.area * 100).toFixed(1) + '%');
      if (typeof st.conf === 'number') bits.push('conf ' + st.conf.toFixed(2));
    }
    el.targetReadout.textContent = st.visible ? bits.join('  ·  ') : 'not visible';
    el.targetReadout.classList.toggle('none', !st.visible);
  }

  /* ================= history ================= */

  function loadHistory() {
    try {
      history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]') || [];
    } catch (e) {
      history = [];
    }
    renderHistory();
  }

  function pushHistory(entry) {
    history.unshift(entry);
    history = history.slice(0, HISTORY_MAX);
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch (e) { /* quota, ignore */ }
    renderHistory();
  }

  function renderHistory() {
    el.historyCount.textContent = String(history.length);
    el.history.innerHTML = '';
    if (!history.length) {
      var empty = document.createElement('li');
      empty.className = 'empty';
      empty.textContent = 'Nothing compiled yet.';
      el.history.appendChild(empty);
      return;
    }
    history.forEach(function (h) {
      var li = document.createElement('li');
      li.className = 'item' + (h.ok ? '' : ' bad');

      var time = document.createElement('span');
      time.className = 'h-time';
      time.textContent = fmt(h.seconds) + 's';

      var text = document.createElement('span');
      text.className = 'h-text';
      text.textContent = h.text;

      var tag = document.createElement('span');
      tag.className = 'h-tag';
      tag.textContent = h.spec ? (h.spec.mode || '') : 'error';

      li.appendChild(time);
      li.appendChild(text);
      li.appendChild(tag);
      li.addEventListener('click', function () {
        el.instruction.value = h.text;
        if (h.spec) {
          currentSpec = h.spec;
          el.json.innerHTML = window.TaskSpec.highlight(h.spec);
          setObjective(h.spec);
          renderTimer(h.seconds);
          el.timerLabel.textContent = 'recalled';
          el.timerSub.textContent = '';
          el.copyBtn.disabled = false;
          el.downloadBtn.disabled = false;
          var r = window.TaskSpec.validate(h.spec);
          setBadge(r.ok ? 'pass' : 'fail', r.ok ? 'VALID' : 'INVALID');
        }
        el.instruction.focus();
      });
      el.history.appendChild(li);
    });
  }

  /* ================= camera =================
     An MJPEG <img> dies quietly: connection refused, or the server goes away
     mid-stream and the picture just freezes forever. So: retry on error with
     backoff, and watch for a stall.

     Chrome fires a load event per MJPEG part, which makes a free frame
     heartbeat — but only some servers/browsers behave that way. The stall
     detector therefore arms itself only after it has seen several loads, so a
     single-load stream is never falsely torn down. */

  var videoFrame = document.querySelector('.video-frame');
  var videoRetry = null;
  var videoBackoff = 1500;
  var frameCount = 0;
  var lastFrameAt = 0;

  function setVideoState(live, note) {
    videoFrame.classList.toggle('live', live);
    el.videoBadge.className = 'badge ' + (live ? 'pass' : 'fail');
    el.videoBadge.textContent = live ? 'LIVE' : 'OFFLINE';
    el.cdot.className = 'hdot ' + (live ? 'up' : 'down');
    el.cdot.title = note || '';
    el.clabel.textContent = live ? 'camera' : 'no stream';
    el.videoHint.textContent = note || settings.videoUrl;
  }

  function attachVideo() {
    clearTimeout(videoRetry);
    frameCount = 0;
    if (!settings.videoUrl) {
      setVideoState(false, 'no video URL set — open SETTINGS');
      return;
    }
    setVideoState(false, 'connecting to ' + settings.videoUrl);
    var sep = settings.videoUrl.indexOf('?') === -1 ? '?' : '&';
    el.videoFeed.src = settings.videoUrl + sep + '_t=' + Date.now();
  }

  function scheduleVideoRetry(reason) {
    setVideoState(false, reason);
    clearTimeout(videoRetry);
    videoRetry = setTimeout(attachVideo, videoBackoff);
    videoBackoff = Math.min(videoBackoff * 1.4, 8000);
  }

  el.videoFeed.addEventListener('load', function () {
    frameCount += 1;
    lastFrameAt = Date.now();
    videoBackoff = 1500;
    setVideoState(true, settings.videoUrl);
  });

  el.videoFeed.addEventListener('error', function () {
    scheduleVideoRetry('cannot reach ' + settings.videoUrl);
  });

  el.videoReconnect.addEventListener('click', function () {
    videoBackoff = 1500;
    attachVideo();
  });

  // Bonus watchdog for browsers that emit a load event per MJPEG part. Chrome
  // does not for multipart streams, so this arms rarely — the origin probe
  // below is what actually catches a dead feed.
  setInterval(function () {
    if (frameCount > 3 && Date.now() - lastFrameAt > 5000) {
      scheduleVideoRetry('stream stalled — reconnecting');
    }
  }, 2000);

  /* When the server goes away mid-stream the <img> fires nothing at all: no
     error, no load, the last frame just sits there looking live forever. Tested
     it — 14 s after killing the stream the pane still said LIVE. So probe the
     stream host directly. Connection refused rejects; any HTTP answer, 404
     included, resolves. Works for Qinkai's pipeline and for a bare phone IP cam
     alike, because it only asks whether the host is still listening. */
  function probeVideoHost() {
    if (!settings.videoUrl || !videoFrame.classList.contains('live')) return;
    var origin;
    try {
      origin = new URL(settings.videoUrl, location.href).origin;
    } catch (e) {
      return;
    }
    var ctrl = new AbortController();
    var t = setTimeout(function () { ctrl.abort(); }, 2500);
    fetch(origin + '/', { mode: 'no-cors', cache: 'no-store', signal: ctrl.signal })
      .then(function () { clearTimeout(t); })
      .catch(function () {
        clearTimeout(t);
        if (videoFrame.classList.contains('live')) {
          scheduleVideoRetry('stream host stopped answering — reconnecting');
        }
      });
  }

  setInterval(probeVideoHost, 4000);

  function setObjective(spec) {
    if (!spec) {
      el.videoObjective.textContent = 'No objective';
      el.videoObjective.classList.add('none');
      return;
    }
    el.videoObjective.textContent = window.TaskSpec.summarize(spec);
    el.videoObjective.classList.remove('none');
  }

  /* ================= health ================= */

  function pollHealth() {
    if (settings.mode === 'mock') {
      el.hdot.className = 'hdot mock';
      el.hdot.title = 'mock adapter — no network';
      return;
    }
    api.health().then(function (r) {
      el.hdot.className = 'hdot ' + (r.up ? 'up' : 'down');
      el.hdot.title = r.detail;
    });
  }

  /* GET /health on the pipeline: fps, resident model and device are the three
     numbers worth glancing at while tuning on the day. */
  function pollPipelineHealth() {
    pipeline.health().then(function (r) {
      if (!r.up) {
        el.pipeReadout.textContent = '';
        if (lastPhase !== null) {
          lastPhase = null;
          el.phaseChip.className = 'badge idle';
          el.phaseChip.textContent = '—';
        }
        return;
      }
      var h = r.health || {};
      el.pipeReadout.textContent = [
        h.fps !== undefined ? h.fps + ' fps' : null,
        h.model || null,
        h.device || null
      ].filter(Boolean).join('  ·  ');
      // /ws/target is the live source; this only fills in before the first frame.
      if (lastPhase === null && h.phase) {
        el.phaseChip.className = 'badge ' + (PHASE_CLASS[h.phase] || 'idle');
        el.phaseChip.textContent = h.phase;
      }
    });
  }

  setInterval(pollPipelineHealth, 2000);

  setInterval(pollHealth, 2000);

  /* ================= output actions ================= */

  el.copyBtn.addEventListener('click', function () {
    if (!currentSpec) return;
    var text = JSON.stringify(window.TaskSpec.reorder(currentSpec), null, 2);
    navigator.clipboard.writeText(text).then(function () {
      var original = el.copyBtn.textContent;
      el.copyBtn.textContent = 'Copied';
      setTimeout(function () { el.copyBtn.textContent = original; }, 1200);
    });
  });

  el.downloadBtn.addEventListener('click', function () {
    if (!currentSpec) return;
    var blob = new Blob([JSON.stringify(window.TaskSpec.reorder(currentSpec), null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (currentSpec.spec_id || 'taskspec') + '.json';
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  });

  el.sendBtn.addEventListener('click', function () { compile(el.instruction.value); });

  el.clearBtn.addEventListener('click', function () {
    el.instruction.value = '';
    el.instruction.focus();
  });

  /* DELETE /spec — the pipeline drops to IDLE and publishes visible=false, so
     anything downstream stops. This replaced E-STOP when the car went away. */
  el.stopBtn.addEventListener('click', function () {
    if (pending && !pending.settled) settle(elapsed(), false, 'stopped');
    pipeline.clearSpec().then(function () {
      setObjective(null);
      setBadge('idle', 'IDLE');
      notice('warn', 'pipeline cleared — back to IDLE');
      speak('Stopped.');
    }).catch(function (e) {
      notice('error', 'could not clear the pipeline: ' + (e.message || e));
    });
  });

  /* ================= keyboard ================= */

  el.instruction.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      compile(el.instruction.value);
    }
  });

  document.addEventListener('keydown', function (e) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);

    if (e.key === 'Escape') {
      if (listening) stopListening();
      if (!el.drawer.hidden) closeDrawer();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && (e.key === 'm' || e.key === 'M')) {
      e.preventDefault();
      if (listening) stopListening(); else startListening();
      return;
    }
    if (!typing && (e.key === 'v' || e.key === 'V')) {
      el.ttsBtn.click();
    }
    // NOTE: space stays free for the E-STOP binding in the full operator UI.
  });

  /* ================= boot ================= */

  resetStages();
  loadHistory();
  setObjective(null);
  applySettings();
  pollPipelineHealth();
  el.instruction.focus();
})();
