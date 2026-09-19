/* API layer. Two adapters behind one compile(text):
     live  -> Danny's orchestrator (POST /instruction, WS /ws/status)
     mock  -> in-page keyword rules, no network
   Nothing about the endpoint is hardcoded: settings drawer, ?api= query
   param and localStorage all feed the same config. */
(function (global) {
  'use strict';

  var STORE_KEY = 'retask.settings.v1';

  var DEFAULTS = {
    mode: 'mock',                       // 'live' | 'openai' | 'mock'
    apiBase: 'http://localhost:8000',
    wsUrl: '',                          // blank = derive from apiBase
    videoUrl: 'http://localhost:8001/video',
    openaiModel: 'gpt-6-astra',         // frontier: most headroom on unrehearsed phrasing
    lang: 'en-US',
    tts: true,
    autoSend: false,
    requestTimeoutMs: 45000             // frontier models can stall; 15s was cutting it close
  };

  function loadSettings() {
    var s = {};
    try {
      s = JSON.parse(localStorage.getItem(STORE_KEY) || '{}') || {};
    } catch (e) {
      s = {};
    }
    var merged = Object.assign({}, DEFAULTS, s);

    // The key is only ever read from config.local.js (gitignored). It is
    // deliberately never written to localStorage — see saveSettings.
    var embedded = global.RETASK_CONFIG || {};
    merged.openaiKey = embedded.openaiKey || '';
    if (!s.openaiModel && embedded.openaiModel) merged.openaiModel = embedded.openaiModel;
    if (!s.mode && merged.openaiKey) merged.mode = 'openai';

    // Query params win, so the team can share a preconfigured link.
    var q = new URLSearchParams(location.search);
    if (q.get('api')) { merged.apiBase = q.get('api'); merged.mode = 'live'; }
    if (q.get('ws')) merged.wsUrl = q.get('ws');
    if (q.get('video')) merged.videoUrl = q.get('video');
    if (q.get('mock') === '1') merged.mode = 'mock';
    if (q.get('live') === '1') merged.mode = 'live';

    merged.apiBase = String(merged.apiBase || '').replace(/\/+$/, '');
    return merged;
  }

  function saveSettings(s) {
    var copy = Object.assign({}, s);
    delete copy.openaiKey;              // the key stays in config.local.js, nowhere else
    localStorage.setItem(STORE_KEY, JSON.stringify(copy));
  }

  function statusUrl(settings) {
    if (settings.wsUrl) return settings.wsUrl;
    return settings.apiBase.replace(/^http/, 'ws') + '/ws/status';
  }

  /* ---------- response shape tolerance ----------
     Until Danny's response shape is nailed down, accept any of:
       { spec_id, targets, ... }            the spec itself
       { spec: {...} } / { taskspec: {...} } / { data: {...} } ...
       { instruction_id }                   spec arrives later over /ws/status
  */
  function looksLikeSpec(v) {
    return v !== null && typeof v === 'object' && !Array.isArray(v) && Array.isArray(v.targets);
  }

  function extractSpec(payload) {
    if (looksLikeSpec(payload)) return payload;
    if (!payload || typeof payload !== 'object') return null;
    var keys = ['spec', 'taskspec', 'task_spec', 'data', 'result', 'compiled', 'program'];
    for (var i = 0; i < keys.length; i++) {
      var v = payload[keys[i]];
      if (looksLikeSpec(v)) return v;
      if (v && typeof v === 'object') {
        var nested = extractSpec(v);
        if (nested) return nested;
      }
    }
    return null;
  }

  function extractInstructionId(payload) {
    if (!payload || typeof payload !== 'object') return null;
    return payload.instruction_id || payload.id || payload.instructionId || null;
  }

  function friendlyError(e) {
    if (e && e.name === 'AbortError') return 'request timed out';
    if (e instanceof TypeError) {
      return 'could not reach the orchestrator (server down, wrong URL, or CORS not enabled on it)';
    }
    return (e && e.message) || String(e);
  }

  /* ---------- live adapter ---------- */
  function LiveApi(settings) {
    this.settings = settings;
  }

  LiveApi.prototype.compile = function (text) {
    var s = this.settings;
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, s.requestTimeoutMs);

    return fetch(s.apiBase + '/instruction', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text }),
      signal: ctrl.signal
    }).then(function (res) {
      return res.text().then(function (body) {
        if (!res.ok) throw new Error('orchestrator returned ' + res.status + ' ' + (body || res.statusText));
        var payload;
        try {
          payload = JSON.parse(body);
        } catch (e) {
          throw new Error('orchestrator returned non-JSON: ' + body.slice(0, 200));
        }
        return {
          instruction_id: extractInstructionId(payload),
          spec: extractSpec(payload),
          raw: payload
        };
      });
    }).catch(function (e) {
      throw new Error(friendlyError(e));
    }).then(function (r) {
      clearTimeout(timer);
      return r;
    }, function (e) {
      clearTimeout(timer);
      throw e;
    });
  };

  LiveApi.prototype.health = function () {
    var s = this.settings;
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, 1500);
    return fetch(s.apiBase + '/health', { signal: ctrl.signal })
      .then(function (res) {
        clearTimeout(timer);
        // Any HTTP answer means the process is alive and CORS is open.
        return { up: true, detail: 'HTTP ' + res.status };
      })
      .catch(function (e) {
        clearTimeout(timer);
        return { up: false, detail: friendlyError(e) };
      });
  };

  /* ---------- mock adapter ----------
     Keyword rules, not an LLM. Enough to exercise every UI path with no
     network at all, and to keep the demo alive if the laptop running the
     orchestrator falls over. */
  var NOUNS = [
    'person', 'people', 'human', 'dog', 'cat', 'pencil', 'eraser', 'pen', 'marker',
    'bottle', 'cup', 'mug', 'chair', 'backpack', 'bag', 'phone', 'book', 'ball',
    'laptop', 'keyboard', 'mouse', 'shoe', 'shoes', 'hat', 'box', 'can', 'car'
  ];
  var COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'grey', 'gray', 'brown'];
  var RELATION_WORDS = /\b(wearing|holding|carrying|with|in)\b/;
  var mockCounter = 0;

  function stripLeadingVerb(phrase) {
    return phrase
      .replace(/^(please\s+)?(can you\s+|could you\s+)?/, '')
      .replace(/^(go\s+)?(and\s+)?(follow|track|chase|center on|centre on|watch|look at|point at|stare at|find|target|drive to|go to)\s+/, '')
      .replace(/^(the|a|an|that|this)\s+/, '')
      .trim();
  }

  function findNoun(phrase) {
    var words = phrase.toLowerCase().replace(/[^a-z\s]/g, ' ').split(/\s+/).filter(Boolean);
    for (var i = words.length - 1; i >= 0; i--) {
      if (NOUNS.indexOf(words[i]) !== -1) {
        var n = words[i];
        if (n === 'people' || n === 'human') return 'person';
        if (n === 'shoes') return 'shoe';
        return n;
      }
    }
    return words.length ? words[words.length - 1] : 'object';
  }

  function buildTarget(phrase) {
    var ref = stripLeadingVerb(phrase);
    if (!ref) ref = phrase.trim();

    var target = { ref: ref, detect: [], select: 'largest' };
    var relMatch = ref.match(RELATION_WORDS);

    if (relMatch && relMatch.index > 0) {
      var head = ref.slice(0, relMatch.index).trim();
      var tail = ref.slice(relMatch.index + relMatch[0].length).trim();
      var headNoun = findNoun(head);
      var tailNoun = findNoun(tail);
      target.detect = [headNoun];
      if (tailNoun && tailNoun !== headNoun) target.detect.push(tailNoun);
      target.relate = { keep: headNoun, if_contains: tailNoun || tail };
      target.verify = { 'class': headNoun, text: ref, min_score: 0.25 };
    } else {
      var noun = findNoun(ref);
      target.detect = [noun];
      var hasColor = COLORS.some(function (c) { return ref.toLowerCase().indexOf(c) !== -1; });
      if (hasColor || ref.split(/\s+/).length > 1) {
        target.verify = { 'class': noun, text: ref, min_score: 0.25 };
      }
    }

    if (/\bcentered|middle\b/.test(ref)) target.select = 'most_centered';
    else if (/\bclosest|nearest|biggest|largest\b/.test(ref)) target.select = 'largest';
    else if (/\bconfident|best match\b/.test(ref)) target.select = 'highest_conf';

    return target;
  }

  function mockCompile(text) {
    var clean = String(text || '').trim();
    var lower = clean.toLowerCase();
    var mode = /\b(follow|chase|drive to|go to|approach)\b/.test(lower) ? 'follow' : 'center';

    var phrases = clean.split(/\s+and\s+(?:the\s+|a\s+)?/i).filter(function (p) { return p.trim(); });
    if (phrases.length > 2) phrases = [phrases[0], phrases[1]];

    var targets = phrases.map(buildTarget);
    if (!targets.length) targets = [buildTarget(clean)];

    mockCounter += 1;
    var spec = {
      spec_id: 'mock_' + String(mockCounter).padStart(3, '0'),
      targets: targets,
      mode: mode,
      model: 'yoloe'
    };
    if (targets.length === 2) spec.arbitration = { alternate_s: 5.0 };
    return spec;
  }

  function MockApi(settings) {
    this.settings = settings;
  }

  /* onStage lets the mock drive the same stage strip the real /ws/status does. */
  MockApi.prototype.compile = function (text, onStage) {
    var instructionId = 'mock_i_' + Date.now();

    // An empty or purely non-noun instruction is the clarify path.
    if (!String(text || '').trim()) {
      return Promise.reject(new Error('empty instruction'));
    }

    function emit(stage, delay, detail) {
      setTimeout(function () {
        if (onStage) onStage({ instruction_id: instructionId, stage: stage, ts: Date.now() / 1000, detail: detail });
      }, delay);
    }

    emit('received', 10);
    emit('compiled', 420);
    emit('sent', 470);
    emit('prepared', 700);
    emit('applied', 780);
    emit('active', 900);

    return new Promise(function (resolve) {
      setTimeout(function () {
        resolve({ instruction_id: instructionId, spec: mockCompile(text), raw: null });
      }, 420 + Math.random() * 260);
    });
  };

  MockApi.prototype.health = function () {
    return Promise.resolve({ up: true, detail: 'mock adapter — no network' });
  };

  /* ---------- OpenAI adapter ----------
     Compiles the instruction in the browser with the key from config.local.js.
     This is the standalone path: no orchestrator, no pipeline, no teammates.

     Note on the schema below: OpenAI strict structured outputs reject
     minItems / maxItems / minimum / maximum. The 1-2 target and 1-6 detect
     bounds are stated in the prompt and enforced by taskspec.js, which is
     exactly what the validator is for. */

  var TASKSPEC_SCHEMA = {
    type: 'object',
    additionalProperties: false,
    required: ['targets', 'mode', 'arbitration'],
    properties: {
      mode: { type: 'string', enum: ['follow', 'center'] },
      arbitration: {
        type: ['object', 'null'],
        additionalProperties: false,
        required: ['alternate_s'],
        properties: { alternate_s: { type: 'number' } }
      },
      targets: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['ref', 'detect', 'verify', 'relate', 'select'],
          properties: {
            ref: { type: 'string' },
            detect: { type: 'array', items: { type: 'string' } },
            verify: {
              type: ['object', 'null'],
              additionalProperties: false,
              required: ['class', 'text', 'min_score'],
              properties: {
                'class': { type: 'string' },
                text: { type: 'string' },
                min_score: { type: 'number' }
              }
            },
            relate: {
              type: ['object', 'null'],
              additionalProperties: false,
              required: ['keep', 'if_contains'],
              properties: {
                keep: { type: 'string' },
                if_contains: { type: 'string' }
              }
            },
            select: { type: 'string', enum: ['largest', 'most_centered', 'highest_conf', 'locked'] }
          }
        }
      }
    }
  };

  var SYSTEM_PROMPT = [
    'You compile an operator instruction into a TaskSpec for a robot car\'s perception pipeline.',
    'Return only the structured object.',
    '',
    'mode "follow": the car drives toward the target and keeps it centered.',
    'mode "center": the car rotates in place, no forward motion. Use it for track, watch,',
    'look at, keep an eye on, point at. Use "follow" for follow, chase, go to, approach.',
    '',
    'targets: 1 or 2, never more. Two only when the operator names two things of',
    'DIFFERENT kinds. Several of the same kind is still ONE target — "the two bottles",',
    '"both pencils", "whichever of the bottles is closest" all describe one target class,',
    'and select picks between the instances. Only a phrase like "the pencil and the',
    'eraser", naming two different classes, is two targets.',
    'Never emit more than two targets. If three or more things are named, keep the two',
    'most prominent and drop the rest — an over-long spec is rejected outright, so a',
    'partial objective beats no objective.',
    'With two targets set arbitration.alternate_s to 5.0; otherwise arbitration is null.',
    '',
    'detect: 1 to 6 short open-vocabulary prompts for YOLOE. Plain singular nouns',
    '("person", "dog", "pencil", "bottle"). Never slang or pronouns — "guy", "dude",',
    '"whoever", "it" all become "person" or the right concrete noun. Every human is',
    '"person": never "woman", "man", "lady", "kid", "guy". No colours or adjectives —',
    'detection is class-level only, attributes belong in verify.',
    'List only the target\'s own class plus any noun named in relate. Clothing, colours',
    'and held items that are not the relate noun stay out of detect and live in the',
    'verify text instead.',
    'detect is never empty, even when select is "locked": name the class being tracked.',
    '',
    'verify: a CLIP check, used when the operator gave an attribute detection cannot',
    'express (colour, pattern, writing). class is the detect noun it refines, text is the',
    'full natural description, min_score 0.25. null when the instruction has no attribute.',
    '',
    'relate: use when the target is defined by another object on or near it, e.g.',
    '"the person holding the blue bottle". keep is the noun to keep, if_contains is the',
    'noun that must be found with it. Both nouns must also appear in detect. null otherwise.',
    '',
    'select: "largest" by default. "most_centered" if they say centered or middle.',
    '"highest_conf" if they say best match. "locked" only if they say keep the current one.',
    '',
    'ref: a short human-readable name echoing the operator\'s words.',
    '',
    'Examples:',
    '"track the pencil" -> center; one target ref "pencil", detect ["pencil"], verify null,',
    '  relate null, select "largest"; arbitration null.',
    '"follow the person in red shoes" -> follow; ref "person in red shoes", detect ["person"],',
    '  verify {class "person", text "person in red shoes", min_score 0.25}, relate null.',
    '"track the pencil and the eraser" -> center; two targets; arbitration {alternate_s 5.0}.',
    '"follow the person holding the blue bottle" -> follow; detect ["person","bottle"],',
    '  relate {keep "person", if_contains "bottle"},',
    '  verify {class "person", text "person holding the blue bottle", min_score 0.25}.'
  ].join('\n');

  function newSpecId() {
    return 'spec_' + Date.now().toString(36).slice(-6);
  }

  /* Which models reject temperature:0. Learned at runtime and persisted, so a
     fresh page load at demo time does not pay the discovery round trip. */
  var TEMP_KEY = 'retask.rejectsTemperature.v1';

  function rejectsTemperature(model) {
    try {
      return (JSON.parse(localStorage.getItem(TEMP_KEY) || '[]') || []).indexOf(model) !== -1;
    } catch (e) {
      return false;
    }
  }

  function rememberRejectsTemperature(model) {
    try {
      var list = JSON.parse(localStorage.getItem(TEMP_KEY) || '[]') || [];
      if (list.indexOf(model) === -1) {
        list.push(model);
        localStorage.setItem(TEMP_KEY, JSON.stringify(list));
      }
    } catch (e) { /* private mode, fall back to retrying each load */ }
  }

  /* strict mode makes every field required, so optionals come back as null */
  function dropNulls(spec) {
    if (spec.arbitration === null) delete spec.arbitration;
    (spec.targets || []).forEach(function (t) {
      if (t.verify === null) delete t.verify;
      if (t.relate === null) delete t.relate;
    });
    // The model drops arbitration on two-target specs about half the time even
    // though the prompt asks for it. The value is fixed by the contract, so
    // fill it here instead of relying on the model to remember.
    if (Array.isArray(spec.targets) && spec.targets.length === 2 && !spec.arbitration) {
      spec.arbitration = { alternate_s: 5.0 };
    }
    return spec;
  }

  function OpenAiApi(settings) {
    this.settings = settings;
  }

  OpenAiApi.prototype.compile = function (text, onStage) {
    var s = this.settings;
    if (!s.openaiKey) {
      return Promise.reject(new Error('no API key — frontend/config.local.js is missing or empty'));
    }

    var instructionId = 'oa_' + Date.now();
    function emit(stage, detail) {
      if (onStage) onStage({ instruction_id: instructionId, stage: stage, ts: Date.now() / 1000, detail: detail });
    }
    emit('received');

    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, s.requestTimeoutMs);

    function attempt(withTemperature) {
      var body = {
        model: s.openaiModel,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: text }
        ],
        response_format: {
          type: 'json_schema',
          json_schema: { name: 'taskspec', strict: true, schema: TASKSPEC_SCHEMA }
        }
      };
      // temperature 0 keeps the demo repeatable, but the frontier models only
      // accept the default. Retry without it and remember, so each model pays
      // the extra round trip once ever rather than once per compile.
      if (withTemperature) body.temperature = 0;

      return fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + s.openaiKey
        },
        body: JSON.stringify(body),
        signal: ctrl.signal
      }).then(function (res) {
        return res.text().then(function (raw) {
          var payload = null;
          try { payload = JSON.parse(raw); } catch (e) { /* handled below */ }

          if (!res.ok) {
            var detail = (payload && payload.error && payload.error.message) || raw.slice(0, 200);
            if (res.status === 400 && withTemperature && /temperature/i.test(detail)) {
              rememberRejectsTemperature(s.openaiModel);
              return attempt(false);
            }
            if (res.status === 401) throw new Error('OpenAI rejected the key (401) — check frontend/config.local.js');
            if (res.status === 429) throw new Error('OpenAI rate limit or no credit (429): ' + detail);
            if (res.status === 404) throw new Error('model "' + s.openaiModel + '" not available on this key (404)');
            throw new Error('OpenAI ' + res.status + ': ' + detail);
          }

          var choice = payload && payload.choices && payload.choices[0];
          if (!choice) throw new Error('OpenAI returned no choices');
          if (choice.message && choice.message.refusal) throw new Error('model refused: ' + choice.message.refusal);

          var spec;
          try {
            spec = JSON.parse(choice.message.content);
          } catch (e) {
            throw new Error('model returned non-JSON: ' + String(choice.message.content).slice(0, 160));
          }

          spec.spec_id = newSpecId();
          spec.model = 'yoloe';
          dropNulls(spec);

          emit('compiled');
          return { instruction_id: instructionId, spec: spec, raw: payload };
        });
      });
    }

    return attempt(!rejectsTemperature(s.openaiModel)).catch(function (e) {
      throw new Error(friendlyError(e));
    }).then(function (r) {
      clearTimeout(timer);
      return r;
    }, function (e) {
      clearTimeout(timer);
      throw e;
    });
  };

  OpenAiApi.prototype.health = function () {
    // Deliberately no network call: polling OpenAI every 2s costs money and
    // burns rate limit. Presence of the key is the only thing worth reporting.
    var s = this.settings;
    return Promise.resolve(s.openaiKey
      ? { up: true, detail: 'key loaded from config.local.js · ' + s.openaiModel }
      : { up: false, detail: 'no key — create frontend/config.local.js' });
  };

  /* ---------- status socket (live mode only) ---------- */
  function StatusSocket(settings, handlers) {
    this.url = statusUrl(settings);
    this.handlers = handlers || {};
    this.ws = null;
    this.backoff = 1000;
    this.closed = false;
  }

  StatusSocket.prototype.connect = function () {
    var self = this;
    if (this.closed) return;

    var ws;
    try {
      ws = new WebSocket(this.url);
    } catch (e) {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = function () {
      self.backoff = 1000;
      if (self.handlers.onOpen) self.handlers.onOpen();
    };

    ws.onmessage = function (evt) {
      var data;
      try {
        data = JSON.parse(evt.data);
      } catch (e) {
        return;
      }
      if (self.handlers.onEvent) self.handlers.onEvent(data);
    };

    ws.onclose = function () {
      if (self.handlers.onClose) self.handlers.onClose();
      self.scheduleReconnect();
    };

    ws.onerror = function () {
      try { ws.close(); } catch (e) { /* onclose handles the retry */ }
    };
  };

  StatusSocket.prototype.scheduleReconnect = function () {
    var self = this;
    if (this.closed) return;
    setTimeout(function () { self.connect(); }, this.backoff);
    this.backoff = Math.min(this.backoff * 1.6, 5000);
  };

  StatusSocket.prototype.close = function () {
    this.closed = true;
    if (this.ws) {
      try { this.ws.close(); } catch (e) { /* already gone */ }
    }
  };

  global.Api = {
    DEFAULTS: DEFAULTS,
    SYSTEM_PROMPT: SYSTEM_PROMPT,       // exported so the model bench uses the real prompt
    TASKSPEC_SCHEMA: TASKSPEC_SCHEMA,
    loadSettings: loadSettings,
    saveSettings: saveSettings,
    statusUrl: statusUrl,
    extractSpec: extractSpec,
    create: function (settings) {
      if (settings.mode === 'live') return new LiveApi(settings);
      if (settings.mode === 'openai') return new OpenAiApi(settings);
      return new MockApi(settings);
    },
    StatusSocket: StatusSocket
  };
})(window);
