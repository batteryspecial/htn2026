/* TaskSpec — client-side mirror of perception/contracts.py.
   Validates a compiled spec before it is posted, so a bad one shows up here in
   red rather than as a 422 from the pipeline.

   Kept deliberately strict in the same places pydantic is: every model there
   sets extra="forbid", so an unknown key is a rejection, not a warning. */
(function (global) {
  'use strict';

  var SELECT_VALUES = ['largest', 'most_centered', 'highest_conf', 'locked'];
  var MODES = ['follow', 'center'];
  var SPEC_KEYS = ['spec_id', 'targets', 'mode', 'arbitration', 'model'];
  var TARGET_KEYS = ['ref', 'detect', 'verify', 'relate', 'select'];
  var VERIFY_KEYS = ['class', 'text', 'min_score'];
  var RELATE_KEYS = ['keep', 'if_contains'];
  var ARBITRATION_KEYS = ['alternate_s'];

  /* Pipeline-side defaults, from perception/contracts.py. Worth agreeing with:
     a spec that omits these gets these. */
  var DEFAULTS = {
    mode: 'follow',
    select: 'largest',
    model: 'yoloe',
    min_score: 0.6,
    alternate_s: 3.0
  };

  function isObject(v) {
    return v !== null && typeof v === 'object' && !Array.isArray(v);
  }

  function isNonEmptyString(v) {
    return typeof v === 'string' && v.trim().length > 0;
  }

  function forbidExtra(obj, allowed, path, err) {
    Object.keys(obj).forEach(function (k) {
      if (allowed.indexOf(k) === -1) {
        err(path ? path + '.' + k : k, 'not in the contract — pydantic sets extra="forbid", so this is a 422');
      }
    });
  }

  function validateVerify(v, path, err, warn) {
    if (!isObject(v)) { err(path, 'must be an object'); return; }
    // Verify.cls carries alias="class", so the wire field is "class".
    if (!isNonEmptyString(v['class'])) err(path + '.class', 'required non-empty string');
    if (!isNonEmptyString(v.text)) err(path + '.text', 'required non-empty string');
    // The pipeline scores softmax over [text, "a {class}"]. Identical strings
    // score 0.5 by construction and can never clear min_score, so the target is
    // detected and then discarded on every frame.
    if (isNonEmptyString(v.text) && isNonEmptyString(v['class'])) {
      var bare = function (s) {
        return s.toLowerCase().replace(/^(a|an|the)\s+/, '').replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, ' ').trim();
      };
      if (bare(v.text) === bare(v['class'])) {
        err(path, 'text and class are the same words — CLIP contrasts them against each '
          + 'other, so this scores 0.5 and discards every detection. Drop verify, or make '
          + 'class the bare noun.');
      }
    }

    if (v.min_score !== undefined) {
      if (typeof v.min_score !== 'number' || Number.isNaN(v.min_score)) {
        err(path + '.min_score', 'must be a number');
      } else if (v.min_score < 0 || v.min_score > 1) {
        err(path + '.min_score', 'must be between 0 and 1');
      } else if (v.min_score < 0.4) {
        warn(path + '.min_score', v.min_score + ' is well under the pipeline default of '
          + DEFAULTS.min_score + ' — the CLIP check will admit false positives');
      }
    }
    forbidExtra(v, VERIFY_KEYS, path, err);
  }

  function validateRelate(r, path, err) {
    if (!isObject(r)) { err(path, 'must be an object'); return; }
    if (!isNonEmptyString(r.keep)) err(path + '.keep', 'required non-empty string');
    if (!isNonEmptyString(r.if_contains)) err(path + '.if_contains', 'required non-empty string');
    forbidExtra(r, RELATE_KEYS, path, err);
  }

  function validateTarget(t, path, err, warn) {
    if (!isObject(t)) { err(path, 'must be an object'); return; }

    if (!isNonEmptyString(t.ref)) err(path + '.ref', 'required non-empty string');

    if (!Array.isArray(t.detect)) {
      err(path + '.detect', 'required array of prompt strings');
    } else if (t.detect.length < 1 || t.detect.length > 6) {
      err(path + '.detect', 'must hold 1-6 entries (got ' + t.detect.length + ')');
    } else {
      t.detect.forEach(function (d, i) {
        if (!isNonEmptyString(d)) err(path + '.detect[' + i + ']', 'must be a non-empty string');
      });
    }

    if (t.verify !== undefined && t.verify !== null) validateVerify(t.verify, path + '.verify', err, warn);
    if (t.relate !== undefined && t.relate !== null) validateRelate(t.relate, path + '.relate', err);

    if (t.select !== undefined && SELECT_VALUES.indexOf(t.select) === -1) {
      err(path + '.select', 'must be one of ' + SELECT_VALUES.join(' | '));
    }
    if (t.select === 'locked') {
      warn(path + '.select', '"locked" holds the current track — only valid once a target is already acquired');
    }

    // The pipeline detects on prompt_union(), which is detect plus
    // relate.if_contains. Naming a relate noun that is not in detect still
    // works, but it is a sign the compile went sideways.
    if (t.relate && Array.isArray(t.detect)) {
      if (t.detect.indexOf(t.relate.keep) === -1) {
        warn(path + '.relate.keep', '"' + t.relate.keep + '" is not in detect — the kept box is never detected');
      }
    }

    forbidExtra(t, TARGET_KEYS, path, err);
  }

  /* opts.models — names from GET /models. When supplied, an unknown model is an
     error, because POST /spec rejects it with a 422 before touching a frame. */
  function validate(spec, opts) {
    var errors = [];
    var warnings = [];
    var known = (opts && opts.models) || null;
    function err(path, msg) { errors.push({ path: path, msg: msg }); }
    function warn(path, msg) { warnings.push({ path: path, msg: msg }); }

    if (!isObject(spec)) {
      return { ok: false, errors: [{ path: '$', msg: 'not a JSON object' }], warnings: warnings };
    }

    if (!isNonEmptyString(spec.spec_id)) err('spec_id', 'required non-empty string');
    if (spec.mode !== undefined && MODES.indexOf(spec.mode) === -1) {
      err('mode', 'must be ' + MODES.join(' | '));
    }

    // model is a free string checked against the live registry, not a Literal.
    if (spec.model !== undefined) {
      if (!isNonEmptyString(spec.model)) {
        err('model', 'must be a non-empty string');
      } else if (known && known.length && known.indexOf(spec.model) === -1) {
        err('model', '"' + spec.model + '" is not in the registry — available: ' + known.join(', '));
      }
    }

    if (!Array.isArray(spec.targets)) {
      err('targets', 'required array');
    } else if (spec.targets.length < 1 || spec.targets.length > 2) {
      err('targets', 'must hold 1-2 targets (got ' + spec.targets.length + ')');
    } else {
      spec.targets.forEach(function (t, i) { validateTarget(t, 'targets[' + i + ']', err, warn); });
    }

    // arbitration has a default_factory, so omitting it is legal and means 3.0.
    if (spec.arbitration !== undefined && spec.arbitration !== null) {
      if (!isObject(spec.arbitration)) {
        err('arbitration', 'must be an object');
      } else {
        if (spec.arbitration.alternate_s !== undefined) {
          if (typeof spec.arbitration.alternate_s !== 'number' || Number.isNaN(spec.arbitration.alternate_s)) {
            err('arbitration.alternate_s', 'must be a number');
          } else if (spec.arbitration.alternate_s <= 0) {
            err('arbitration.alternate_s', 'must be greater than 0');
          }
        }
        forbidExtra(spec.arbitration, ARBITRATION_KEYS, 'arbitration', err);
      }
    }

    forbidExtra(spec, SPEC_KEYS, '', err);

    return { ok: errors.length === 0, errors: errors, warnings: warnings };
  }

  /* Spoken/label summary: "following the person in red shoes and the dog" */
  function summarize(spec) {
    if (!isObject(spec) || !Array.isArray(spec.targets)) return 'no objective';
    var refs = spec.targets
      .map(function (t) { return (t && t.ref) || 'unknown'; })
      .join(' and ');
    return (spec.mode === 'center' ? 'tracking ' : 'following ') + refs;
  }

  /* Contract key order, so the JSON always reads the same way on the projector. */
  function reorder(spec) {
    if (!isObject(spec)) return spec;
    var out = {};
    SPEC_KEYS.forEach(function (k) { if (spec[k] !== undefined) out[k] = spec[k]; });
    Object.keys(spec).forEach(function (k) { if (out[k] === undefined) out[k] = spec[k]; });
    if (Array.isArray(out.targets)) {
      out.targets = out.targets.map(function (t) {
        if (!isObject(t)) return t;
        var o = {};
        TARGET_KEYS.forEach(function (k) { if (t[k] !== undefined) o[k] = t[k]; });
        Object.keys(t).forEach(function (k) { if (o[k] === undefined) o[k] = t[k]; });
        return o;
      });
    }
    return out;
  }

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* Minimal JSON syntax highlighter — no dependency, no build step. */
  function highlight(value) {
    var text = escapeHtml(JSON.stringify(reorder(value), null, 2));
    return text.replace(
      /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
      function (match) {
        var cls = 'j-num';
        if (/^"/.test(match)) {
          cls = /:$/.test(match) ? 'j-key' : 'j-str';
        } else if (/true|false/.test(match)) {
          cls = 'j-bool';
        } else if (/null/.test(match)) {
          cls = 'j-null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
      }
    );
  }

  global.TaskSpec = {
    DEFAULTS: DEFAULTS,
    validate: validate,
    summarize: summarize,
    reorder: reorder,
    highlight: highlight,
    SELECT_VALUES: SELECT_VALUES,
    MODES: MODES
  };
})(window);
