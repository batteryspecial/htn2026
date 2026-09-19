/* TaskSpec — client-side mirror of the frozen contract in /shared/schemas.py.
   Validates what the orchestrator returns so a hallucinated field shows up
   here, in red, instead of at demo time in the pipeline. */
(function (global) {
  'use strict';

  var SELECT_VALUES = ['largest', 'most_centered', 'highest_conf', 'locked'];
  var MODES = ['follow', 'center'];
  var SPEC_KEYS = ['spec_id', 'targets', 'mode', 'arbitration', 'model'];
  var TARGET_KEYS = ['ref', 'detect', 'verify', 'relate', 'select'];

  function isObject(v) {
    return v !== null && typeof v === 'object' && !Array.isArray(v);
  }

  function isNonEmptyString(v) {
    return typeof v === 'string' && v.trim().length > 0;
  }

  function validateVerify(v, path, err) {
    if (!isObject(v)) { err(path, 'must be an object'); return; }
    // Verify.cls carries alias="class", so the wire field is "class".
    if (!isNonEmptyString(v['class'])) err(path + '.class', 'required non-empty string');
    if (!isNonEmptyString(v.text)) err(path + '.text', 'required non-empty string');
    if (v.min_score !== undefined) {
      if (typeof v.min_score !== 'number' || Number.isNaN(v.min_score)) {
        err(path + '.min_score', 'must be a number');
      } else if (v.min_score < 0 || v.min_score > 1) {
        err(path + '.min_score', 'must be between 0 and 1');
      }
    }
  }

  function validateRelate(r, path, err) {
    if (!isObject(r)) { err(path, 'must be an object'); return; }
    if (!isNonEmptyString(r.keep)) err(path + '.keep', 'required non-empty string');
    if (!isNonEmptyString(r.if_contains)) err(path + '.if_contains', 'required non-empty string');
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

    if (t.verify !== undefined && t.verify !== null) validateVerify(t.verify, path + '.verify', err);
    if (t.relate !== undefined && t.relate !== null) validateRelate(t.relate, path + '.relate', err);

    if (t.select !== undefined && SELECT_VALUES.indexOf(t.select) === -1) {
      err(path + '.select', 'must be one of ' + SELECT_VALUES.join(' | '));
    }
    if (t.select === 'locked') {
      warn(path + '.select', '"locked" holds the current track — only valid once a target is already acquired');
    }

    Object.keys(t).forEach(function (k) {
      if (TARGET_KEYS.indexOf(k) === -1) warn(path + '.' + k, 'not in the contract — the pipeline will drop it');
    });
  }

  function validate(spec) {
    var errors = [];
    var warnings = [];
    function err(path, msg) { errors.push({ path: path, msg: msg }); }
    function warn(path, msg) { warnings.push({ path: path, msg: msg }); }

    if (!isObject(spec)) {
      return { ok: false, errors: [{ path: '$', msg: 'not a JSON object' }], warnings: warnings };
    }

    if (!isNonEmptyString(spec.spec_id)) err('spec_id', 'required non-empty string');
    if (MODES.indexOf(spec.mode) === -1) err('mode', 'must be ' + MODES.join(' | '));
    if (spec.model !== 'yoloe') err('model', 'must be "yoloe"');

    if (!Array.isArray(spec.targets)) {
      err('targets', 'required array');
    } else if (spec.targets.length < 1 || spec.targets.length > 2) {
      err('targets', 'must hold 1-2 targets (got ' + spec.targets.length + ')');
    } else {
      spec.targets.forEach(function (t, i) { validateTarget(t, 'targets[' + i + ']', err, warn); });
    }

    if (spec.arbitration !== undefined && spec.arbitration !== null) {
      if (!isObject(spec.arbitration)) {
        err('arbitration', 'must be an object');
      } else if (typeof spec.arbitration.alternate_s !== 'number' || Number.isNaN(spec.arbitration.alternate_s)) {
        err('arbitration.alternate_s', 'must be a number');
      } else if (spec.arbitration.alternate_s <= 0) {
        err('arbitration.alternate_s', 'must be greater than 0');
      }
    } else if (Array.isArray(spec.targets) && spec.targets.length === 2) {
      warn('arbitration', 'two targets and no arbitration — the pipeline falls back to alternate_s 5.0');
    }

    Object.keys(spec).forEach(function (k) {
      if (SPEC_KEYS.indexOf(k) === -1) warn(k, 'not in the contract — the pipeline will drop it');
    });

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
    validate: validate,
    summarize: summarize,
    reorder: reorder,
    highlight: highlight,
    SELECT_VALUES: SELECT_VALUES,
    MODES: MODES
  };
})(window);
