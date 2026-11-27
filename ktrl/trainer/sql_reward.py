"""Read-only SQLite execution reward for SynSQL; no model or GPU dependencies."""

from collections import Counter, OrderedDict, deque
import json
import math
from pathlib import Path
import re
import sqlite3
import time

_FENCE = re.compile(r"```sql\s*(.*?)```", re.I | re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)


def sql_words(sql):
    """Unquoted SQL words with parenthesis depth; skip comments and quoted values."""
    i, depth = 0, 0
    while i < len(sql):
        if sql.startswith('--', i):
            end = sql.find('\n', i + 2)
            i = len(sql) if end < 0 else end + 1
        elif sql.startswith('/*', i):
            end = sql.find('*/', i + 2)
            i = len(sql) if end < 0 else end + 2
        elif sql[i] in "'\"`[":
            close = ']' if sql[i] == '[' else sql[i]
            i += 1
            while i < len(sql):
                if sql[i] == close:
                    if i + 1 < len(sql) and sql[i + 1] == close:
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    i += 1
        elif sql[i] == '(':
            depth += 1
            i += 1
        elif sql[i] == ')':
            depth -= 1
            i += 1
        elif sql[i].isalpha() or sql[i] == '_':
            end = i + 1
            while end < len(sql) and (sql[end].isalnum() or sql[end] in '_$'):
                end += 1
            yield sql[i:end].upper(), depth
            i = end
        else:
            i += 1


def has_top_level_order_by(sql):
    words = [word for word, depth in sql_words(sql) if depth == 0]
    return any(a == 'ORDER' and b == 'BY' for a, b in zip(words, words[1:]))


def extract_sql(text):
    """Use the final answer after thinking; never score an unfinished thought."""
    if '</think>' in text.lower():
        text = re.split(r'</think>', text, flags=re.I)[-1]
    elif re.search(r'<think>', text, re.I):
        return '', False
    answers = _ANSWER.findall(text)
    if answers:
        text = answers[-1].strip()
    fences = _FENCE.findall(text)
    sql = (fences[-1] if fences else text).strip()
    first = next(sql_words(sql), ('', 0))[0]
    if first not in ('SELECT', 'WITH'):
        return '', bool(fences or answers)
    return sql, bool(fences or answers)


def parse_ground_truth(ground_truth):
    value = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if not isinstance(value, dict) or not isinstance(value.get('db_id'), str) or not isinstance(value.get('sql'), str):
        raise ValueError('SynSQL ground_truth must contain string db_id and sql')
    db_id, sql = value['db_id'], value['sql'].strip()
    if not db_id or db_id in ('.', '..') or any(ch in db_id for ch in '/\\:\0') or not sql:
        raise ValueError('Invalid SynSQL db_id or reference SQL')
    return db_id, sql


class QueryFailure(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def execute_readonly(path, sql, *, timeout_seconds, max_rows, max_result_bytes):
    """Consume all rows with VM/time/size limits; deny writes, ATTACH and state-changing PRAGMA."""
    if not path.is_file():
        raise QueryFailure('missing_database', str(path))
    started = time.monotonic()
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    def authorize(action, arg1, arg2, database, trigger):
        if action == sqlite3.SQLITE_PRAGMA and (arg1 or '').lower() in ('table_info', 'table_xinfo'):
            return sqlite3.SQLITE_OK
        if action not in allowed:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or '').lower() in ('load_extension', 'readfile', 'writefile', 'random', 'randomblob'):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('PRAGMA trusted_schema=OFF')
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max_result_bytes)
        connection.set_authorizer(authorize)
        connection.set_progress_handler(lambda: int(time.monotonic() - started > timeout_seconds), 1000)
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise QueryFailure('not_query', 'SQL must return rows')
        columns = len(cursor.description)
        rows, size = [], 0
        while True:
            batch = cursor.fetchmany(256)
            if not batch:
                break
            rows.extend(batch)
            size += sum(len(v.encode('utf-8')) if isinstance(v, str) else len(v) if isinstance(v, bytes) else 8
                        for row in batch for v in row)
            if len(rows) > max_rows or size > max_result_bytes:
                raise QueryFailure('result_limit', 'SQL result exceeds row/byte limit')
            if time.monotonic() - started > timeout_seconds:
                raise QueryFailure('timeout', 'SQL execution time limit exceeded')
        return columns, rows, size
    except sqlite3.Error as error:
        status = 'timeout' if time.monotonic() - started > timeout_seconds else 'sql_error'
        raise QueryFailure(status, str(error)) from error
    finally:
        connection.close()


def results_equal(expected, actual, *, ordered, abs_tol=1e-6, rel_tol=1e-6, max_comparisons=1_000_000):
    """Preserve columns and duplicate rows; tolerate numeric rounding, not strings/NULL."""
    if len(expected) != len(actual):
        return False
    if expected == actual:
        return True
    if not ordered and Counter(expected) == Counter(actual):
        return True
    comparisons = 0
    def same_row(a, b):
        nonlocal comparisons
        comparisons += 1
        if comparisons > max_comparisons:
            raise QueryFailure('comparison_limit', 'SQL row matching comparison budget exceeded')
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if x == y:
                continue
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if math.isclose(x, y, abs_tol=abs_tol, rel_tol=rel_tol):
                    continue
            return False
        return True
    if ordered:
        return all(same_row(a, b) for a, b in zip(expected, actual))
    # Bipartite matching avoids the incorrect greedy treatment of close duplicates.
    edges = [[j for j, gold in enumerate(expected) if same_row(pred, gold)] for pred in actual]
    matched_left, matched_right = {}, {}
    for left in range(len(actual)):
        queue, seen, previous = deque([left]), {left}, {}
        endpoint = None
        while queue and endpoint is None:
            current = queue.popleft()
            for right in edges[current]:
                if right in previous:
                    continue
                previous[right] = current
                if right not in matched_right:
                    endpoint = right
                    break
                other = matched_right[right]
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        if endpoint is None:
            return False
        while endpoint is not None:
            current = previous[endpoint]
            old = matched_left.get(current)
            matched_left[current], matched_right[endpoint] = endpoint, current
            endpoint = old
    return True


class SqlReward:
    """Cache valid reference results within one scoring phase; candidates are always executed."""
    def __init__(self, config):
        self.root = Path(config['database_root']).resolve()
        if not self.root.is_dir():
            raise ValueError(f'SQL database_root does not exist: {self.root}')
        self.options = {key: config.get(key, default) for key, default in (
            ('timeout_seconds', 2.0), ('max_rows', 100000), ('max_result_bytes', 8388608))}
        self.abs_tol = config.get('float_abs_tol', 1e-6)
        self.rel_tol = config.get('float_rel_tol', 1e-6)
        for key, value in {**self.options, 'float_abs_tol': self.abs_tol, 'float_rel_tol': self.rel_tol}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid SQL reward option {key}')
        if any(self.options[k] <= 0 for k in self.options):
            raise ValueError('SQL query limits must be positive')
        if any(type(self.options[k]) is not int for k in ('max_rows', 'max_result_bytes')):
            raise ValueError('SQL row/byte limits must be integers')
        self.cache, self.cache_bytes = OrderedDict(), 0

    def __call__(self, text, ground_truth):
        db_id, sql = parse_ground_truth(ground_truth)
        path = (self.root / db_id / (db_id + '.sqlite')).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('SynSQL database path escapes configured root')
        key = db_id, sql
        reference = self.cache.get(key)
        if reference is None:
            try:
                reference = execute_readonly(path, sql, **self.options)
            except QueryFailure as error:
                raise ValueError(f'Invalid SynSQL reference for {db_id}: {error.status}: {error}') from error
            if not reference[1]:
                raise ValueError(f'Empty SynSQL reference result for {db_id}')
            self.cache[key] = reference
            self.cache_bytes += reference[2]
            while len(self.cache) > 128 or self.cache_bytes > 32 * 1024 * 1024:
                _, removed = self.cache.popitem(last=False)
                self.cache_bytes -= removed[2]
        else:
            self.cache.move_to_end(key)
        pred, has_format = extract_sql(text)
        ordered = has_top_level_order_by(sql)
        base = dict(score=0.0, acc=False, execution_match=False, pred=pred, has_format=has_format,
                    ordered=ordered, reference_rows=len(reference[1]), candidate_rows=None)
        if not pred:
            return dict(base, candidate_status='no_sql', error='No final SELECT/WITH query')
        try:
            columns, rows, _ = execute_readonly(path, pred, **self.options)
            matched = columns == reference[0] and results_equal(reference[1], rows, ordered=ordered,
                                                               abs_tol=self.abs_tol, rel_tol=self.rel_tol)
        except QueryFailure as error:
            return dict(base, candidate_status=error.status, error=str(error)[:256])
        base.update(score=float(matched), acc=matched, execution_match=matched, candidate_rows=len(rows))
        return dict(base, candidate_status='match' if matched else 'mismatch', error='')
