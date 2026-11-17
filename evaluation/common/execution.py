"""Timeout-isolated code evaluation for self-contained solve() programs.

The process boundary and import allowlist are resource controls, not a security
sandbox. Execute model programs in an isolated environment without credentials.
"""
import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import time

ALLOWED_MODULES = {'ast', 'bisect', 'calendar', 'collections', 'datetime', 'decimal',
                   'fractions', 'functools', 'heapq', 'itertools', 'json', 'math',
                   'operator', 're', 'statistics', 'string', 'typing'}


def execute(code, timeout=30):
    if not isinstance(code, str) or not code.strip():
        return dict(success=False, error='missing_code')
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or '').split('.')[0]]
            else:
                continue
            if any(name not in ALLOWED_MODULES for name in names):
                return dict(success=False, error='unsupported_import')
        result = subprocess.run([sys.executable, '-I', __file__], input=json.dumps(dict(code=code)),
                                text=True, capture_output=True, timeout=timeout,
                                env={k: os.environ[k] for k in ['PATH', 'LANG', 'TMPDIR', 'SYSTEMROOT'] if k in os.environ})
        if result.returncode:
            return dict(success=False, error='worker_failed', details=result.stderr[-1000:])
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return dict(success=False, error='timeout')
    except (ValueError, SyntaxError) as error:
        return dict(success=False, error=type(error).__name__, details=str(error))


def worker():
    import builtins
    payload = json.load(sys.stdin)
    original_import = builtins.__import__
    def limited_import(name, *args, **kwargs):
        if name.split('.')[0] not in ALLOWED_MODULES:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)
    namespace = {'__builtins__': {k: v for k, v in vars(builtins).items()
                  if k not in {'open', 'eval', 'exec', 'compile', 'input', 'breakpoint'}}}
    namespace['__builtins__']['__import__'] = limited_import
    started = time.monotonic()
    try:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
        except (ImportError, ValueError, OSError):
            pass
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(payload['code'], namespace)
            result = namespace['solve']()
        json.dump(dict(success=True, result=result, elapsed_seconds=time.monotonic() - started), sys.stdout)
    except Exception as error:
        json.dump(dict(success=False, error=type(error).__name__, details=str(error)), sys.stdout)


if __name__ == '__main__':
    worker()
