"""Restricted LaTeX math rendering using the installed TeX toolchain.

Rendering verifies supported syntax, never mathematical correctness. No Paddle
or Torch import occurs here. Unknown commands are rejected before TeX sees them.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tempfile

# Mathematical notation and inert text only. In particular no macro definition,
# file I/O, URL, package loading, code evaluation, catcode or dynamic-name commands.
_COMMANDS = set('''frac dfrac tfrac sqrt left right middle big Big bigg Bigg bigl bigr Bigl Bigr
biggl biggr Biggl Biggr overline underline widehat widetilde hat tilde bar vec dot ddot
acute grave breve check mathring overrightarrow overleftarrow
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu
nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
sum prod coprod int iint iiint oint lim limsup liminf max min sup inf
sin cos tan cot sec csc sinh cosh tanh log ln exp det gcd mod bmod pmod
arcsin arccos arctan arg deg dim ker hom Pr
infty partial nabla ell hbar imath jmath Re Im wp
pm mp times div cdot ast star circ bullet oplus ominus otimes oslash odot
le leq ge geq neq ne approx sim simeq equiv cong propto ll gg prec succ preceq succeq
in notin ni subset supset subseteq supseteq nsubseteq nsupseteq cup cap setminus emptyset varnothing
forall exists nexists neg lnot land lor wedge vee
rightarrow leftarrow leftrightarrow Rightarrow Leftarrow Leftrightarrow to mapsto longrightarrow
longleftarrow longleftrightarrow Longrightarrow Longleftarrow Longleftrightarrow implies iff
uparrow downarrow updownarrow Uparrow Downarrow Updownarrow
ldots cdots vdots ddots dots colon mid parallel perp angle triangle degree
langle rangle lvert rvert lVert rVert vert Vert lbrace rbrace backslash
quad qquad thinspace negthinspace enspace
text textrm textnormal textbf textit mathrm mathbf mathit mathsf mathtt mathcal mathbb mathfrak
operatorname boldsymbol binom dbinom tbinom overset underset substack
begin end displaystyle textstyle scriptstyle scriptscriptstyle nonumber notag
hline cline multicolumn cases matrix pmatrix bmatrix Bmatrix vmatrix Vmatrix aligned gathered array
underbrace overbrace not vphantom hphantom phantom limits nolimits'''.split())
_ENVIRONMENTS = {'aligned', 'alignedat', 'gathered', 'cases', 'matrix', 'pmatrix',
                 'bmatrix', 'Bmatrix', 'vmatrix', 'Vmatrix', 'smallmatrix', 'array'}
_SINGLE = set('\\{},;! :|_%#&')
_POLICY_VERSION = 1


def validate_latex(latex):
    """Return an error or None; accept a deliberately bounded math-only subset."""
    if not isinstance(latex, str) or not latex.strip():
        return 'LaTeX must be a nonempty string'
    if len(latex) > 6000:
        return 'LaTeX exceeds the 6000-character limit'
    if any(ord(c) < 32 and c not in '\n\r\t' for c in latex) or '^^' in latex:
        return 'Control characters and TeX character-code escapes are unsupported'
    # Do not permit a model string to leave the math wrapper or comment it out.
    stripped = re.sub(r'\\[%#$&_]','',latex)
    if any(c in stripped for c in '$%#'):
        return 'Unescaped math delimiters, comments, and parameter tokens are unsupported'
    depth = 0
    for match in re.finditer(r'\\.|[{}]', latex):
        token = match.group()
        if token == '{':
            depth += 1
            if depth > 40: return 'LaTeX nesting exceeds 40 levels'
        elif token == '}':
            depth -= 1
            if depth < 0: return 'Unbalanced braces'
    if depth:
        return 'Unbalanced braces'
    for match in re.finditer(r'\\([A-Za-z]+|.)', latex):
        command = match.group(1)
        if command not in _COMMANDS and command not in _SINGLE:
            return f'Unsupported LaTeX command: \\{command}'
    environments = []
    for match in re.finditer(r'\\(begin|end)\s*\{([^{}]+)\}', latex):
        action, name = match.groups()
        if name not in _ENVIRONMENTS:
            return f'Unsupported LaTeX environment: {name}'
        if action == 'begin': environments.append(name)
        elif not environments or environments.pop() != name:
            return 'Mismatched LaTeX environments'
    if environments:
        return 'Unclosed LaTeX environment'
    return None


@lru_cache(maxsize=1)
def engine_info():
    tex, raster = shutil.which('pdflatex'), shutil.which('pdftoppm')
    if not tex or not raster:
        return None
    try:
        version = subprocess.run([tex, '--version'], capture_output=True, text=True,
                                 timeout=5, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        return None
    return {'tex': tex, 'raster': raster, 'version': version}


def _limits():
    # Linux CPU runtime: bound maliciously expensive input even after validation.
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_AS, (768*1024*1024, 768*1024*1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (24*1024*1024, 24*1024*1024))


def _failure(status, error, output, **extra):
    return {'status': status, 'engine': 'pdflatex+pdftoppm', 'path': None,
            'error': str(error), 'cache_hit': False, 'semantic_correctness': 'not evaluated', **extra}


def render_latex(latex: str, output: Path) -> dict:
    """Atomically produce a PNG; failed/unsupported input never replaces output.

    The supported math subset includes aligned/cases/matrices/fractions. Unknown
    commands remain available as literal source in JSON/PDF, rather than being
    silently rewritten to a different expression.
    """
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('render_latex output must be a PNG path')
    error = validate_latex(latex)
    if error:
        return _failure('unsupported', error, output)
    engine = engine_info()
    if engine is None:
        return _failure('unsupported', 'pdflatex and pdftoppm are required; no renderer was installed automatically', output)
    key = hashlib.sha256(json.dumps({'latex': latex, 'engine': engine,
                                    'policy': _POLICY_VERSION, 'dpi': 180}, sort_keys=True).encode()).hexdigest()
    sidecar = output.with_suffix('.render.json')
    if output.is_file() and sidecar.is_file():
        try:
            record = json.loads(sidecar.read_text())
            if (record.get('key') == key and record.get('status') == 'rendered'
                    and record.get('png_sha256') == hashlib.sha256(output.read_bytes()).hexdigest()):
                return {**record, 'cache_hit': True, 'path': str(output.resolve())}
        except (ValueError, OSError):
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.latex-', dir=output.parent) as temporary:
        work = Path(temporary)
        source = work/'formula.tex'
        source.write_text('\\documentclass[border=4pt]{standalone}\n'
                          '\\usepackage{amsmath,amssymb}\n'
                          '\\begin{document}\n\\(\\displaystyle '+latex+'\\)\n\\end{document}\n', encoding='utf-8')
        env = os.environ.copy()
        env.update({'openin_any': 'p', 'openout_any': 'p', 'shell_escape': 'f',
                    'TEXMFOUTPUT': str(work), 'SOURCE_DATE_EPOCH': '0'})
        try:
            result = subprocess.run([engine['tex'], '-no-shell-escape', '-interaction=nonstopmode',
                                     '-halt-on-error', 'formula.tex'], cwd=work, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    timeout=12, preexec_fn=_limits)
            if result.returncode:
                details = '\n'.join(result.stdout.splitlines()[-18:])
                return _failure('failed', 'LaTeX compilation failed: '+details, output,
                                validation='supported tokens; syntax rejected by pdflatex')
            raster = subprocess.run([engine['raster'], '-singlefile', '-png', '-r', '180',
                                     'formula.pdf', 'formula'], cwd=work, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    timeout=12, preexec_fn=_limits)
            png = work/'formula.png'
            if raster.returncode or not png.exists():
                return _failure('failed', 'Formula rasterization failed: '+raster.stderr[-1200:], output)
            from PIL import Image
            with Image.open(png) as image:
                width, height = image.size
                if width*height > 12_000_000:
                    return _failure('failed', 'Rendered formula exceeds 12 million pixels', output)
                image.verify()
            os.replace(png, output)
            record = {'status': 'rendered', 'engine': 'pdflatex+pdftoppm', 'engine_version': engine['version'],
                      'path': str(output.resolve()), 'error': None, 'cache_hit': False, 'key': key,
                      'png_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'width': width, 'height': height, 'dpi': 180,
                      'semantic_correctness': 'not evaluated',
                      'validation': 'math command allowlist, balanced braces/environments, successful pdflatex compilation'}
            sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
            return record
        except (OSError, subprocess.SubprocessError) as error:
            return _failure('failed', f'{type(error).__name__}: {error}', output)
