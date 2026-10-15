import json, sys, os, re

src_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
KEY_PAT = re.compile(r'''(api[_-]?key|token|secret|apikey|QiskitRuntimeService\.save_account|IBMQ\.save_account|channel=)''', re.I)
LONGSTR = re.compile(r'''["']([A-Za-z0-9_\-]{40,})["']''')
MAXOUT = 3000

def redact(s):
    return LONGSTR.sub(lambda m: '"<REDACTED_LONG_STRING>"', s)

for fn in sorted(os.listdir(src_dir)):
    if not fn.endswith('.ipynb'):
        continue
    p = os.path.join(src_dir, fn)
    try:
        nb = json.load(open(p, encoding='utf-8'))
    except Exception as e:
        print(fn, 'ERR', e); continue
    lines = []
    keycells = []
    for i, c in enumerate(nb.get('cells', [])):
        src = ''.join(c.get('source', []))
        if KEY_PAT.search(src) and LONGSTR.search(src):
            keycells.append(i)
        lines.append(f"\n===== CELL {i} [{c.get('cell_type')}] =====")
        lines.append(redact(src))
        outs = []
        for o in c.get('outputs', []) or []:
            t = o.get('output_type')
            if t == 'stream':
                outs.append(''.join(o.get('text', [])))
            elif t in ('execute_result', 'display_data'):
                d = o.get('data', {})
                if 'text/plain' in d:
                    outs.append(''.join(d['text/plain']))
                elif 'image/png' in d:
                    outs.append('<image/png>')
                else:
                    outs.append('<' + ','.join(d.keys()) + '>')
            elif t == 'error':
                outs.append('ERROR: ' + o.get('ename', '') + ': ' + o.get('evalue', '')[:500])
        if outs:
            o = '\n'.join(outs)
            if len(o) > MAXOUT:
                o = o[:MAXOUT // 2] + f'\n...[{len(o)} chars truncated]...\n' + o[-MAXOUT // 2:]
            lines.append('----- OUTPUT -----')
            lines.append(redact(o))
    txt = '\n'.join(lines)
    with open(os.path.join(out_dir, fn.replace('.ipynb', '.txt')), 'w', encoding='utf-8') as f:
        f.write(f"# {fn} cells={len(nb.get('cells', []))} keycells={keycells}\n")
        f.write(txt)
    print(f"{fn}\tcells={len(nb.get('cells', []))}\tchars={len(txt)}\tkeycells={keycells}")
